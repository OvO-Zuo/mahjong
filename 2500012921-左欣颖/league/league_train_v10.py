"""
v10 — SL-Constrained RL Training.

Core philosophy:
  SL defines the behaviour floor.
  RL is only allowed to improve when it provably beats SL.

Three-tier experience buffer:
  50% SL anchor — fixed high weight, defines behaviour distribution
  30% SL-mistake — states where RL previously failed vs SL, priority zone
  20% RL self-play — expands state coverage only

Loss structure:
  SL loss — always active (primary constraint)
  RL loss — only on SL-beating samples (gradient masked otherwise)
  KL — hard constraint (rate reduction / pause / rollback)

Stop condition:
  SL win rate >= 0.55 (rolling, stable)
  worst-case win >= SL baseline + margin
  KL stable < 0.12
"""

import os, sys, json, time, math
import torch, torch.nn.functional as F
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RL'))
from model import CNNModel
from feature import FeatureAgent
from env import MahjongGBEnv
from elo import EloRanker
from league_model_pool import LeagueModelPool


# ======================================================================
# Config
# ======================================================================
CFG = {
    # --- PPO ---
    'lr': 3e-5,                         # Lower: gentle improvements only
    'clip': 0.1,
    'gamma': 0.98, 'gae_lambda': 0.95,
    'value_coeff': 0.3,                 # Further reduced
    'ppo_epochs': 2,                    # Minimal RL epochs
    'batch_size': 256,
    'min_buffer_size': 512,

    # --- Training scale ---
    'total_episodes': 2000,
    'eval_interval': 200,
    'eval_games': 100,
    'print_interval': 50,
    'ckpt_interval': 300,

    # --- SL as behavior lower bound ---
    'sl_lower_bound_eps': 0.1,         # logπ(a|s) ≥ logπ_SL(a|s) - ε
    'sl_bound_enforce': True,          # Mask RL gradient when bound violated

    # --- Monotonic Improvement Gate ---
    'monotonic_window': 100,           # Rolling window for ΔSL_win_rate check
    'monotonic_min_delta': -0.02,      # Allow tiny regression (noise tolerance)

    # --- KL trust region (tight) ---
    'kl_normal': 0.08,                 # KL ≤ this → normal training
    'kl_reduce_rl': 0.10,              # KL > this → reduce RL weight
    'kl_freeze_rl': 0.12,              # KL > this → freeze RL, SL only
    'kl_rollback': 0.18,               # KL > this → restore checkpoint
    'kl_anchor_coeff': 0.03,           # Lower anchor coeff (KL is hard gate now)

    # --- SL-beating gate (RL only on SL error states) ---
    'sl_beat_warmup': 200,             # First N episodes: RL on all samples

    # --- Three-tier buffer ---
    'sl_anchor_buffer_size': 15000,
    'sl_failure_buffer_size': 10000,
    'rl_buffer_size': 5000,            # Reduced: coverage only
    'sl_anchor_ratio': 0.50,           # 50% SL anchor (permanent, high priority)
    'sl_failure_ratio': 0.35,          # 35% SL failure (primary RL optimization zone)
    'rl_selfplay_ratio': 0.15,         # 15% RL self-play (distribution coverage only)

    # --- Self-play (minimal) ---
    'selfplay_ratio': 0.15,            # Further reduced
    'sl_anchor_opponent_ratio': 0.55,  # SL as primary opponent
    'hard_ratio': 0.20,

    # --- Stop condition ---
    'stop_sl_win_rate': 0.58,
    'stop_worst_case': 0.50,
    'stop_kl_max': 0.10,
    'stop_rolling_window': 200,
    'stop_max_regression_ep': 100,     # No regression >100ep allowed

    # --- Exploit (evaluation only, no training gradient) ---
    'exploit_eval_only': True,

    # --- Entropy ---
    'entropy_coeff': 0.05,
    'entropy_floor': 0.06,             # Hard floor
    'entropy_temperature_scale': 0.5,  # Increase temperature when below floor

    # --- Paths ---
    'sl_model_path': os.path.join(os.path.dirname(__file__), '..', 'SL', 'model',
                                   'checkpoint', 'model_20.pt'),
    'ckpt_dir': os.path.join(os.path.dirname(__file__), 'league_checkpoints_v10'),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# ======================================================================
# Three-tier Buffer
# ======================================================================
class TieredBuffer:
    def __init__(self, cfg):
        self.sl_anchor = deque(maxlen=cfg['sl_anchor_buffer_size'])
        self.sl_failure = deque(maxlen=cfg['sl_failure_buffer_size'])
        self.rl_selfplay = deque(maxlen=cfg['rl_buffer_size'])

    def push_sl_anchor(self, sample):
        self.sl_anchor.append(sample)

    def push_sl_failure(self, sample):
        self.sl_failure.append(sample)

    def push_rl(self, sample):
        self.rl_selfplay.append(sample)

    def sample(self, batch_size, cfg):
        n_sl_anchor = int(batch_size * cfg['sl_anchor_ratio'])
        n_sl_failure = int(batch_size * cfg['sl_failure_ratio'])
        n_rl = batch_size - n_sl_anchor - n_sl_failure

        samples = []
        for buf, n in [(self.sl_anchor, n_sl_anchor),
                        (self.sl_failure, n_sl_failure),
                        (self.rl_selfplay, n_rl)]:
            if n > 0 and len(buf) > 0:
                idx = np.random.choice(len(buf), min(n, len(buf)), replace=False)
                for i in idx:
                    samples.append(buf[i])

        if len(samples) < 32:
            return None

        np.random.shuffle(samples)
        obs = torch.from_numpy(np.stack([s['observation'] for s in samples]).astype(np.float32))
        masks = torch.from_numpy(np.stack([s['action_mask'] for s in samples]).astype(np.float32))
        acts = torch.tensor([s['action'] for s in samples], dtype=torch.long)
        advs = torch.tensor([s.get('advantage', 0) for s in samples], dtype=torch.float32)
        tgts = torch.tensor([s.get('target', 0) for s in samples], dtype=torch.float32)
        old_logps = torch.tensor([s.get('old_logp', 0) for s in samples], dtype=torch.float32)
        tiers = [s.get('tier', 'rl') for s in samples]
        return obs, masks, acts, advs, tgts, old_logps, tiers

    def total_size(self):
        return len(self.sl_anchor) + len(self.sl_failure) + len(self.rl_selfplay)


# ======================================================================
# SL-Anchored Game Runner
# ======================================================================
def run_game_sl_anchored(model, sl_model, cfg):
    """Run game with SL anchor as primary opponent for coverage."""
    device = cfg['device']

    # Opponents: SL anchor fills seats 1-3 (coverage-focused)
    opp_nets = [sl_model] * 3 if sl_model else [model] * 3
    models = [model] + opp_nets

    env_config = {'agent_clz': FeatureAgent, 'duplicate': True, 'variety': 10000}
    env = MahjongGBEnv(env_config)
    obs_dict = env.reset()

    agents = [FeatureAgent(i) for i in range(4)]
    for i in range(4):
        agents[i].request2obs('Wind %d' % (i % 4))

    trajectory = []
    done = False
    turns = 0
    while not done and turns < 500:
        actions = {}
        for name in env.agent_names:
            i = int(name.split('_')[1]) - 1
            obs = obs_dict.get(name)
            if obs is not None:
                action = model_infer(models[i], obs, agents[i], device, argmax=(i != 0))
                if i == 0 and obs is not None:
                    with torch.no_grad():
                        obs_t = torch.from_numpy(np.expand_dims(obs['observation'], 0)).float().to(device)
                        mask_t = torch.from_numpy(np.expand_dims(obs['action_mask'], 0)).float().to(device)
                        logits, value = model({'observation': obs_t, 'action_mask': mask_t})
                        probs = F.softmax(logits, dim=-1)
                        log_probs = F.log_softmax(logits, dim=-1)
                        old_logp = log_probs[0, action].item()

                        # Compare RL vs SL: probability margin on chosen action
                        sl_out = sl_model({'observation': obs_t, 'action_mask': mask_t})
                        sl_logits = sl_out[0] if isinstance(sl_out, tuple) else sl_out
                        sl_probs = F.softmax(sl_logits, dim=-1)
                        sl_action = int(sl_logits[0].argmax().item())
                        rl_prob = probs[0, action].item()
                        sl_prob = sl_probs[0, action].item()

                    trajectory.append({
                        'observation': obs['observation'].copy(),
                        'action_mask': obs['action_mask'].copy(),
                        'action': action,
                        'sl_action': sl_action,
                        'value': value.item(),
                        'old_logp': old_logp,
                        'reward': 0.0,
                        'prob_margin': rl_prob - sl_prob,
                        'sl_beats_rl': False,
                    })
                actions[name] = action

        if not actions: break
        obs_dict, reward_dict, done_dict = env.step(actions)
        if trajectory:
            trajectory[-1]['reward'] = reward_dict.get('player_1', 0)
        done = done_dict
        turns += 1

    # Determine winner
    winner_seat = None
    for name, r in reward_dict.items():
        if r > 0:
            winner_seat = int(name.split('_')[1]) - 1
            break
    fan = 0
    if winner_seat is not None:
        r = reward_dict.get(f'player_{winner_seat+1}', 0)
        if r > 0: fan = max(0, int(r / 3 - 8))

    # GAE + SL-beating check
    if trajectory:
        gae = 0.0
        for t in reversed(range(len(trajectory))):
            step = trajectory[t]
            next_val = trajectory[t+1]['value'] if t+1 < len(trajectory) else 0.0
            delta = step['reward'] + cfg['gamma'] * next_val - step['value']
            gae = delta + cfg['gamma'] * cfg['gae_lambda'] * gae
            step['advantage'] = gae
            step['target'] = gae + step['value']

            # SL-beating: RL prob > SL prob on chosen action
            if step.get('prob_margin', -1) > -cfg['sl_lower_bound_eps']:
                step['sl_beats_rl'] = True
            elif step['action'] != step['sl_action'] and step.get('prob_margin', -1) > -0.2:
                step['sl_beats_rl'] = True   # Different action with reasonable prob

    return {'samples': trajectory, 'winner_seat': winner_seat, 'fan': fan,
            'turns': turns, 'hu': winner_seat is not None}


def model_infer(model, obs, agent, device, argmax=False):
    obs_t = torch.from_numpy(np.expand_dims(obs['observation'], 0)).float().to(device)
    mask_t = torch.from_numpy(np.expand_dims(obs['action_mask'], 0)).float().to(device)
    with torch.no_grad():
        output = model({'observation': obs_t, 'action_mask': mask_t})
        logits = output[0] if isinstance(output, tuple) else output
    if argmax: return int(logits[0].argmax().item())
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


# ======================================================================
# SL-Gated Update (v10.1 — SL lower bound + monotonic gate + KL trust region)
# ======================================================================
def sl_gated_update(model, sl_model, optimizer, buffer, cfg, ep, sl_win_history):
    batch = buffer.sample(cfg['batch_size'], cfg)
    if batch is None:
        return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl': 0,
                'rl_active': 0, 'sl_loss': 0, 'sl_bound_violations': 0}

    obs, masks, acts, advs, tgts, old_logps, tiers = batch
    device = cfg['device']
    obs, masks = obs.to(device), masks.to(device)
    acts = acts.to(device); advs = advs.to(device)
    tgts = tgts.to(device); old_logps = old_logps.to(device)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    is_sl_anchor = torch.tensor([t == 'sl_anchor' for t in tiers], device=device)
    is_sl_failure = torch.tensor([t == 'sl_failure' for t in tiers], device=device)
    is_rl = torch.tensor([t == 'rl' for t in tiers], device=device)
    in_warmup = ep <= cfg['sl_beat_warmup']

    # --- KL pre-check (trust region) ---
    with torch.no_grad():
        logits, _ = model({'observation': obs, 'action_mask': masks})
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        sl_out = sl_model({'observation': obs, 'action_mask': masks})
        sl_logits = sl_out[0] if isinstance(sl_out, tuple) else sl_out
        sl_log_probs = F.log_softmax(sl_logits, dim=-1)
        valid_mask = (masks > 0.5).float()
        pre_kl = (valid_mask * probs * (log_probs - sl_log_probs.detach())).sum(dim=-1).mean().item()

    # KL trust region gate
    if pre_kl > cfg['kl_rollback']:
        return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl': pre_kl,
                'rl_active': 0, 'sl_loss': 0, 'rollback': True}
    if pre_kl > cfg['kl_freeze_rl']:
        rl_allowed_kl = torch.zeros_like(acts, dtype=torch.bool)
    elif pre_kl > cfg['kl_reduce_rl']:
        rl_allowed_kl = (is_sl_failure & ~is_sl_anchor)  # Only SL-failure zone
    else:
        rl_allowed_kl = torch.ones_like(acts, dtype=torch.bool)

    # --- Monotonic Improvement Gate ---
    monotonic_ok = True
    if len(sl_win_history) >= cfg['monotonic_window']:
        recent = list(sl_win_history)[-cfg['monotonic_window']:]
        delta = recent[-1] - recent[0] if len(recent) >= 2 else 0
        monotonic_ok = delta >= cfg['monotonic_min_delta']
    if not monotonic_ok and not in_warmup:
        rl_allowed_kl = torch.zeros_like(acts, dtype=torch.bool)

    # --- SL-beating gate: RL only on SL-error states ---
    with torch.no_grad():
        sl_logp_sel = sl_log_probs.gather(1, acts.unsqueeze(1)).squeeze(1)
        rl_logp_sel = log_probs.gather(1, acts.unsqueeze(1)).squeeze(1)
        sl_logp_best = sl_log_probs.max(dim=-1).values
        sl_best_act = sl_logits.argmax(dim=-1)
        # SL error state: SL's optimal action != taken action AND RL improves
        sl_is_wrong = (sl_best_act != acts)
        rl_beats_sl = (rl_logp_sel - sl_logp_sel) > -cfg['sl_lower_bound_eps']
    rl_active_mask = (sl_is_wrong & rl_beats_sl) | in_warmup
    rl_active_mask = rl_active_mask & rl_allowed_kl

    # --- SL lower bound enforcement ---
    with torch.no_grad():
        violation = (rl_logp_sel - sl_logp_sel) < -cfg['sl_lower_bound_eps']
    sl_bound_violations = violation.sum().item()

    total_pl, total_vl, total_el, total_kl, total_sl = 0, 0, 0, 0, 0
    rl_active_count = 0

    for _ in range(cfg['ppo_epochs']):
        logits, values = model({'observation': obs, 'action_mask': masks})
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # SL loss (always active, primary constraint)
        sl_out2 = sl_model({'observation': obs, 'action_mask': masks})
        sl_model_logits = sl_out2[0] if isinstance(sl_out2, tuple) else sl_out2
        sl_targets = sl_model_logits.argmax(dim=-1).detach()
        sl_weight = torch.where(is_sl_failure, 1.5, 1.0)
        sl_loss = (sl_weight * F.cross_entropy(logits, sl_targets, reduction='none')).mean()

        # RL loss (SL-error states only)
        if rl_active_mask.any():
            rl_idx = rl_active_mask.nonzero(as_tuple=True)[0]
            if len(rl_idx) > 0:
                rl_logits_s, rl_values_s = model(
                    {'observation': obs[rl_idx], 'action_mask': masks[rl_idx]})
                rl_lp = F.log_softmax(rl_logits_s, dim=-1).gather(
                    1, acts[rl_idx].unsqueeze(1)).squeeze(1)
                ratio = torch.exp(rl_lp - old_logps[rl_idx])
                clip_adv = torch.clamp(ratio, 1-cfg['clip'], 1+cfg['clip']) * advs[rl_idx]
                policy_loss = -torch.min(ratio*advs[rl_idx], clip_adv).mean()
                value_loss = F.mse_loss(rl_values_s.squeeze(1), tgts[rl_idx])
                rl_active_count = len(rl_idx)
        if not rl_active_mask.any():
            policy_loss = torch.tensor(0.0, device=device)
            value_loss = torch.tensor(0.0, device=device)

        # KL
        sl_log_probs2 = F.log_softmax(sl_model_logits, dim=-1)
        valid_mask2 = (masks > 0.5).float()
        kl = (valid_mask2 * probs * (log_probs - sl_log_probs2.detach())).sum(dim=-1).mean()

        # Entropy floor: increase temperature when below floor
        entropy_bonus = (2.0 if entropy.item() < cfg['entropy_floor'] else 1.0)
        temp_scale = (1.0 + cfg['entropy_temperature_scale']
                       if entropy.item() < cfg['entropy_floor'] else 1.0)

        # Structured loss
        loss = (sl_loss +
                cfg['value_coeff'] * value_loss -
                cfg['entropy_coeff'] * entropy * entropy_bonus / temp_scale +
                cfg['kl_anchor_coeff'] * kl)
        if rl_active_mask.any():
            loss = loss + policy_loss

        # SL lower bound: mask gradient for violating samples
        if cfg['sl_bound_enforce'] and violation.any() and not in_warmup:
            pass  # Gradient already restricted via rl_active_mask

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_pl += policy_loss.item() if rl_active_mask.any() else 0
        total_vl += value_loss.item() if rl_active_mask.any() else 0
        total_el += entropy.item()
        total_kl += kl.item()
        total_sl += sl_loss.item()

    n = cfg['ppo_epochs']
    return {'policy_loss': total_pl/n, 'value_loss': total_vl/n, 'entropy': total_el/n,
            'kl': total_kl/n, 'sl_loss': total_sl/n, 'rl_active': rl_active_count,
            'sl_bound_violations': sl_bound_violations, 'monotonic_ok': monotonic_ok}


# ======================================================================
# Evaluation
# ======================================================================
def evaluate_vs_sl(model, sl_model, cfg, n_games=50):
    """Evaluate RL model vs SL anchor directly."""
    device = cfg['device']
    wins, total_fan, total_turns = 0, 0, 0
    for g in range(n_games):
        # Alternate seat 0
        if g % 2 == 0:
            seat0, seat1 = model, sl_model
            seat0_is_rl = True
        else:
            seat0, seat1 = sl_model, model
            seat0_is_rl = False
        opp_entries = [{'state_dict': {k: v.cpu().clone() for k, v in sl_model.state_dict().items()},
                         'model_id': 0}] * 3
        result = run_game_sl_anchored(seat0, sl_model, cfg)
        rl_won = (seat0_is_rl and result['winner_seat'] == 0) or \
                 (not seat0_is_rl and result['winner_seat'] != 0)
        if rl_won: wins += 1
        if result['hu']: total_fan += result['fan']
        total_turns += result['turns']
    return {'win_rate': wins / n_games, 'avg_fan': total_fan / max(wins, 1),
            'avg_turns': total_turns / n_games}


# ======================================================================
# Main Training
# ======================================================================
def train():
    cfg = CFG
    os.makedirs(cfg['ckpt_dir'], exist_ok=True)
    device = cfg['device']
    print(f'[v10] SL-Constrained RL. Device: {device}')

    # ---- SL anchor ----
    sl_model = CNNModel().to(device)
    sl_model.load_state_dict(torch.load(cfg['sl_model_path'], map_location=device, weights_only=False), strict=False)
    sl_model.eval()
    for p in sl_model.parameters(): p.requires_grad = False

    # ---- RL model (init from SL) ----
    model = CNNModel().to(device)
    model.load_state_dict(sl_model.state_dict(), strict=False)

    # ---- Buffer ----
    buffer = TieredBuffer(cfg)

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    # ---- Checkpoint for rollback ----
    best_ckpt_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_ckpt_ep = 0

    # ---- State ----
    total_games, best_sl_win, rollback_count = 0, 0.0, 0
    sl_win_history = deque(maxlen=cfg['stop_rolling_window'])

    for ep in range(1, cfg['total_episodes'] + 1):
        # 1. Run game (SL-anchored coverage)
        result = run_game_sl_anchored(model, sl_model, cfg)
        total_games += 1

        # 2. Classify samples into three tiers
        for s in result.get('samples', []):
            sample = {'observation': s['observation'], 'action_mask': s['action_mask'],
                      'action': s['action'], 'advantage': s.get('advantage', 0),
                      'target': s.get('target', 0), 'old_logp': s.get('old_logp', 0)}

            if s.get('sl_beats_rl') or ep <= cfg['sl_beat_warmup']:
                sample['tier'] = 'sl_failure'
                buffer.push_sl_failure(sample)
            # Always also push to RL self-play (coverage)
            rl_sample = dict(sample)
            rl_sample['tier'] = 'rl'
            buffer.push_rl(rl_sample)

            # SL anchor reference (always)
            sl_sample = dict(sample)
            sl_sample['action'] = s['sl_action']
            sl_sample['tier'] = 'sl_anchor'
            buffer.push_sl_anchor(sl_sample)

        # 3. SL-gated update
        ppo_stats = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl': 0,
                      'sl_loss': 0, 'rl_active': 0, 'rollback': False,
                      'sl_bound_violations': 0, 'monotonic_ok': True}
        if buffer.total_size() >= cfg['min_buffer_size']:
            ppo_stats = sl_gated_update(model, sl_model, optimizer, buffer, cfg, ep, sl_win_history)

            # KL rollback
            if ppo_stats.get('rollback'):
                model.load_state_dict(best_ckpt_sd)
                rollback_count += 1
                print(f'  [ROLLBACK] KL={ppo_stats["kl"]:.3f} > {cfg["kl_rollback"]} → restored ep{best_ckpt_ep}')
                continue

            # Update best checkpoint (if KL is healthy)
            if ppo_stats['kl'] < cfg['kl_normal']:
                best_ckpt_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_ckpt_ep = ep

        # 4. Evaluation every 200ep
        if ep % cfg['eval_interval'] == 0:
            eval_r = evaluate_vs_sl(model, sl_model, cfg, n_games=cfg['eval_games'])
            sl_win_history.append(eval_r['win_rate'])
            rolling_win = np.mean(list(sl_win_history)[-50:]) if len(sl_win_history) >= 50 else np.mean(sl_win_history)

            if eval_r['win_rate'] > best_sl_win:
                best_sl_win = eval_r['win_rate']
                torch.save({'model_state_dict': model.state_dict(), 'ep': ep,
                            'sl_win': best_sl_win}, os.path.join(cfg['ckpt_dir'], 'best_model.pt'))

            kl_val = ppo_stats['kl']
            mono = ppo_stats.get('monotonic_ok', True)
            viol = ppo_stats.get('sl_bound_violations', 0)
            print(f'\n[Eval ep{ep}] SL_win={eval_r["win_rate"]:.3f} rolling={rolling_win:.3f} best={best_sl_win:.3f} '
                  f'KL={kl_val:.4f} viol={viol} mono={mono} fan={eval_r["avg_fan"]:.0f}')

            # Stop condition
            if (rolling_win >= cfg['stop_sl_win_rate'] and len(sl_win_history) >= 100 and
                ppo_stats['kl'] < cfg['stop_kl_max']):
                print(f'[v10] STOP: rolling_SL_win={rolling_win:.3f} ≥ {cfg["stop_sl_win_rate"]}, KL={kl_val:.4f}')
                break

        # 5. Periodic stats
        if ep % cfg['print_interval'] == 0:
            s = ppo_stats
            kl_s = 'RB' if s.get('rollback') else f'{s["kl"]:.4f}'
            mono = '!' if not s.get('monotonic_ok', True) else ''
            print(f'[Ep {ep:5d}] g={total_games:5d} buf={buffer.total_size():5d} '
                  f'hu={result["hu"]} fan={result["fan"]:2d} '
                  f'sl={s.get("sl_loss",0):.3f} pl={s.get("policy_loss",0):.3f} '
                  f'kl={kl_s}{mono} ent={s.get("entropy",0):.3f} '
                  f'rl={s.get("rl_active",0)} v={s.get("sl_bound_violations",0)} rb={rollback_count}')

        # 6. Checkpoint
        if ep % cfg['ckpt_interval'] == 0:
            torch.save({'model_state_dict': model.state_dict(), 'ep': ep, 'sl_win': best_sl_win},
                       os.path.join(cfg['ckpt_dir'], f'ckpt_ep{ep}.pt'))

    # Final
    torch.save({'model_state_dict': model.state_dict(), 'ep': ep, 'sl_win': best_sl_win},
               os.path.join(cfg['ckpt_dir'], 'final_model.pt'))
    print(f'[v10] Done. best_sl_win={best_sl_win:.3f} games={total_games} rollbacks={rollback_count}')


if __name__ == '__main__':
    train()
