"""
League Self-Play Training — Integrated Version.

Fixes all issues from v1:
1. Opponents sampled from league pool each episode (real models, not copies)
2. Elo updated after every game with real results
3. KL computed on real game observations
4. Exploit detector fed with actual failure data
5. Evaluation runs on schedule with real matchups
6. total_games increments each episode
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RL'))

from model import CNNModel
from feature import FeatureAgent
from env import MahjongGBEnv

from elo import EloRanker
from league_model_pool import LeagueModelPool
from dynamic_sampler import DynamicSampler
from exploit_detector import ExploitDetector


# ======================================================================
# Config
# ======================================================================
CFG = {
    # --- PPO ---
    'lr': 5e-5,
    'clip': 0.1,
    'gamma': 0.98,
    'gae_lambda': 0.95,
    'value_coeff': 0.5,               # Halved: reduce critic noise amplification
    'entropy_coeff': 0.08,            # Raised: maintain exploration
    'ppo_epochs': 3,                  # Reduced: gentler updates
    'batch_size': 256,
    'min_buffer_size': 512,
    'adv_clip': 3.0,                  # Clip advantages to [-N, N] for stability

    # --- Training scale ---
    'total_episodes': 2000,
    'eval_interval': 200,
    'eval_games': 100,
    'print_interval': 50,
    'ckpt_interval': 300,
    'exploit_delay': 1000,             # Delay exploit until ep 1000+

    # --- Opponent sampling (reduced self-play, SL-dominant) ---
    'hard_ratio': 0.25,                # Reduced
    'selfplay_ratio': 0.25,            # Reduced
    'baseline_ratio': 0.50,            # SL + similar → primary opponent
    'exploiter_ratio': 0.0,            # Disabled until exploit_delay

    # --- Stability guards ---
    'kl_hard_limit': 0.18,            # KL above this → PAUSE RL update
    'kl_warn_threshold': 0.15,         # KL above this → reduce RL step size
    'lr_reduction_factor': 0.70,
    'entropy_critical': 0.035,         # Only trigger on real emergency

    # --- League ---
    'max_historical': 3,
    'initial_elo': 1500.0,
    'elo_k': 16,
    'margin_elo_bonus': 0.2,

    # --- SL anchor (primary constraint) ---
    'anchor_coeff': 0.05,              # Raised: stronger KL penalty
    'kl_clip_threshold': 0.18,         # Hard gate: pause RL if KL > this
    'kl_penalty_scale': 3.0,          # Quadratic penalty scale
    'sl_batch_ratio': 0.40,            # 40% of PPO batch from SL replay
    'sl_imitation_coeff': 0.8,         # Strong SL pull throughout
    'sl_enforce_interval': 25,         # Every N ep: pure SL gradient step
    'sl_replay_capacity': 10000,
    'sl_failure_threshold': -0.3,

    # --- Entropy ---
    'entropy_floor': 0.08,            # Raised floor

    # --- Paths ---
    'sl_model_path': os.path.join(os.path.dirname(__file__), '..', 'SL', 'model',
                                   'checkpoint', 'model_20.pt'),
    'ckpt_dir': os.path.join(os.path.dirname(__file__), 'league_checkpoints_v2'),
    'log_dir': os.path.join(os.path.dirname(__file__), 'league_logs'),

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# ======================================================================
# Replay Buffer (simple in-process version)
# ======================================================================
class SimpleReplayBuffer:
    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def push_many(self, samples):
        for s in samples:
            self.buffer.append(s)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        # Pack into dict of tensors
        obs = torch.from_numpy(np.stack([b['observation'] for b in batch]).astype(np.float32))
        masks = torch.from_numpy(np.stack([b['action_mask'] for b in batch]).astype(np.float32))
        acts = torch.tensor([b['action'] for b in batch], dtype=torch.long)
        advs = torch.tensor([b['advantage'] for b in batch], dtype=torch.float32)
        tgts = torch.tensor([b['target'] for b in batch], dtype=torch.float32)
        old_logps = torch.tensor([b['old_logp'] for b in batch], dtype=torch.float32)
        return obs, masks, acts, advs, tgts, old_logps

    def size(self):
        return len(self.buffer)


# ======================================================================
# Opponent Sampling
# ======================================================================
class OpponentSampler:
    """40% hard + 30% selfplay + 20% SL + 10% exploiter sampling.

    Hard opponent definition:
      - top 2 Elo models
      - OR last checkpoint that beat current model
      - OR win_rate vs current > 0.55 in last 200ep
    """

    def __init__(self, pool, elo, cfg):
        self.pool = pool
        self.elo = elo
        self.cfg = cfg
        self.recent_beaters = deque(maxlen=20)
        self._opp_win_tracker = {}  # opp_id -> (wins, total) vs current

    def sample_three(self, current_id, current_elo, ep):
        all_opp = self._collect_all(current_id)
        if not all_opp:
            sl = self.pool.get_sl()
            return [sl]*3 if sl else []

        all_opp.sort(key=lambda m: self.elo.get_elo(
            m.get('model_id', m.get('category', 'unk'))), reverse=True)
        n = len(all_opp)

        # --- Build hard pool (top 2 Elo + beaters + win_rate>0.55) ---
        hard_pool = []
        # Top 2 Elo
        for m in all_opp[:2]:
            if m not in hard_pool:
                hard_pool.append(m)
        # Recent beaters (last 20)
        for beater in list(self.recent_beaters)[-5:]:
            if beater in all_opp and beater not in hard_pool:
                hard_pool.append(beater)
        # Opponents with win_rate > 0.55 vs current in recent history
        for m in all_opp:
            mid = m.get('model_id', m.get('category', 'unk'))
            w, t = self._opp_win_tracker.get(mid, (0, 1))
            if t >= 10 and w / t > 0.55 and m not in hard_pool:
                hard_pool.append(m)

        # --- Selfplay pool (similar Elo ±30) ---
        selfplay_pool = [m for m in all_opp if abs(
            self.elo.get_elo(m.get('model_id', m.get('category', 'unk'))) - current_elo) <= 30]
        if len(selfplay_pool) < 2:
            selfplay_pool = all_opp[:max(3, n//2)]

        # --- Baseline pool (SL + low Elo) ---
        sl = self.pool.get_sl()
        baseline_pool = all_opp[-max(1, n//4):]
        if sl and sl not in baseline_pool:
            baseline_pool.append(sl)

        # --- Sample 3 opponents ---
        selected = []
        for _ in range(3):
            r = np.random.random()
            if r < self.cfg['hard_ratio'] and hard_pool:
                pool_use = hard_pool
            elif r < self.cfg['hard_ratio'] + self.cfg['selfplay_ratio'] and selfplay_pool:
                pool_use = selfplay_pool
            elif r < self.cfg['hard_ratio'] + self.cfg['selfplay_ratio'] + self.cfg['baseline_ratio'] and baseline_pool:
                pool_use = baseline_pool
            else:
                pool_use = all_opp
            choice = np.random.choice(pool_use)
            selected.append(choice)
        return selected

    def record_result(self, current_won, opponents, fan_margin=0):
        for opp in opponents:
            opp_id = opp.get('model_id', opp.get('category', 'unk'))
            w, t = self._opp_win_tracker.get(opp_id, (0, 0))
            if current_won:
                self._opp_win_tracker[opp_id] = (w, t + 1)
            else:
                self._opp_win_tracker[opp_id] = (w + 1, t + 1)
                self.recent_beaters.append(opp)

    def _collect_all(self, current_id):
        candidates = []
        for entry in self.pool.get_all_opponents():
            if entry.get('model_id', -1) != current_id:
                candidates.append(entry)
        return candidates


# ======================================================================
# SL-Informed Replay Buffer — weighted SL experiences
# ======================================================================
class SLReplayBuffer:
    """Holds state-action pairs where SL policy would have done better.
    These are high-value experiences for auxiliary SL supervision."""

    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def push(self, obs, mask, sl_action, advantage):
        """Store a state where SL disagreed with current policy and SL was better."""
        if advantage < 0:  # Only store if current action was worse than SL would do
            self.buffer.append({
                'observation': obs.copy(),
                'action_mask': mask.copy(),
                'sl_action': sl_action,
                'weight': min(1.0, abs(advantage)),  # Higher weight for bigger failures
            })

    def sample(self, n):
        """Sample n items, weighted by failure magnitude."""
        if len(self.buffer) < n:
            return None
        idx = np.random.choice(len(self.buffer), min(n, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in idx]
        obs = torch.from_numpy(np.stack([b['observation'] for b in batch]).astype(np.float32))
        masks = torch.from_numpy(np.stack([b['action_mask'] for b in batch]).astype(np.float32))
        sl_acts = torch.tensor([b['sl_action'] for b in batch], dtype=torch.long)
        weights = torch.tensor([b['weight'] for b in batch], dtype=torch.float32)
        return obs, masks, sl_acts, weights

    def size(self):
        return len(self.buffer)


def mine_sl_failures(model, sl_model, trajectory, device, threshold=-0.5):
    """Mine state-action pairs where SL would have done significantly better.

    For each step in the trajectory, compare:
      advantage(π_action) vs what advantage would be if SL action was taken.
    If current action has negative advantage AND SL action differs,
    record this as a 'SL failure' sample.

    Returns: list of (obs, mask, sl_action, advantage_deficit) tuples
    """
    if sl_model is None or not trajectory:
        return []

    failures = []
    for step in trajectory:
        if step.get('advantage', 0) >= threshold:
            continue  # Only mine failures (negative advantage)

        obs_t = torch.from_numpy(np.expand_dims(step['observation'], 0)).float().to(device)
        mask_t = torch.from_numpy(np.expand_dims(step['action_mask'], 0)).float().to(device)

        with torch.no_grad():
            # Current policy action
            cur_out = model({'observation': obs_t, 'action_mask': mask_t})
            cur_logits = cur_out[0] if isinstance(cur_out, tuple) else cur_out
            cur_action = int(cur_logits[0].argmax().item())

            # SL anchor action
            sl_out = sl_model({'observation': obs_t, 'action_mask': mask_t})
            sl_logits = sl_out[0] if isinstance(sl_out, tuple) else sl_out
            sl_action = int(sl_logits[0].argmax().item())

        # SL disagrees with current policy (potential improvement signal)
        if sl_action != cur_action and sl_action != step['action']:
            failures.append((
                step['observation'], step['action_mask'],
                sl_action, step.get('advantage', 0)
            ))

    return failures


# ======================================================================
# PPO Update (with auxiliary SL loss)
# ======================================================================
def ppo_update(model, sl_model, optimizer, buffer, sl_replay_buffer, cfg, ep):
    batch = buffer.sample(cfg['batch_size'])
    if batch is None:
        return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl': 0, 'aux_sl': 0}

    obs, masks, acts, advs, tgts, old_logps = batch
    device = cfg['device']
    obs, masks = obs.to(device), masks.to(device)
    acts = acts.to(device)
    advs = advs.to(device)
    tgts = tgts.to(device)
    old_logps = old_logps.to(device)

    # Normalize + clip advantages
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    advs = torch.clamp(advs, -cfg['adv_clip'], cfg['adv_clip'])

    # SL replay batch (40% of batch)
    sl_obs_batch = None
    if sl_replay_buffer is not None and sl_replay_buffer.size() >= 32:
        n_sl = int(cfg['batch_size'] * cfg['sl_batch_ratio'])
        sl_batch = sl_replay_buffer.sample(min(n_sl, sl_replay_buffer.size()))
        if sl_batch is not None:
            sl_obs_batch = sl_batch

    # --- KL pre-check: if KL too high, skip RL update entirely ---
    with torch.no_grad():
        logits, _ = model({'observation': obs, 'action_mask': masks})
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        if sl_model is not None:
            sl_out = sl_model({'observation': obs, 'action_mask': masks})
            sl_logits = sl_out[0] if isinstance(sl_out, tuple) else sl_out
            sl_log_probs = F.log_softmax(sl_logits, dim=-1)
            valid_mask = (masks > 0.5).float()
            pre_kl = (valid_mask * probs * (log_probs - sl_log_probs.detach())).sum(dim=-1).mean().item()

    if pre_kl > cfg['kl_hard_limit']:
        # KL too high — skip RL update, do SL-enforced step only
        if sl_obs_batch is not None:
            sl_o, sl_m, sl_a, sl_w = sl_obs_batch
            sl_o, sl_m = sl_o.to(device), sl_m.to(device)
            sl_a = sl_a.to(device); sl_w = sl_w.to(device)
            sl_logits, _ = model({'observation': sl_o, 'action_mask': sl_m})
            sl_loss = (sl_w * F.cross_entropy(sl_logits, sl_a, reduction='none')).mean()
            optimizer.zero_grad()
            sl_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl': pre_kl, 'aux_sl': 0, 'skipped': True}

    # --- Normal PPO + SL step ---
    rl_scale = 1.0 if pre_kl <= cfg['kl_warn_threshold'] else 0.5  # Half RL strength when KL warns

    total_pl, total_vl, total_el, total_kl, total_sl = 0, 0, 0, 0, 0
    for _ in range(cfg['ppo_epochs']):
        logits, values = model({'observation': obs, 'action_mask': masks})
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_logp = log_probs.gather(1, acts.unsqueeze(1)).squeeze(1)

        # PPO policy loss (scaled by RL strength)
        ratio = torch.exp(selected_logp - old_logps.to(device))
        clip_adv = torch.clamp(ratio, 1 - cfg['clip'], 1 + cfg['clip']) * advs
        policy_loss = -torch.min(ratio * advs, clip_adv).mean()

        # Value loss (reduced weight)
        value_loss = F.mse_loss(values.squeeze(1), tgts)

        # Entropy
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # KL(π || SL)
        kl = torch.tensor(0.0, device=device)
        if sl_model is not None:
            with torch.no_grad():
                sl_out = sl_model({'observation': obs, 'action_mask': masks})
                sl_logits = sl_out[0] if isinstance(sl_out, tuple) else sl_out
            sl_log_probs = F.log_softmax(sl_logits, dim=-1)
            valid_mask = (masks > 0.5).float()
            kl = (valid_mask * probs * (log_probs - sl_log_probs.detach())).sum(dim=-1).mean()

        # KL hard penalty
        kl_penalty = torch.tensor(0.0, device=device)
        if kl > cfg['kl_clip_threshold']:
            kl_penalty = cfg['kl_penalty_scale'] * (kl - cfg['kl_clip_threshold']) ** 2

        # SL imitation (primary learning signal)
        sl_loss = torch.tensor(0.0, device=device)
        if sl_obs_batch is not None:
            sl_o, sl_m, sl_a, sl_w = sl_obs_batch
            sl_o, sl_m = sl_o.to(device), sl_m.to(device)
            sl_a = sl_a.to(device); sl_w = sl_w.to(device)
            sl_logits, _ = model({'observation': sl_o, 'action_mask': sl_m})
            sl_loss = (sl_w * F.cross_entropy(sl_logits, sl_a, reduction='none')).mean()

        # Total loss: SL-dominant
        entropy_bonus_scale = 2.0 if entropy.item() < cfg['entropy_floor'] else 1.0
        loss = (rl_scale * policy_loss +
                cfg['value_coeff'] * value_loss -
                cfg['entropy_coeff'] * entropy * entropy_bonus_scale +
                cfg['anchor_coeff'] * kl + kl_penalty +
                cfg['sl_imitation_coeff'] * sl_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_pl += policy_loss.item()
        total_vl += value_loss.item()
        total_el += entropy.item()
        total_kl += kl.item()
        total_sl += sl_loss.item()

    n = cfg['ppo_epochs']
    return {'policy_loss': total_pl/n, 'value_loss': total_vl/n, 'entropy': total_el/n,
            'kl': total_kl/n, 'aux_sl': total_sl/n, 'skipped': False}


# ======================================================================
# SL-Enforced Update (pure SL gradient step)
# ======================================================================
def sl_enforce_step(model, sl_replay_buffer, optimizer, cfg):
    """Pure SL gradient step — keep policy anchored to SL distribution."""
    if sl_replay_buffer is None or sl_replay_buffer.size() < 64:
        return
    batch = sl_replay_buffer.sample(min(128, sl_replay_buffer.size()))
    if batch is None:
        return
    obs, masks, sl_acts, weights = batch
    device = cfg['device']
    obs, masks = obs.to(device), masks.to(device)
    sl_acts = sl_acts.to(device)
    weights = weights.to(device)

    logits, _ = model({'observation': obs, 'action_mask': masks})
    loss = (weights * F.cross_entropy(logits, sl_acts, reduction='none')).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


# ======================================================================
# Exploiter Training
# ======================================================================
def _train_exploiter(exploiter_model, current_model, sl_replay, optimizer, cfg):
    """Train exploiter to maximize policy gap on failure states.

    Exploiter learns to OPPOSE current policy on states where current fails.
    Loss = -CE(exploiter_logits, current_argmax) on failure states.
    """
    device = cfg['device']
    if sl_replay.size() < 32:
        return

    batch = sl_replay.sample(min(128, sl_replay.size()))
    if batch is None:
        return
    obs, masks, _, _ = batch
    obs, masks = obs.to(device), masks.to(device)

    with torch.no_grad():
        cur_out = current_model({'observation': obs, 'action_mask': masks})
        cur_logits = cur_out[0] if isinstance(cur_out, tuple) else cur_out
        cur_probs = F.softmax(cur_logits, dim=-1)

    for _ in range(3):
        exp_logits, _ = exploiter_model({'observation': obs, 'action_mask': masks})
        exp_probs = F.softmax(exp_logits, dim=-1)
        # Maximize KL(exploiter || current) on failure states
        kl_gap = (exp_probs * (F.log_softmax(exp_logits, dim=-1) -
                                F.log_softmax(cur_logits, dim=-1).detach())).sum(dim=-1).mean()
        loss = -kl_gap  # Maximize divergence
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(exploiter_model.parameters(), 1.0)
        optimizer.step()


# ======================================================================
# Single Game Runner
# ======================================================================
def run_game(current_model, opponent_models, cfg):
    """Run one 4-player game.

    Args:
        current_model: RL policy (seat 0)
        opponent_models: list of 3 models for seats 1-3 (state_dict entries)

    Returns:
        dict with keys: samples, winner_seat, fan_points, total_turns
    """
    device = cfg['device']

    # Load opponent models
    opp_nets = []
    for opp_entry in opponent_models:
        m = CNNModel().to(device)
        if isinstance(opp_entry, dict):
            sd = opp_entry.get('state_dict')
            if sd is None and '_tower.0.weight' in opp_entry:
                # Raw state_dict passed directly — wrap
                sd = opp_entry
        else:
            sd = None
        if sd is None:
            continue
        m.load_state_dict(sd, strict=False)
        m.eval()
        opp_nets.append(m)

    # Build model array: seat 0 = current, seats 1-3 = opponents
    # Ensure we have exactly 4 models (pad with current_model copies if needed)
    while len(opp_nets) < 3:
        opp_nets.append(current_model)
    models = [current_model] + opp_nets[:3]

    # Create env
    env_config = {'agent_clz': FeatureAgent, 'duplicate': True, 'variety': 10000}
    env = MahjongGBEnv(env_config)
    obs_dict = env.reset()

    # Create agents
    agents = [FeatureAgent(i) for i in range(4)]
    for i in range(4):
        agents[i].request2obs('Wind %d' % (i % 4))

    # Experience collection for seat 0
    seat0_trajectory = []

    done = False
    turns = 0
    while not done and turns < 500:
        actions = {}
        for name in env.agent_names:
            i = int(name.split('_')[1]) - 1
            obs = obs_dict.get(name)
            if obs is not None:
                action = model_inference(models[i], obs, agents[i], device,
                                         argmax=(i != 0))
                if i == 0 and obs is not None:
                    # Record experience for seat 0
                    with torch.no_grad():
                        obs_t = torch.from_numpy(
                            np.expand_dims(obs['observation'], 0)).float().to(device)
                        mask_t = torch.from_numpy(
                            np.expand_dims(obs['action_mask'], 0)).float().to(device)
                        logits, value = current_model({
                            'observation': obs_t, 'action_mask': mask_t})
                        probs = F.softmax(logits, dim=-1)
                        log_probs = F.log_softmax(logits, dim=-1)
                        old_logp = log_probs[0, action].item()
                        seat0_trajectory.append({
                            'observation': obs['observation'].copy(),
                            'action_mask': obs['action_mask'].copy(),
                            'action': action,
                            'value': value.item(),
                            'old_logp': old_logp,
                            'reward': 0.0,  # placeholder
                        })
                actions[name] = action

        if not actions:
            break

        obs_dict, reward_dict, done_dict = env.step(actions)
        # Record reward for seat 0
        if seat0_trajectory:
            r = reward_dict.get('player_1', 0)
            seat0_trajectory[-1]['reward'] = r

        done = done_dict
        turns += 1

    # Determine winner
    final_rewards = {name: reward_dict.get(name, 0) for name in env.agent_names}
    winner_seat = None
    for name, r in final_rewards.items():
        if r > 0:
            winner_seat = int(name.split('_')[1]) - 1
            break

    # Fan points (approximate from reward)
    fan = 0
    if winner_seat is not None:
        r = final_rewards.get(f'player_{winner_seat+1}', 0)
        if r > 0:
            fan = max(0, int(r / 3 - 8))

    # Compute GAE for seat 0 trajectory
    if seat0_trajectory:
        # Terminal value = 0
        gae = 0.0
        for t in reversed(range(len(seat0_trajectory))):
            step = seat0_trajectory[t]
            next_val = seat0_trajectory[t+1]['value'] if t+1 < len(seat0_trajectory) else 0.0
            delta = step['reward'] + cfg['gamma'] * next_val - step['value']
            gae = delta + cfg['gamma'] * cfg['gae_lambda'] * gae
            step['advantage'] = gae
            step['target'] = gae + step['value']

    return {
        'samples': seat0_trajectory,
        'winner_seat': winner_seat,
        'fan': fan,
        'turns': turns,
        'hu': winner_seat is not None,
    }


def model_inference(model, obs, agent, device, argmax=False):
    """Run model inference for one observation."""
    obs_t = torch.from_numpy(np.expand_dims(obs['observation'], 0)).float().to(device)
    mask_t = torch.from_numpy(np.expand_dims(obs['action_mask'], 0)).float().to(device)
    with torch.no_grad():
        output = model({'observation': obs_t, 'action_mask': mask_t})
        logits = output[0] if isinstance(output, tuple) else output
    if argmax:
        return int(logits[0].argmax().item())
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


# ======================================================================
# League Evaluation
# ======================================================================
def league_evaluate(current_model, pool, cfg):
    """Evaluate current model against all league opponents."""
    device = cfg['device']
    results = {}

    categories = [
        ('sl', pool.get_sl()),
        ('best', pool.get_best()),
    ]
    for h in pool.get_historical():
        categories.append((f'hist_{h["model_id"]}', h))

    for cat_name, entry in categories:
        if not entry:
            continue
        sd = entry.get('state_dict') if isinstance(entry, dict) else None
        if sd is None and isinstance(entry, dict) and '_tower.0.weight' in entry:
            sd = entry
        if sd is None:
            continue
        opp_model = CNNModel().to(device)
        opp_model.load_state_dict(sd, strict=False)
        opp_model.eval()

        wins = 0
        total_fan = 0
        total_turns = 0
        n = cfg['eval_games'] // max(len(categories), 1)
        n = max(n, 10)

        for _ in range(n):
            # Alternate seat 0 between current and opponent
            seat0_model = current_model if np.random.random() < 0.5 else opp_model
            seat1_model = opp_model if seat0_model is current_model else current_model
            # Create proper pool entries for opponent models
            opp_entry_list = [
                {'state_dict': {k: v.cpu().clone() for k, v in seat1_model.state_dict().items()}, 'model_id': 999},
                {'state_dict': {k: v.cpu().clone() for k, v in opp_model.state_dict().items()}, 'model_id': 998},
                {'state_dict': {k: v.cpu().clone() for k, v in opp_model.state_dict().items()}, 'model_id': 997},
            ]
            result = run_game(seat0_model, opp_entry_list, cfg)
            if result['winner_seat'] is not None:
                if (seat0_model is current_model and result['winner_seat'] == 0) or \
                   (seat0_model is opp_model and result['winner_seat'] != 0):
                    wins += 1
            if result['hu']:
                total_fan += result['fan']
            total_turns += result['turns']

        results[cat_name] = {
            'games': n, 'win_rate': wins / n,
            'avg_fan': total_fan / max(wins, 1),
            'avg_turns': total_turns / n,
        }

    return results


# ======================================================================
# Main Training Loop
# ======================================================================
def train():
    cfg = CFG
    os.makedirs(cfg['ckpt_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'], exist_ok=True)
    device = cfg['device']

    print(f'[League v2] Device: {device}')
    print(f'[League v2] Total episodes: {cfg["total_episodes"]}')

    # ---- Init components ----
    pool = LeagueModelPool(max_historical=cfg['max_historical'],
                           save_dir=os.path.join(cfg['ckpt_dir'], 'models'))
    elo = EloRanker(initial_elo=cfg['initial_elo'], k_factor=cfg['elo_k'])
    sampler = OpponentSampler(pool, elo, cfg)
    exploit_detector = ExploitDetector(window_size=200)
    buffer = SimpleReplayBuffer(max_size=50000)
    sl_replay = SLReplayBuffer(max_size=cfg['sl_replay_capacity'])

    # ---- Load SL anchor ----
    sl_model = None
    sl_path = cfg['sl_model_path']
    if os.path.exists(sl_path):
        sl_model = CNNModel().to(device)
        sl_ckpt = torch.load(sl_path, map_location=device, weights_only=False)
        sl_model.load_state_dict(sl_ckpt, strict=False)
        sl_model.eval()
        sl_sd = {k: v.cpu().clone() for k, v in sl_model.state_dict().items()}
        pool.add_sl(sl_sd, {'path': sl_path})
        elo.register('sl', 1400.0)
        print(f'[League v2] SL anchor loaded: {sl_path}')

    # ---- Init RL model ----
    current_model = CNNModel().to(device)
    if sl_model is not None:
        # Initialize from SL weights
        current_model.load_state_dict(sl_model.state_dict(), strict=False)

    rl_sd_cpu = {k: v.cpu().clone() for k, v in current_model.state_dict().items()}
    rl_id = pool.add_rl_active(rl_sd_cpu, {'ep': 0})
    elo.register(rl_id, cfg['initial_elo'])
    pool.set_champion(rl_sd_cpu, {'ep': 0, 'elo': cfg['initial_elo']})
    pool.add_best(rl_sd_cpu, {'ep': 0, 'hu_rate': 0})
    elo.register(f'best_{rl_id}', cfg['initial_elo'])

    optimizer = torch.optim.Adam(current_model.parameters(), lr=cfg['lr'])

    # Exploiter model (delayed until exploit_delay)
    exploiter_model = None
    exploiter_optimizer = None

    # ---- Training state ----
    total_games = 0
    best_eval_hu = 0.0

    # ---- Main loop ----
    for ep in range(1, cfg['total_episodes'] + 1):
        # 1. Sample opponents
        current_elo_val = elo.get_elo(rl_id)
        opponents = sampler.sample_three(rl_id, current_elo_val, ep)
        if len(opponents) < 3:
            # Pad with SL or current
            sl_entry = pool.get_sl()
            while len(opponents) < 3:
                opponents.append(sl_entry if sl_entry else opponents[0] if opponents else None)

        # 2. Run game
        if ep == 1:
            print(f'[DEBUG] opp types: {[type(o).__name__ for o in opponents]}')
            print(f'[DEBUG] opp keys: {[list(o.keys()) if isinstance(o,dict) else "N/A" for o in opponents]}')
        game_result = run_game(current_model, opponents, cfg)
        total_games += 1

        # 3. Push experience to buffer
        if game_result['samples']:
            buffer.push_many(game_result['samples'])
            # SL failure mining: find states where SL would have done better
            failures = mine_sl_failures(
                current_model, sl_model, game_result['samples'],
                device, threshold=cfg['sl_failure_threshold'])
            for obs, mask, sl_act, adv in failures:
                sl_replay.push(obs, mask, sl_act, adv)

        # 4. Update Elo with margin bonus
        if game_result['winner_seat'] is not None:
            winner_seat = game_result['winner_seat']
            margin = game_result.get('fan', 0)
            bonus = cfg['margin_elo_bonus'] if margin >= 8 else 0
            if winner_seat == 0:
                for opp in opponents:
                    opp_id = opp.get('model_id', opp.get('category', 'opp'))
                    elo.update(rl_id, opp_id, score=1.0 + bonus)
            elif 1 <= winner_seat <= 3:
                opp_winner = opponents[winner_seat-1]
                opp_id = opp_winner.get('model_id', opp_winner.get('category', 'opp'))
                elo.update(opp_id, rl_id, score=1.0 + bonus)
            sampler.record_result(winner_seat == 0, opponents, game_result.get('fan', 0))

        # 5. Exploit data collection
        exploit_detector.record_episode({
            'hu': game_result['winner_seat'] == 0,
            'fan': game_result['fan'] if game_result['winner_seat'] == 0 else 0,
            'reward': 1.0 if game_result['winner_seat'] == 0 else -1.0,
            'seat': 0,
            'wind': ep % 4,
        })

        # 6. PPO update + SL enforce steps
        ppo_stats = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl': 0, 'aux_sl': 0, 'skipped': False}
        if buffer.size() >= cfg['min_buffer_size']:
            ppo_stats = ppo_update(current_model, sl_model, optimizer, buffer, sl_replay, cfg, ep)
            # SL-enforced update: periodic pure SL gradient step
            if ep % cfg['sl_enforce_interval'] == 0:
                sl_enforce_step(current_model, sl_replay, optimizer, cfg)

            # Push updated model to pool periodically
            if ep % 10 == 0:
                rl_sd_cpu = {k: v.cpu().clone() for k, v in current_model.state_dict().items()}
                pool.add_rl_active(rl_sd_cpu, {'ep': ep, 'elo': elo.get_elo(rl_id)})

        # 7. Periodic evaluation
        if ep % cfg['eval_interval'] == 0:
            print(f'\n{"="*60}')
            print(f'[League Eval] Episode {ep}')
            eval_results = league_evaluate(current_model, pool, cfg)
            elo_vals = [e for _, e in elo.get_rankings()]
            elo_spread = max(elo_vals) - min(elo_vals) if elo_vals else 0
            print(f'  KL={ppo_stats["kl"]:.4f} entropy={ppo_stats["entropy"]:.3f} spread={elo_spread:.0f}')
            for cat, r in eval_results.items():
                print(f'  vs {cat}: win={r["win_rate"]:.3f} fan={r["avg_fan"]:.1f} turns={r["avg_turns"]:.0f}')

            # Worst-case scoring: min across all opponents
            win_rates = [r['win_rate'] for r in eval_results.values()]
            avg_win = np.mean(win_rates) if win_rates else 0
            worst_win = np.min(win_rates) if win_rates else 0
            print(f'  Score: avg={avg_win:.3f} worst={worst_win:.3f} (best={best_eval_hu:.3f})')
            if worst_win > best_eval_hu:
                best_eval_hu = worst_win
                rl_sd_cpu = {k: v.cpu().clone() for k, v in current_model.state_dict().items()}
                pool.add_best(rl_sd_cpu, {'ep': ep, 'worst_win': worst_win, 'avg_win': avg_win})
                print(f'  >>> NEW BEST: worst_win={worst_win:.3f} avg={avg_win:.3f}')

            # Historical snapshot
            if ep % cfg['eval_interval'] == 0:
                rl_sd_cpu = {k: v.cpu().clone() for k, v in current_model.state_dict().items()}
                pool.add_historical(rl_sd_cpu, {'ep': ep, 'win_rate': avg_win})

        # 7b. Stability guard (only when PPO actually ran)
        if not ppo_stats.get('skipped') and ppo_stats['kl'] > 0:
            if ppo_stats['kl'] > cfg['kl_hard_limit']:
                old_lr = optimizer.param_groups[0]['lr']
                new_lr = max(1e-6, old_lr * cfg['lr_reduction_factor'])
                optimizer.param_groups[0]['lr'] = new_lr
                cfg['selfplay_ratio'] = max(0.10, cfg['selfplay_ratio'] - 0.05)
                cfg['baseline_ratio'] = min(0.70, cfg['baseline_ratio'] + 0.05)
                print(f'  [STABILITY] KL={ppo_stats["kl"]:.3f} > {cfg["kl_hard_limit"]}: '
                      f'lr {old_lr:.1e}→{new_lr:.1e}')
            if ppo_stats['entropy'] < cfg['entropy_critical']:
                cfg['entropy_coeff'] = min(0.15, cfg['entropy_coeff'] + 0.02)
                print(f'  [STABILITY] entropy={ppo_stats["entropy"]:.3f} < {cfg["entropy_critical"]}: '
                      f'entropy_coeff→{cfg["entropy_coeff"]:.3f}')

        # 7c. Exploiter (delayed)
        if ep >= cfg['exploit_delay'] and exploiter_model is None:
            exploiter_model = CNNModel().to(device)
            exploiter_model.load_state_dict(current_model.state_dict(), strict=False)
            exploiter_optimizer = torch.optim.Adam(exploiter_model.parameters(), lr=cfg['lr'] * 0.3)
            cfg['exploiter_ratio'] = 0.05
            print(f'  [EXPLOIT] Activated at ep {ep}')
        if exploiter_model is not None and ep % 200 == 0 and sl_replay.size() >= 128:
            _train_exploiter(exploiter_model, current_model, sl_replay, exploiter_optimizer, cfg)

        # 8. Periodic stats
        if ep % cfg['print_interval'] == 0:
            rankings = elo.get_rankings()
            top5 = rankings[:5]
            elo_vals = [e for _, e in rankings]
            elo_spread = max(elo_vals) - min(elo_vals) if elo_vals else 0
            skip = ' SKIP' if ppo_stats.get('skipped') else ''
            print(f'[Ep {ep:5d}] games={total_games:6d} '
                  f'buf={buffer.size():6d} '
                  f'hu={game_result["hu"]} fan={game_result["fan"]} '
                  f'pl={ppo_stats["policy_loss"]:.3f} vl={ppo_stats["value_loss"]:.3f} '
                  f'kl={ppo_stats["kl"]:.4f} ent={ppo_stats["entropy"]:.3f} '
                  f'sl={ppo_stats["aux_sl"]:.3f} sp={elo_spread:.0f}{skip} '
                  f'elo=[{", ".join(f"{m}:{e:.0f}" for m,e in top5)}]')

        # 9. Checkpoint
        if ep % cfg['ckpt_interval'] == 0:
            ckpt_path = os.path.join(cfg['ckpt_dir'], f'ckpt_ep{ep}.pt')
            torch.save({
                'model_state_dict': current_model.state_dict(),
                'ep': ep, 'elo': elo.get_elo(rl_id),
                'best_eval_hu': best_eval_hu,
                'total_games': total_games,
            }, ckpt_path)
            pool.save_to_disk()
            print(f'  Checkpoint: {ckpt_path}')

    # ---- Final save ----
    final_path = os.path.join(cfg['ckpt_dir'], 'final_model.pt')
    torch.save({
        'model_state_dict': current_model.state_dict(),
        'ep': cfg['total_episodes'],
        'elo': elo.get_elo(rl_id),
        'best_eval_hu': best_eval_hu,
        'total_games': total_games,
    }, final_path)
    pool.save_to_disk()

    print(f'\n{"="*60}')
    print(f'[League v2] Training complete.')
    print(f'  Total games: {total_games}')
    print(f'  Best eval win_rate: {best_eval_hu:.3f}')
    print(f'  Final Elo ranking:')
    for m, e in elo.get_rankings()[:10]:
        print(f'    {m}: {e:.0f}')
    print(f'  Pool: sl={len(pool._models["sl"])} rl={len(pool._models["rl_active"])} '
          f'best={len(pool._models["best"])} hist={len(pool._models["historical"])}')


if __name__ == '__main__':
    train()
