"""
v13 — v12 baseline + best-model rollback + adaptive KL + two-timescale update
      + league curriculum + early stop.
"""
import os, sys, torch, torch.nn.functional as F, numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RL'))
from model import CNNModel
from feature import FeatureAgent
from env import MahjongGBEnv

CFG = {
    'lr': 5e-5, 'clip': 0.1, 'gamma': 0.98, 'gae_lambda': 0.95,
    'value_coeff': 0.5, 'ppo_epochs': 4, 'batch_size': 256,
    'total_episodes': 2000, 'eval_interval': 200, 'eval_games': 80,
    'print_interval': 50, 'ckpt_interval': 300,

    # Adaptive KL (target center 0.08, band 0.05~0.12)
    'kl_coeff': 0.05,
    'kl_target': 0.08,
    'kl_low': 0.05,                # Below: reduce KL pressure
    'kl_high': 0.12,               # Above: increase KL pressure

    # BC replay
    'bc_buf_size': 10000,
    'bc_batch_ratio': 0.25,

    # Two-timescale BC weight schedule
    'bc_weight_early': 0.2,        # 0-500ep
    'bc_weight_mid': 0.3,          # 500-1000ep
    'bc_weight_late': 0.5,         # 1000ep+

    # Exploiter
    'exploiter_train_every': 100,
    'exploiter_buf_size': 5000,
    'exploiter_batch_ratio': 0.15,
    'exploiter_priority': 2.0,

    # Entropy
    'entropy_coeff': 0.03,
    'entropy_low': 0.06,
    'entropy_high': 0.10,

    # League curriculum opponents
    'sl_ratio_early': 0.70,        # 0-500ep
    'exploiter_ratio_early': 0.20,
    'selfplay_ratio_early': 0.10,
    'sl_ratio_late': 0.50,         # 500ep+
    'exploiter_ratio_late': 0.30,
    'selfplay_ratio_late': 0.20,

    # Rollback
    'rollback_decline_count': 3,   # Consecutive evals declining > this
    'rollback_decline_threshold': 0.05,  # Min rolling decline to trigger

    # Early stop
    'early_stop_win': 0.58,        # rolling SL_win target
    'early_stop_patience': 5,      # consecutive evals without improvement

    # Paths
    'sl_model_path': os.path.join(os.path.dirname(__file__), '..', 'SL', 'model',
                                   'checkpoint', 'model_20.pt'),
    'ckpt_dir': os.path.join(os.path.dirname(__file__), 'league_checkpoints_v13'),
    'seed': None,  # Set to int for reproducibility
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

def get_bc_weight(ep, cfg):
    if ep <= 500: return cfg['bc_weight_early']
    if ep <= 1000: return cfg['bc_weight_mid']
    return cfg['bc_weight_late']

def get_opponent_ratios(ep, cfg):
    if ep <= 500:
        return cfg['sl_ratio_early'], cfg['exploiter_ratio_early'], cfg['selfplay_ratio_early']
    return cfg['sl_ratio_late'], cfg['exploiter_ratio_late'], cfg['selfplay_ratio_late']

# ======================================================================
class SimpleBuffer:
    def __init__(self, max_size=50000):
        self.buf = deque(maxlen=max_size)

    def push(self, s): self.buf.append(s)
    def size(self): return len(self.buf)

    def sample(self, bs):
        if len(self.buf) < bs: return None
        idx = np.random.choice(len(self.buf), bs, replace=False)
        batch = [self.buf[i] for i in idx]
        obs = torch.from_numpy(np.stack([b['obs'] for b in batch]).astype(np.float32))
        masks = torch.from_numpy(np.stack([b['mask'] for b in batch]).astype(np.float32))
        acts = torch.tensor([b['act'] for b in batch], dtype=torch.long)
        advs = torch.tensor([b.get('adv', 0) for b in batch], dtype=torch.float32)
        tgts = torch.tensor([b.get('tgt', 0) for b in batch], dtype=torch.float32)
        old_lp = torch.tensor([b.get('old_lp', 0) for b in batch], dtype=torch.float32)
        return obs, masks, acts, advs, tgts, old_lp


# ======================================================================
def run_game(model, sl_model, exp_model, ep, cfg):
    device = cfg['device']
    sl_r, exp_r, self_r = get_opponent_ratios(ep, cfg)
    r = np.random.random()
    if r < sl_r:
        opp = [sl_model]*3
    elif r < sl_r + exp_r and exp_model is not None:
        opp = [exp_model]*3
    else:
        opp = [model]*3
    models = [model] + opp
    env = MahjongGBEnv({'agent_clz': FeatureAgent, 'duplicate': True, 'variety': 10000})
    obs_dict = env.reset()
    agents = [FeatureAgent(i) for i in range(4)]
    for i in range(4): agents[i].request2obs('Wind %d' % (i % 4))

    traj = []; done = False; turns = 0
    while not done and turns < 500:
        actions = {}
        for name in env.agent_names:
            i = int(name.split('_')[1]) - 1
            obs = obs_dict.get(name)
            if obs is not None:
                ot = torch.from_numpy(np.expand_dims(obs['observation'], 0)).float().to(device)
                mt = torch.from_numpy(np.expand_dims(obs['action_mask'], 0)).float().to(device)
                with torch.no_grad():
                    logits, _ = models[i]({'observation': ot, 'action_mask': mt})
                probs = F.softmax(logits, dim=-1)
                action = int(logits[0].argmax().item()) if i != 0 else int(torch.multinomial(probs, 1).item())
                if i == 0:
                    lp = F.log_softmax(logits, dim=-1)[0, action].item()
                    _, val = model({'observation': ot, 'action_mask': mt})
                    # SL action for BC
                    sl_out = sl_model({'observation': ot, 'action_mask': mt})
                    sl_act = int(sl_out[0].argmax().item()) if isinstance(sl_out, tuple) else int(sl_out.argmax().item())
                    traj.append({'obs': obs['observation'].copy(), 'mask': obs['action_mask'].copy(),
                                 'act': action, 'sl_act': sl_act, 'val': val.item(), 'old_lp': lp, 'r': 0.0})
                actions[name] = action
        if not actions: break
        obs_dict, reward_dict, done_dict = env.step(actions)
        if traj: traj[-1]['r'] = reward_dict.get('player_1', 0)
        done = done_dict; turns += 1

    ws = None
    for n, r in reward_dict.items():
        if r > 0: ws = int(n.split('_')[1]) - 1; break
    fan = 0
    if ws is not None:
        r = reward_dict.get(f'player_{ws+1}', 0)
        if r > 0: fan = max(0, int(r / 3 - 8))

    if traj:
        gae = 0.0
        for t in reversed(range(len(traj))):
            s = traj[t]; nv = traj[t+1]['val'] if t+1 < len(traj) else 0.0
            d = s['r'] + cfg['gamma'] * nv - s['val']
            gae = d + cfg['gamma'] * cfg['gae_lambda'] * gae
            s['adv'] = gae; s['tgt'] = gae + s['val']
    return {'samples': traj, 'ws': ws, 'fan': fan, 'turns': turns, 'hu': ws is not None}


# ======================================================================
def ppo_update(model, sl_model, opt, buf, bc_buf, exp_buf, ep, cfg):
    bs = cfg['batch_size']; device = cfg['device']
    if buf.size() < bs: return {'pl': 0, 'vl': 0, 'ent': 0, 'kl': 0, 'bc': 0}

    batch = buf.sample(bs)
    if batch is None: return {'pl': 0, 'vl': 0, 'ent': 0, 'kl': 0, 'bc': 0}
    obs, masks, acts, advs, tgts, old_lp = batch
    obs, masks = obs.to(device), masks.to(device)
    acts = acts.to(device); advs = advs.to(device)
    tgts = tgts.to(device); old_lp = old_lp.to(device)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    # BC batch (behavior cloning toward SL)
    bc_obs = bc_masks = bc_acts = None
    if bc_buf.size() >= 32:
        n_bc = int(bs * cfg['bc_batch_ratio'])
        bc_batch = bc_buf.sample(min(n_bc, bc_buf.size()))
        if bc_batch: bc_obs, bc_masks, bc_acts, _, _, _ = bc_batch

    # Exploiter batch
    exp_obs = exp_masks = exp_w = None
    if exp_buf.size() >= 16:
        n_exp = int(bs * cfg['exploiter_batch_ratio'])
        exp_batch = exp_buf.sample(min(n_exp, exp_buf.size()))
        if exp_batch: exp_obs, exp_masks, exp_acts, _, _, _ = exp_batch

    tp, tv, te, tk, tbc = 0, 0, 0, 0, 0
    for _ in range(cfg['ppo_epochs']):
        logits, values = model({'observation': obs, 'action_mask': masks})
        probs = F.softmax(logits, dim=-1); logp = F.log_softmax(logits, dim=-1)
        sel_lp = logp.gather(1, acts.unsqueeze(1)).squeeze(1)

        # PPO
        ratio = torch.exp(sel_lp - old_lp)
        clip_adv = torch.clamp(ratio, 1-cfg['clip'], 1+cfg['clip']) * advs
        policy_loss = -torch.min(ratio*advs, clip_adv).mean()
        value_loss = F.mse_loss(values.squeeze(1), tgts)
        entropy = -(probs*logp).sum(dim=-1).mean()

        # KL(π || SL)
        kl = torch.tensor(0.0, device=device)
        if sl_model is not None:
            with torch.no_grad():
                sl_out = sl_model({'observation': obs, 'action_mask': masks})
                sl_l = sl_out[0] if isinstance(sl_out, tuple) else sl_out
            sl_lp = F.log_softmax(sl_l, dim=-1)
            valid = (masks > 0.5).float()
            kl = (valid * probs * (logp - sl_lp.detach())).sum(dim=-1).mean()

        # BC loss (behavior cloning to SL on replay data)
        bc_loss = torch.tensor(0.0, device=device)
        if bc_obs is not None:
            bc_obs_d, bc_masks_d = bc_obs.to(device), bc_masks.to(device)
            bc_acts_d = bc_acts.to(device)
            bc_logits, _ = model({'observation': bc_obs_d, 'action_mask': bc_masks_d})
            bc_loss = F.cross_entropy(bc_logits, bc_acts_d)

        # Exploiter adversarial loss (weighted cross-entropy)
        exp_loss = torch.tensor(0.0, device=device)
        if exp_obs is not None:
            exp_obs_d, exp_masks_d = exp_obs.to(device), exp_masks.to(device)
            exp_acts_d = exp_acts.to(device)
            exp_logits, _ = model({'observation': exp_obs_d, 'action_mask': exp_masks_d})
            exp_loss = F.cross_entropy(exp_logits, exp_acts_d)

        # Adaptive KL centered at 0.08
        kl_item = kl.item()
        if kl_item > cfg['kl_high']:
            kl_scale = 2.0
        elif kl_item < cfg['kl_low']:
            kl_scale = 0.5
        else:
            kl_scale = 1.0

        # Entropy in band
        ent_item = entropy.item()
        ent_scale = 2.0 if ent_item < cfg['entropy_low'] else (0.5 if ent_item > cfg['entropy_high'] else 1.0)

        # Two-timescale BC weight
        bc_w = get_bc_weight(ep, cfg)

        loss = (policy_loss + cfg['value_coeff']*value_loss
                + kl_scale * cfg['kl_coeff'] * kl
                + bc_w * bc_loss
                + cfg['exploiter_priority'] * exp_loss
                - cfg['entropy_coeff'] * entropy * ent_scale)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        tp += policy_loss.item(); tv += value_loss.item()
        te += ent_item; tk += kl_item; tbc += bc_loss.item()

    n = cfg['ppo_epochs']
    return {'pl': tp/n, 'vl': tv/n, 'ent': te/n, 'kl': tk/n, 'bc': tbc/n}


# ======================================================================
def train_exploiter(exp_model, model, sl_model, opt, cfg):
    """Exploiter: maximize KL vs current policy, bounded to SL."""
    device = cfg['device']; bs = 128
    d = SimpleBuffer(max_size=5000)
    # Generate states by running the model
    for _ in range(5):
        r = run_game(model, sl_model, None, 9999, cfg)
        for s in r.get('samples', []):
            d.push({'obs': s['obs'], 'mask': s['mask'], 'act': s['act'], 'adv': 0, 'tgt': 0, 'old_lp': 0})
    batch = d.sample(min(bs, d.size()))
    if batch is None: return
    obs, masks, _, _, _, _ = batch
    obs, masks = obs.to(device), masks.to(device)

    with torch.no_grad():
        cur_out = model({'observation': obs, 'action_mask': masks})
        cur_logits = cur_out[0] if isinstance(cur_out, tuple) else cur_out
        cur_probs = F.softmax(cur_logits, dim=-1)

    for _ in range(5):
        exp_logits, _ = exp_model({'observation': obs, 'action_mask': masks})
        exp_probs = F.softmax(exp_logits, dim=-1)
        exp_lp = F.log_softmax(exp_logits, dim=-1)
        cur_lp = F.log_softmax(cur_logits, dim=-1).detach()
        kl_gap = (exp_probs * (exp_lp - cur_lp)).sum(dim=-1).mean()

        with torch.no_grad():
            sl_out = sl_model({'observation': obs, 'action_mask': masks})
            sl_l = sl_out[0] if isinstance(sl_out, tuple) else sl_out
            sl_lp = F.log_softmax(sl_l, dim=-1)
        kl_to_sl = (exp_probs * (exp_lp - sl_lp.detach())).sum(dim=-1).mean()

        loss = -kl_gap + 0.5 * kl_to_sl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(exp_model.parameters(), 1.0)
        opt.step()


def evaluate(model, sl_model, exp_model, cfg, n=50):
    device = cfg['device']; w, tf, tt = 0, 0, 0
    for g in range(n):
        r = run_game(model, sl_model, exp_model, 9999, cfg)  # late curriculum for eval
        rl_won = (g % 2 == 0 and r['ws'] == 0) or (g % 2 == 1 and r['ws'] != 0)
        if rl_won: w += 1
        if r['hu']: tf += r['fan']
        tt += r['turns']
    return {'win': w/n, 'fan': tf/max(w, 1), 'turns': tt/n}


# ======================================================================
def train():
    cfg = CFG; os.makedirs(cfg['ckpt_dir'], exist_ok=True)
    device = cfg['device']
    print(f'[v13] v12 + rollback + adaptive KL + 2x timescale + curriculum + early stop')

    # SL anchor
    sl = CNNModel().to(device)
    sl_chk = torch.load(cfg['sl_model_path'], map_location=device, weights_only=False)
    sl.load_state_dict(sl_chk, strict=False); sl.eval()
    for p in sl.parameters(): p.requires_grad = False

    # RL model (init from SL)
    model = CNNModel().to(device)
    model.load_state_dict(sl_chk, strict=False)

    # Exploiter
    exp = CNNModel().to(device)
    exp.load_state_dict(sl_chk, strict=False)
    exp_opt = torch.optim.Adam(exp.parameters(), lr=cfg['lr']*0.3)

    # Buffers
    buf = SimpleBuffer(max_size=50000)
    bc_buf = SimpleBuffer(max_size=cfg['bc_buf_size'])
    exp_buf = SimpleBuffer(max_size=cfg['exploiter_buf_size'])

    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    base_lr = cfg['lr']

    # Best-model rollback state
    best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_ep, best_win = 0, 0.0
    rolling_wins = deque(maxlen=cfg['rollback_decline_count']+1)
    decline_count = 0
    no_improve_count = 0
    tg = 0

    for ep in range(1, cfg['total_episodes']+1):
        # Game with curriculum opponents
        r = run_game(model, sl, exp, ep, cfg); tg += 1

        # Fill buffers
        for s in r.get('samples', []):
            smp = {'obs': s['obs'], 'mask': s['mask'], 'act': s['act'],
                    'adv': s.get('adv',0), 'tgt': s.get('tgt',0), 'old_lp': s.get('old_lp',0)}
            buf.push(smp)
            bc_buf.push({'obs': s['obs'], 'mask': s['mask'], 'act': s['sl_act'],
                          'adv': 0, 'tgt': 0, 'old_lp': 0})

        # Exploiter
        if ep % cfg['exploiter_train_every'] == 0 and buf.size() >= 512:
            train_exploiter(exp, model, sl, exp_opt, cfg)
            for _ in range(3):
                er = run_game(exp, sl, None, 9999, cfg)
                for s in er.get('samples', []):
                    exp_buf.push({'obs': s['obs'], 'mask': s['mask'], 'act': s['act'],
                                   'adv': -abs(s.get('adv',0)), 'tgt': s.get('tgt',0), 'old_lp': s.get('old_lp',0)})
            print(f'  [EXPLOITER] ep{ep} buf={buf.size()} exp={exp_buf.size()}')

        # PPO update
        st = ppo_update(model, sl, opt, buf, bc_buf, exp_buf, ep, cfg)

        # Eval
        if ep % cfg['eval_interval'] == 0:
            ev = evaluate(model, sl, exp, cfg, n=cfg['eval_games'])
            rolling_wins.append(ev['win'])
            rolling = np.mean(rolling_wins) if rolling_wins else ev['win']

            # 1. Best-model update
            if ev['win'] > best_win:
                best_win = ev['win']; best_ep = ep
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save({'sd': best_sd, 'ep': ep, 'sl_win': best_win},
                           os.path.join(cfg['ckpt_dir'], 'best_model.pt'))
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 2. Rollback: 3 consecutive evals declining > 0.05
            if len(rolling_wins) >= cfg['rollback_decline_count'] + 1:
                wins_list = list(rolling_wins)
                recent = wins_list[-cfg['rollback_decline_count']-1:]
                if all(recent[i] - recent[i+1] > cfg['rollback_decline_threshold']
                       for i in range(len(recent)-1)):
                    model.load_state_dict(best_sd)
                    new_lr = max(1e-6, base_lr * 0.5)
                    for g in opt.param_groups: g['lr'] = new_lr
                    opt.state.clear()
                    decline_count += 1
                    print(f'  [ROLLBACK] restored ep{best_ep} (win={best_win:.3f}), '
                          f'lr→{new_lr:.1e} (decline #{decline_count})')

            # 3. Early stop
            if rolling >= cfg['early_stop_win'] and no_improve_count >= cfg['early_stop_patience']:
                print(f'\n[EARLY STOP] rolling={rolling:.3f}≥{cfg["early_stop_win"]} '
                      f'no_improve={no_improve_count}≥{cfg["early_stop_patience"]}')
                break

            bc_w = get_bc_weight(ep, cfg)
            print(f'\n[Eval ep{ep}] SL_win={ev["win"]:.3f} rolling={rolling:.3f} best={best_win:.3f}@{best_ep} '
                  f'KL={st["kl"]:.4f} ent={st["ent"]:.3f} bc_w={bc_w:.2f} ni={no_improve_count} dc={decline_count}')

        # Print
        if ep % cfg['print_interval'] == 0:
            bc_w = get_bc_weight(ep, cfg)
            print(f'[Ep{ep:5d}] g={tg:5d} buf={buf.size():5d} hu={r["hu"]} fan={r["fan"]:2d} '
                  f'pl={st["pl"]:.3f} vl={st["vl"]:.3f} kl={st["kl"]:.4f} ent={st["ent"]:.3f} bc={st["bc"]:.3f} bw={bc_w:.1f}')

        if ep % cfg['ckpt_interval'] == 0:
            torch.save({'sd': model.state_dict(), 'ep': ep, 'sl_win': best_win},
                       os.path.join(cfg['ckpt_dir'], f'ckpt_ep{ep}.pt'))

    torch.save({'sd': model.state_dict(), 'ep': ep, 'sl_win': best_win},
               os.path.join(cfg['ckpt_dir'], 'final_model.pt'))
    print(f'[v13] Done. best SL_win={best_win:.3f}@{best_ep} games={tg} rollbacks={decline_count}')


if __name__ == '__main__':
    train()
