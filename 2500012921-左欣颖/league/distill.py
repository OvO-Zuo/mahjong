"""
Policy Distillation — merge top-3 Elo teachers into one student.
Teachers: s0 (Elos0), low_clip (Elo#1), entropy_up (Elo#2)
"""
import os, sys, torch, torch.nn.functional as F, numpy as np, json
from collections import deque
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RL'))
from model import CNNModel
from feature import FeatureAgent
from env import MahjongGBEnv

CFG = {
    'lr': 1e-5, 'weight_decay': 1e-4,
    'batch_size': 256, 'buffer_capacity': 50000,
    'total_episodes': 1500, 'eval_interval': 100,
    'eval_games_vs_sl': 500, 'eval_games_vs_teacher': 300,
    'eval_games_vs_s0': 500,
    'temp': 2.0, 'ema_decay': 0.995,
    'opp_sl': 0.40, 'opp_selfplay': 0.30, 'opp_exploiter': 0.30,
    'sl_model_path': os.path.join(os.path.dirname(__file__), '..', 'SL', 'model',
                                   'checkpoint', 'model_20.pt'),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

class Buffer:
    def __init__(self, cap=50000): self.b = deque(maxlen=cap)
    def push(self, s): self.b.append(s)
    def size(self): return len(self.b)
    def sample(self, n):
        if len(self.b) < n: return None
        idx = np.random.choice(len(self.b), n, replace=False)
        batch = [self.b[i] for i in idx]
        obs = torch.from_numpy(np.stack([b['obs'] for b in batch]).astype(np.float32))
        masks = torch.from_numpy(np.stack([b['mask'] for b in batch]).astype(np.float32))
        sl_acts = torch.tensor([b['sl_act'] for b in batch], dtype=torch.long)
        return obs, masks, sl_acts

def load_teacher(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt['sd'] if 'sd' in ckpt else ckpt
    m = CNNModel().to(device); m.load_state_dict(sd, strict=False); m.eval()
    for p in m.parameters(): p.requires_grad = False
    return m

def teacher_ensemble(state_obs, state_mask, teachers, weights, temp, device):
    """Build p_teacher = sum(w_i * softmax(logits_i / T))."""
    ot = state_obs.float().to(device); mt = state_mask.float().to(device)
    p_teacher = None
    for t, w in zip(teachers, weights):
        with torch.no_grad():
            logits, _ = t({'observation': ot, 'action_mask': mt})
        probs = F.softmax(logits / temp, dim=-1)
        p_teacher = w * probs if p_teacher is None else p_teacher + w * probs
    return p_teacher

def run_game(student, sl_model, exploiter, cfg):
    """Collect game with specified opponent mix."""
    device = cfg['device']
    r = np.random.random()
    if r < cfg['opp_sl']: opp = [sl_model]*3
    elif r < cfg['opp_sl'] + cfg['opp_selfplay']: opp = [student]*3
    else: opp = [exploiter]*3 if exploiter is not None else [sl_model]*3
    models = [student] + opp
    env = MahjongGBEnv({'agent_clz': FeatureAgent, 'duplicate': True, 'variety': 10000})
    obs_dict = env.reset()
    agents = [FeatureAgent(i) for i in range(4)]
    for i in range(4): agents[i].request2obs('Wind %d' % (i % 4))
    traj = []; done = False; turns = 0
    while not done and turns < 500:
        actions = {}
        for name in env.agent_names:
            i = int(name.split('_')[1]) - 1; obs = obs_dict.get(name)
            if obs is not None:
                ot = torch.from_numpy(np.expand_dims(obs['observation'], 0)).float().to(device)
                mt = torch.from_numpy(np.expand_dims(obs['action_mask'], 0)).float().to(device)
                with torch.no_grad():
                    logits, _ = models[i]({'observation': ot, 'action_mask': mt})
                probs = F.softmax(logits, dim=-1)
                action = int(logits[0].argmax().item()) if i != 0 else int(
                    torch.multinomial(probs, 1).item())
                if i == 0:
                    sl_out = sl_model({'observation': ot, 'action_mask': mt})
                    sl_act = int(sl_out[0].argmax().item()) if isinstance(sl_out, tuple) else int(
                        sl_out.argmax().item())
                    traj.append({'obs': obs['observation'].copy(),
                                 'mask': obs['action_mask'].copy(), 'sl_act': sl_act})
                actions[name] = action
        if not actions: break
        obs_dict, reward_dict, done_dict = env.step(actions)
        done = done_dict; turns += 1
    ws = None
    for n, r in reward_dict.items():
        if r > 0: ws = int(n.split('_')[1]) - 1; break
    return {'samples': traj, 'ws': ws, 'hu': ws is not None}

def evaluate(student, opponent, cfg, n_games):
    """Evaluate student vs opponent. Alternating seats."""
    device = cfg['device']; wins = 0
    for g in range(n_games):
        sm = student if g % 2 == 0 else opponent
        om = opponent if g % 2 == 0 else student
        opp = [om]*3; models = [sm] + opp
        env = MahjongGBEnv({'agent_clz': FeatureAgent, 'duplicate': True, 'variety': 10000})
        obs_dict = env.reset()
        agents = [FeatureAgent(i) for i in range(4)]
        for i in range(4): agents[i].request2obs('Wind %d' % (i % 4))
        done = False; turns = 0
        while not done and turns < 500:
            actions = {}
            for name in env.agent_names:
                i = int(name.split('_')[1]) - 1; obs = obs_dict.get(name)
                if obs is not None:
                    ot = torch.from_numpy(np.expand_dims(obs['observation'], 0)).float().to(
                        device)
                    mt = torch.from_numpy(np.expand_dims(obs['action_mask'], 0)).float().to(
                        device)
                    with torch.no_grad():
                        logits, _ = models[i]({'observation': ot, 'action_mask': mt})
                    action = int(logits[0].argmax().item())
                    actions[name] = action
            if not actions: break
            obs_dict, reward_dict, done_dict = env.step(actions)
            done = done_dict; turns += 1
        rl_won = (g % 2 == 0 and any(
            reward_dict.get(f'player_{j+1}', 0) > 0 for j in [0])) or \
                 (g % 2 == 1 and any(
                     reward_dict.get(f'player_{j+1}', 0) > 0 for j in [1, 2, 3]))
        if rl_won: wins += 1
    return wins / n_games

def distill_step(student, ema_model, sl_model, teachers, t_weights, opt, buf, cfg):
    """One distillation update."""
    batch = buf.sample(cfg['batch_size'])
    if batch is None: return {'distill': 0, 'bc': 0, 'kl_sl': 0}
    obs, masks, sl_acts = batch
    device = cfg['device']; obs = obs.to(device); masks = masks.to(device)
    sl_acts = sl_acts.to(device)

    # Teacher ensemble target
    p_teacher = teacher_ensemble(obs, masks, teachers, t_weights, cfg['temp'], device)

    # Student forward
    logits, _ = student({'observation': obs, 'action_mask': masks})
    student_log_probs = F.log_softmax(logits / cfg['temp'], dim=-1)

    # L_distill = KL(student/T || teacher) * T^2 = KL(log_softmax(student/T), teacher) * 4
    student_probs_T = F.softmax(logits / cfg['temp'], dim=-1)
    student_log_probs_T = F.log_softmax(logits / cfg['temp'], dim=-1)
    # KL(student || teacher) = Σ student_probs * (log_student - log_teacher)
    l_distill = (student_probs_T * (student_log_probs_T - torch.log(p_teacher + 1e-10))).sum(
        dim=-1).mean() * (cfg['temp'] ** 2)

    # L_bc = CrossEntropy(student_logits, sl_action)
    l_bc = F.cross_entropy(logits, sl_acts)

    # L_kl = KL(student || SL_anchor)
    with torch.no_grad():
        sl_out = sl_model({'observation': obs, 'action_mask': masks})
        sl_logits = sl_out[0] if isinstance(sl_out, tuple) else sl_out
    sl_probs = F.softmax(sl_logits, dim=-1)
    student_probs = F.softmax(logits, dim=-1)
    student_lp = F.log_softmax(logits, dim=-1)
    sl_lp = F.log_softmax(sl_logits, dim=-1)
    valid = (masks > 0.5).float()
    l_kl = (valid * student_probs * (student_lp - sl_lp.detach())).sum(dim=-1).mean()

    loss = 1.0 * l_distill + 0.30 * l_bc + 0.05 * l_kl

    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()

    # EMA
    with torch.no_grad():
        for ep, sp in zip(ema_model.parameters(), student.parameters()):
            ep.data = cfg['ema_decay'] * ep.data + (1 - cfg['ema_decay']) * sp.data

    return {'distill': l_distill.item(), 'bc': l_bc.item(), 'kl_sl': l_kl.item()}

def train():
    cfg = CFG; os.makedirs('distill_output', exist_ok=True)
    device = cfg['device']
    print(f'[Distill] Device: {device}')

    # Teachers
    print('[Distill] Loading teachers...')
    t0 = load_teacher('cloud_results_v15/best_model_s0.pt', device)  # s0 champion
    t1 = load_teacher('refine_results_local/branch_low_clip/best_model.pt', device)
    t2 = load_teacher('refine_results_local/branch_entropy_up/best_model.pt', device)
    teachers = [t0, t1, t2]
    t_weights = [0.60, 0.25, 0.15]
    print(f'  Teachers: s0 (0.60), low_clip (0.25), entropy_up (0.15)')

    # Student from s0
    student = CNNModel().to(device)
    s0_sd = torch.load('cloud_results_v15/best_model_s0.pt', map_location='cpu',
                        weights_only=False)['sd']
    student.load_state_dict(s0_sd, strict=False)
    ema = CNNModel().to(device); ema.load_state_dict(student.state_dict())

    # SL anchor
    sl = CNNModel().to(device)
    sl_chk = torch.load(cfg['sl_model_path'], map_location=device, weights_only=False)
    sl.load_state_dict(sl_chk, strict=False); sl.eval()
    for p in sl.parameters(): p.requires_grad = False

    # Exploiter
    exp = load_teacher('refine_results_local/exploiter_vs_s0.pt', device) if os.path.exists(
        'refine_results_local/exploiter_vs_s0.pt') else sl

    opt = torch.optim.AdamW(student.parameters(), lr=cfg['lr'],
                             weight_decay=cfg['weight_decay'])
    buf = Buffer(cap=cfg['buffer_capacity'])

    # Best tracking
    best_sd = {k: v.cpu().clone() for k, v in ema.state_dict().items()}
    best_win_sl = 0.0; best_wins = {}
    eval_log = []; rollbacks = 0

    for ep in range(1, cfg['total_episodes'] + 1):
        # Collect
        r = run_game(student, sl, exp, cfg)
        for s in r.get('samples', []): buf.push(s)

        # Distill step
        st = distill_step(student, ema, sl, teachers, t_weights, opt, buf, cfg)

        # Eval
        if ep % cfg['eval_interval'] == 0:
            win_sl = evaluate(ema, sl, cfg, cfg['eval_games_vs_sl'])
            win_s0 = evaluate(ema, t0, cfg, cfg['eval_games_vs_s0'])
            win_lc = evaluate(ema, t1, cfg, cfg['eval_games_vs_teacher'])
            win_eu = evaluate(ema, t2, cfg, cfg['eval_games_vs_teacher'])

            # Rollback check
            should_rb = False
            if best_win_sl > 0:
                if win_sl < best_win_sl - 0.03 or win_s0 < 0.45:
                    should_rb = True
            if should_rb:
                student.load_state_dict(best_sd); ema.load_state_dict(best_sd)
                opt = torch.optim.AdamW(student.parameters(), lr=cfg['lr'],
                                         weight_decay=cfg['weight_decay'])
                rollbacks += 1
                print(f'  [ROLLBACK #{rollbacks}] win_sl={win_sl:.3f} win_s0={win_s0:.3f}')

            # Best model
            if win_sl >= 0.64 and win_s0 >= 0.50:
                best_win_sl = win_sl
                best_sd = {k: v.cpu().clone() for k, v in ema.state_dict().items()}
                best_wins = {'sl': win_sl, 's0': win_s0, 'low_clip': win_lc, 'entropy_up': win_eu}
                torch.save({'sd': best_sd, 'ep': ep, 'wins': best_wins},
                           'distill_output/best_model.pt')
                torch.save({'sd': {k: v.cpu().clone() for k, v in ema.state_dict().items()},
                            'ep': ep, 'wins': best_wins}, 'distill_output/best_model_ema.pt')
                print(f'  [NEW BEST] ep={ep} win_sl={win_sl:.3f} win_s0={win_s0:.3f}')

            eval_log.append(
                {'ep': ep, 'win_sl': win_sl, 'win_s0': win_s0, 'win_low_clip': win_lc,
                 'win_entropy_up': win_eu, 'distill': st['distill'], 'bc': st['bc'],
                 'kl_sl': st['kl_sl']})
            print(
                f'[Eval ep{ep}] SL={win_sl:.3f} s0={win_s0:.3f} low_clip={win_lc:.3f} entropy_up={win_eu:.3f} | dist={st["distill"]:.3f} bc={st["bc"]:.3f} kl_sl={st["kl_sl"]:.4f} rb={rollbacks}')

    # Final save
    torch.save({'sd': best_sd, 'wins': best_wins}, 'distill_output/distilled_model.pt')
    with open('distill_output/eval_report.json', 'w') as f: json.dump(eval_log, f, indent=2)
    print(f'\n[Distill] Done. Best: {best_wins} rollbacks={rollbacks}')
    return best_wins


if __name__ == '__main__':
    train()
