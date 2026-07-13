"""
Pure offline distillation. Collect fixed dataset once, then train without env interaction.
Teachers: s0 (0.55), s2 (0.30), s3 (0.15).  Only KL distillation loss.
"""
import os, sys, torch, torch.nn.functional as F, numpy as np, json
from collections import deque
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RL'))
from model import CNNModel
from feature import FeatureAgent
from env import MahjongGBEnv

CFG = {
    'lr': 1e-4, 'batch_size': 512,
    'total_steps': 50000, 'eval_interval': 500,
    'temp': 2.0, 'ema_decay': 0.995,
    'dataset_size': 80000,
    'data_rl_ratio': 0.40, 'data_exploiter_ratio': 0.40, 'data_sl_ratio': 0.20,
    'eval_games': 500,
    'sl_model_path': os.path.join(os.path.dirname(__file__), '..', 'SL', 'model',
                                   'checkpoint', 'model_20.pt'),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

def load_teacher(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt['sd'] if 'sd' in ckpt else ckpt
    m = CNNModel().to(device); m.load_state_dict(sd, strict=False); m.eval()
    for p in m.parameters(): p.requires_grad = False
    return m

def collect_dataset(cfg, device):
    """Collect fixed dataset: RL self-play + exploiter + SL anchor states."""
    print(f'[Collect] Target: {cfg["dataset_size"]} states...')
    sl = CNNModel().to(device)
    sl_chk = torch.load(cfg['sl_model_path'], map_location=device, weights_only=False)
    sl.load_state_dict(sl_chk, strict=False); sl.eval()
    for p in sl.parameters(): p.requires_grad = False

    t0 = load_teacher('cloud_results_v15/best_model_s0.pt', device)  # s0
    t2 = load_teacher('cloud_results_v15/best_model_s2.pt', device)  # s2
    exp = load_teacher('refine_results_local/exploiter_vs_s0.pt', device) if os.path.exists(
        'refine_results_local/exploiter_vs_s0.pt') else t2

    data = []
    n_rl = int(cfg['dataset_size'] * cfg['data_rl_ratio'])
    n_exp = int(cfg['dataset_size'] * cfg['data_exploiter_ratio'])
    n_sl = cfg['dataset_size'] - n_rl - n_exp

    # RL self-play (s0 vs s0-like opponents)
    for _ in range(n_rl // 40 + 1):
        r = run_game_collect(t0, sl, t2, cfg, 'rl')
        for s in r: data.append(s)
        if len(data) >= n_rl: break
    data = data[:n_rl]
    print(f'  RL replay: {len(data)} states')

    # Exploiter replay
    exp_data = []
    for _ in range(n_exp // 40 + 1):
        r = run_game_collect(exp, sl, t0, cfg, 'exploiter')
        for s in r: exp_data.append(s)
        if len(exp_data) >= n_exp: break
    data += exp_data[:n_exp]
    print(f'  Exploiter replay: {len(exp_data[:n_exp])} states')

    # SL anchor states (SL model playing)
    sl_data = []
    for _ in range(n_sl // 40 + 1):
        r = run_game_collect(sl, sl, sl, cfg, 'sl')
        for s in r: sl_data.append(s)
        if len(sl_data) >= n_sl: break
    data += sl_data[:n_sl]
    print(f'  SL replay: {len(sl_data[:n_sl])} states')

    print(f'  Total: {len(data)} states')
    return data

def run_game_collect(model, sl_model, opponent, cfg, mode):
    """Collect states from self-play games."""
    device = cfg['device']
    if mode == 'rl': opp = [opponent]*3
    elif mode == 'exploiter': opp = [model]*3  # exploiter self-play
    else: opp = [sl_model]*3
    models = [model] + opp
    env = MahjongGBEnv({'agent_clz': FeatureAgent, 'duplicate': True, 'variety': 10000})
    obs_dict = env.reset()
    agents = [FeatureAgent(i) for i in range(4)]
    for i in range(4): agents[i].request2obs('Wind %d' % (i % 4))
    states = []; done = False; turns = 0
    while not done and turns < 500:
        for name in env.agent_names:
            i = int(name.split('_')[1]) - 1; obs = obs_dict.get(name)
            if obs is not None and i == 0:
                states.append({'obs': obs['observation'].copy(),
                               'mask': obs['action_mask'].copy()})
        actions = {}
        for name in env.agent_names:
            i = int(name.split('_')[1]) - 1; obs = obs_dict.get(name)
            if obs is not None:
                ot = torch.from_numpy(np.expand_dims(obs['observation'], 0)).float().to(device)
                mt = torch.from_numpy(np.expand_dims(obs['action_mask'], 0)).float().to(device)
                with torch.no_grad():
                    logits, _ = models[i]({'observation': ot, 'action_mask': mt})
                action = int(logits[0].argmax().item())
                actions[name] = action
        if not actions: break
        obs_dict, reward_dict, done_dict = env.step(actions)
        done = done_dict; turns += 1
    return states

def _eval_worker(args):
    """Parallel eval worker. args = (student_sd, opp_sd, start, end, device_str)"""
    student_sd, opp_sd, g_start, g_end, device_str = args
    import torch, numpy as np
    from model import CNNModel
    from feature import FeatureAgent
    from env import MahjongGBEnv
    device = torch.device(device_str)
    student = CNNModel().to(device); student.load_state_dict(student_sd, strict=False); student.eval()
    opp = CNNModel().to(device); opp.load_state_dict(opp_sd, strict=False); opp.eval()
    wins = 0
    for g in range(g_start, g_end):
        sm = student if g % 2 == 0 else opp
        om = opp if g % 2 == 0 else student
        models = [sm, om, om, om]
        env = MahjongGBEnv({'agent_clz': FeatureAgent, 'duplicate': True, 'variety': 10000})
        obs_dict = env.reset(); agents = [FeatureAgent(i) for i in range(4)]
        for i in range(4): agents[i].request2obs('Wind %d' % (i % 4))
        done = False; turns = 0
        while not done and turns < 500:
            actions = {}
            for name in env.agent_names:
                i = int(name.split('_')[1]) - 1; obs = obs_dict.get(name)
                if obs is not None:
                    ot = torch.from_numpy(np.expand_dims(obs['observation'], 0)).float().to(device)
                    mt = torch.from_numpy(np.expand_dims(obs['action_mask'], 0)).float().to(device)
                    with torch.no_grad(): logits, _ = models[i]({'observation': ot, 'action_mask': mt})
                    actions[name] = int(logits[0].argmax().item())
            if not actions: break
            obs_dict, reward_dict, done_dict = env.step(actions); done = done_dict; turns += 1
        rl_won = (g % 2 == 0 and any(reward_dict.get(f'player_{j+1}', 0) > 0 for j in [0])) or \
                 (g % 2 == 1 and any(reward_dict.get(f'player_{j+1}', 0) > 0 for j in [1, 2, 3]))
        if rl_won: wins += 1
    return wins


def evaluate_model(model, opponent, cfg, n_games):
    """Parallel CPU evaluation using spawn (NPU incompatible with fork)."""
    import multiprocessing as mp
    n_workers = min(48, max(1, n_games // 15))
    chunk_size = n_games // n_workers
    student_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    opp_sd = {k: v.cpu().clone() for k, v in opponent.state_dict().items()}
    # Always use CPU for eval workers (NPU doesn't support fork)
    args_list = [(student_sd, opp_sd, i * chunk_size,
                  (i + 1) * chunk_size if i < n_workers - 1 else n_games, 'cpu')
                 for i in range(n_workers)]
    ctx = mp.get_context('spawn')
    with ctx.Pool(n_workers) as p:
        results = p.map(_eval_worker, args_list)
    return sum(results) / n_games

def teacher_ensemble_batch(obs, masks, teachers, weights, temp, device):
    """Compute p_teacher = Σ w_i * softmax(logits_i / T) for a batch."""
    ot = obs.float().to(device); mt = masks.float().to(device)
    p = None
    for t, w in zip(teachers, weights):
        with torch.no_grad():
            logits, _ = t({'observation': ot, 'action_mask': mt})
        probs = F.softmax(logits / temp, dim=-1)
        p = w * probs if p is None else p + w * probs
    return p

def train():
    cfg = CFG; os.makedirs('distill_offline_output', exist_ok=True)
    device = cfg['device']
    print(f'[Offline Distill] Device: {device}')

    # Teachers
    print('[Offline] Loading teachers...')
    t0 = load_teacher('cloud_results_v15/best_model_s0.pt', device)
    t2 = load_teacher('cloud_results_v15/best_model_s2.pt', device)
    t3 = load_teacher('cloud_results_v15/best_model_s3.pt', device)
    teachers = [t0, t2, t3]
    t_weights = [0.55, 0.30, 0.15]
    print(f'  s0 (0.55), s2 (0.30), s3 (0.15)')

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

    # Collect fixed dataset
    dataset = collect_dataset(cfg, device)
    obs_all = torch.from_numpy(np.stack([d['obs'] for d in dataset]).astype(np.float32))
    masks_all = torch.from_numpy(np.stack([d['mask'] for d in dataset]).astype(np.float32))
    N = len(dataset)
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'work', 'distill_output')
    os.makedirs(output_dir, exist_ok=True)
    print(f'[Offline] Dataset: {N} states ready. Output dir: {output_dir}')

    opt = torch.optim.Adam(student.parameters(), lr=cfg['lr'])
    total_steps = cfg['total_steps']

    for step in range(1, total_steps + 1):
        # Sample batch
        idx = np.random.choice(N, cfg['batch_size'], replace=False)
        obs = obs_all[idx]; masks = masks_all[idx]

        # Teacher ensemble target
        p_teacher = teacher_ensemble_batch(obs, masks, teachers, t_weights, cfg['temp'], device)

        # Student forward
        logits, _ = student({'observation': obs.to(device), 'action_mask': masks.to(device)})
        student_probs_T = F.softmax(logits / cfg['temp'], dim=-1)
        student_lp_T = F.log_softmax(logits / cfg['temp'], dim=-1)

        # KL(student || teacher) * T² — only loss
        loss = (student_probs_T * (student_lp_T - torch.log(p_teacher + 1e-10))).sum(
            dim=-1).mean() * (cfg['temp'] ** 2)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()

        # EMA
        with torch.no_grad():
            for ep, sp in zip(ema.parameters(), student.parameters()):
                ep.data = cfg['ema_decay'] * ep.data + (1 - cfg['ema_decay']) * sp.data

        # Eval every 500 steps
        if step % cfg['eval_interval'] == 0:
            win_sl = evaluate_model(ema, sl, cfg, cfg['eval_games'])
            win_s0 = evaluate_model(ema, t0, cfg, cfg['eval_games'])
            win_s2 = evaluate_model(ema, t2, cfg, cfg['eval_games'])
            win_s3 = evaluate_model(ema, t3, cfg, cfg['eval_games'])

            print(f'[Step {step:5d}] SL={win_sl:.3f} s0={win_s0:.3f} s2={win_s2:.3f} s3={win_s3:.3f} | loss={loss.item():.4f}')

            if win_sl >= 0.62 and win_s0 >= 0.50:
                sd = {k: v.cpu().clone() for k, v in ema.state_dict().items()}
                torch.save({'sd': sd, 'step': step, 'win_sl': win_sl, 'win_s0': win_s0,
                            'win_s2': win_s2, 'win_s3': win_s3},
                           os.path.join(output_dir, 'best_model.pt'))
                print(f'  [BEST] step={step} win_sl={win_sl:.3f} win_s0={win_s0:.3f} → saved!')

    # Final
    final_win_sl = evaluate_model(ema, sl, cfg, 1000)
    final_win_s0 = evaluate_model(ema, t0, cfg, 1000)
    print(f'\n[Offline] Done. Final: SL={final_win_sl:.3f} s0={final_win_s0:.3f}')
    if not os.path.exists(os.path.join(output_dir, 'best_model.pt')):
        sd = {k: v.cpu().clone() for k, v in ema.state_dict().items()}
        torch.save({'sd': sd, 'step': total_steps, 'win_sl': final_win_sl, 'win_s0': final_win_s0},
                   os.path.join(output_dir, 'final_model.pt'))


if __name__ == '__main__':
    train()
