"""
v17 — 90% SL + 10% constrained RL fine-tuning.
Core: PPO + heavy BC + EMA + peak checkpoint + rollback + KL monitor.
Three buffers: BC(100k) + DAgger(20k) + Hard Failure(15k).
"""
import os, sys, torch, torch.nn.functional as F, numpy as np, json, random
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RL'))
from model import CNNModel
from feature import FeatureAgent
from env import MahjongGBEnv

CFG = {
    'lr': 3e-5, 'clip': 0.1, 'gamma': 0.98, 'gae_lambda': 0.95,
    'value_coeff': 0.3, 'ppo_epochs': 3, 'batch_size': 256,
    'min_buffer_size': 512,
    'total_episodes': 3000, 'eval_interval': 100, 'eval_games': 500,
    'print_interval': 25, 'ckpt_interval': 300,

    # BC
    'bc_weight': 0.8,                  # Moderate: anchors but allows RL improvement
    'bc_calibrate_every': 100,
    'bc_calibrate_epochs': 1.5,

    # Buffers
    'bc_buf_cap': 100000,
    'dagger_buf_cap': 20000,
    'failure_buf_cap': 15000,

    # KL (soft control in loss, target band 0.06-0.12)
    'kl_coeff': 0.1,                   # λ in: RL_loss + λ * KL(student || SL)
    'kl_low': 0.06,                    # Below: reduce KL push
    'kl_high': 0.12,                   # Above: increase KL push
    'kl_safety_threshold': 0.15,       # EMA KL > this → force BC calibration
    'kl_safety_window': 50,

    # Entropy (floor, no decay)
    'entropy_floor': 0.015,

    # Rollback (relaxed)
    'rollback_margin': 0.05,           # best - current > this
    'rollback_declines': 3,            # AND N consecutive declining evals

    # Opponents: 40% SL, 30% exploiter pool, 30% self-play
    'sl_opponent_ratio': 0.40,
    'exploiter_opponent_ratio': 0.30,

    # Exploiter training
    'exploiter_train_every': 200,      # Clone + train exploiter
    'exploiter_train_games': 75,       # Games to train exploiter
    'exploiter_pool_size': 3,          # Keep last N exploiters

    # Policy pool
    'pool_save_threshold': 0.55,
    'pool_max_size': 5,

    # EMA
    'ema_decay': 0.995,

    # Paths
    'sl_model_path': os.path.join(os.path.dirname(__file__), '..', 'SL', 'model',
                                   'checkpoint', 'model_20.pt'),
    'ckpt_dir': os.path.join(os.path.dirname(__file__), 'v17_checkpoints'),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

def get_entropy_coeff(ep, cfg):
    if ep <= 1000: return cfg['entropy_start']
    if ep <= 2000: return cfg['entropy_mid']
    return cfg['entropy_end']

def get_opponent_ratio(sl_win, cfg):
    return cfg['sl_high_ratio'] if sl_win > cfg['dynamic_threshold'] else cfg['sl_low_ratio']

# ======================================================================
class Buffer:
    def __init__(self, cap): self.b = deque(maxlen=cap)
    def push(self, s): self.b.append(s)
    def size(self): return len(self.b)
    def sample(self, n):
        if len(self.b) < n: return None
        idx = np.random.choice(len(self.b), n, replace=False)
        batch = [self.b[i] for i in idx]
        obs = torch.from_numpy(np.stack([b['obs'] for b in batch]).astype(np.float32))
        masks = torch.from_numpy(np.stack([b['mask'] for b in batch]).astype(np.float32))
        acts = torch.tensor([b['act'] for b in batch], dtype=torch.long)
        advs = torch.tensor([b.get('adv',0) for b in batch], dtype=torch.float32)
        tgts = torch.tensor([b.get('tgt',0) for b in batch], dtype=torch.float32)
        old_lp = torch.tensor([b.get('old_lp',0) for b in batch], dtype=torch.float32)
        return obs, masks, acts, advs, tgts, old_lp

# ======================================================================
def run_game(model, sl_model, exp_pool, cfg):
    device = cfg['device']
    r = np.random.random()
    if r < cfg['sl_opponent_ratio']: opp = [sl_model]*3
    elif r < cfg['sl_opponent_ratio'] + cfg['exploiter_opponent_ratio'] and len(exp_pool) > 0:
        # Sample one exploiter from pool, fill 3 seats
        exp_sd = random.choice(exp_pool)
        exp_tmp = CNNModel().to(device); exp_tmp.load_state_dict(exp_sd, strict=False); exp_tmp.eval()
        opp = [exp_tmp]*3
    else: opp = [model]*3
    models = [model]+opp
    env = MahjongGBEnv({'agent_clz': FeatureAgent, 'duplicate': True, 'variety': 10000})
    obs_dict = env.reset(); agents = [FeatureAgent(i) for i in range(4)]
    for i in range(4): agents[i].request2obs('Wind %d'%(i%4))
    traj = []; done = False; turns = 0
    while not done and turns < 500:
        actions = {}
        for name in env.agent_names:
            i = int(name.split('_')[1])-1; obs = obs_dict.get(name)
            if obs is not None:
                ot = torch.from_numpy(np.expand_dims(obs['observation'],0)).float().to(device)
                mt = torch.from_numpy(np.expand_dims(obs['action_mask'],0)).float().to(device)
                with torch.no_grad():
                    logits,_ = models[i]({'observation':ot,'action_mask':mt})
                probs = F.softmax(logits,dim=-1)
                action = int(logits[0].argmax().item()) if i!=0 else int(torch.multinomial(probs,1).item())
                if i==0:
                    lp = F.log_softmax(logits,dim=-1)[0,action].item()
                    _,val = model({'observation':ot,'action_mask':mt})
                    sl_out = sl_model({'observation':ot,'action_mask':mt})
                    sl_logits = sl_out[0] if isinstance(sl_out,tuple) else sl_out
                    sl_act = int(sl_logits.argmax().item())
                    sl_probs = F.softmax(sl_logits, dim=-1)
                    sl_prob_chosen = sl_probs[0, action].item()
                    traj.append({'obs':obs['observation'].copy(),'mask':obs['action_mask'].copy(),
                                 'act':action,'sl_act':sl_act,'val':val.item(),'old_lp':lp,'r':0.0,
                                 'sl_prob_chosen': sl_prob_chosen})
                actions[name] = action
        if not actions: break
        obs_dict, reward_dict, done_dict = env.step(actions)
        if traj: traj[-1]['r'] = reward_dict.get('player_1',0)
        done = done_dict; turns += 1
    ws = None
    for n,r in reward_dict.items():
        if r>0: ws = int(n.split('_')[1])-1; break
    fan = 0
    if ws is not None:
        r = reward_dict.get(f'player_{ws+1}',0)
        if r>0: fan = max(0, int(r/3-8))
    if traj:
        gae = 0.0
        for t in reversed(range(len(traj))):
            s = traj[t]; nv = traj[t+1]['val'] if t+1<len(traj) else 0.0
            d = s['r']+cfg['gamma']*nv-s['val']; gae = d+cfg['gamma']*cfg['gae_lambda']*gae
            s['adv'] = gae; s['tgt'] = gae+s['val']
    return {'samples':traj,'ws':ws,'fan':fan,'turns':turns,'hu':ws is not None}

# ======================================================================
def ppo_update(model, sl_model, opt, buf, cfg, ep):
    bs = cfg['batch_size']; device = cfg['device']
    if buf.size() < bs: return {'pl':0,'vl':0,'ent':0,'kl':0,'bc':0}
    batch = buf.sample(bs)
    if batch is None: return {'pl':0,'vl':0,'ent':0,'kl':0,'bc':0}
    obs,masks,acts,advs,tgts,old_lp = batch
    obs,masks = obs.to(device),masks.to(device); acts = acts.to(device)
    advs = advs.to(device); tgts = tgts.to(device); old_lp = old_lp.to(device)
    advs = (advs-advs.mean())/(advs.std()+1e-8)

    # BC batch (50% from combined BC+DAgger+Failure)
    bc_obs = bc_masks = bc_acts = bc_weights = None
    if cfg.get('_bc_batch'):
        bc_obs, bc_masks, bc_acts, bc_weights = cfg['_bc_batch']
        bc_obs, bc_masks = bc_obs.to(device), bc_masks.to(device)
        bc_acts = bc_acts.to(device); bc_weights = bc_weights.to(device)

    tp, tv, te, tk, tbc = 0,0,0,0,0
    for _ in range(cfg['ppo_epochs']):
        logits, values = model({'observation':obs,'action_mask':masks})
        probs = F.softmax(logits,dim=-1); logp = F.log_softmax(logits,dim=-1)
        sel_lp = logp.gather(1,acts.unsqueeze(1)).squeeze(1)
        ratio = torch.exp(sel_lp-old_lp)
        clip_adv = torch.clamp(ratio,1-cfg['clip'],1+cfg['clip'])*advs
        policy_loss = -torch.min(ratio*advs,clip_adv).mean()
        value_loss = F.mse_loss(values.squeeze(1),tgts)
        entropy = -(probs*logp).sum(dim=-1).mean()
        kl = torch.tensor(0.0, device=device)
        if sl_model is not None:
            with torch.no_grad():
                sl_out = sl_model({'observation':obs,'action_mask':masks})
                sl_l = sl_out[0] if isinstance(sl_out,tuple) else sl_out
            sl_lp = F.log_softmax(sl_l,dim=-1)
            valid = (masks>0.5).float()
            kl = (valid*probs*(logp-sl_lp.detach())).sum(dim=-1).mean()
        bc_loss = torch.tensor(0.0, device=device)
        if bc_obs is not None:
            bc_logits, _ = model({'observation':bc_obs,'action_mask':bc_masks})
            loss_per_sample = F.cross_entropy(bc_logits, bc_acts, reduction='none')
            bc_loss = (bc_weights * loss_per_sample).mean() if bc_weights is not None else loss_per_sample.mean()
        # KL soft control: push into band [0.06, 0.12]
        kl_item = kl.item()
        if kl_item > cfg['kl_high']: kl_scale = 2.0
        elif kl_item < cfg['kl_low']: kl_scale = 0.5
        else: kl_scale = 1.0
        # Entropy floor: bonus if below floor
        ent_bonus = 2.0 if entropy.item() < cfg['entropy_floor'] else 1.0
        loss = (policy_loss + cfg['value_coeff']*value_loss
                + kl_scale * cfg['kl_coeff'] * kl
                - ent_bonus * entropy * 0.01
                + cfg['bc_weight'] * bc_loss)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tp += policy_loss.item(); tv += value_loss.item(); te += entropy.item()
        tk += kl.item(); tbc += bc_loss.item()
    n = cfg['ppo_epochs']
    return {'pl':tp/n,'vl':tv/n,'ent':te/n,'kl':tk/n,'bc':tbc/n}

# ======================================================================
def bc_calibrate(model, bc_buf, dagger_buf, failure_buf, opt, cfg):
    """Pure BC steps: 60% DAgger + 30% BC + 10% Failure×2."""
    device = cfg['device']; bs = cfg['batch_size']
    n_steps = int((bc_buf.size() + dagger_buf.size()) * cfg['bc_calibrate_epochs'] / bs)
    n_steps = max(5, min(n_steps, 20))
    total_loss = 0
    for _ in range(n_steps):
        n_dagger = int(bs * 0.60); n_bc = int(bs * 0.30); n_fail = bs - n_dagger - n_bc
        samples = []
        for buf, n in [(dagger_buf, n_dagger), (bc_buf, n_bc), (failure_buf, n_fail)]:
            if buf.size() >= max(1, n//2):
                idx = np.random.choice(buf.size(), min(n, buf.size()), replace=False)
                for i in idx: samples.append(buf.b[i])
        if len(samples) < 32: continue
        np.random.shuffle(samples)
        obs = torch.from_numpy(np.stack([s['obs'] for s in samples]).astype(np.float32)).to(device)
        masks = torch.from_numpy(np.stack([s['mask'] for s in samples]).astype(np.float32)).to(device)
        acts = torch.tensor([s['act'] for s in samples], dtype=torch.long).to(device)
        weights = torch.ones(len(samples), device=device)
        for j, s in enumerate(samples):
            if s.get('tier') == 'failure': weights[j] = 2.0
        logits, _ = model({'observation':obs,'action_mask':masks})
        loss = (weights * F.cross_entropy(logits, acts, reduction='none')).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total_loss += loss.item()
    return total_loss / max(n_steps, 1)

def train_exploiter(exp_model, model, sl_model, failure_buf, opt, cfg):
    """Train exploiter on failure states: maximize KL gap vs current policy."""
    device = cfg['device']; bs = 128
    if failure_buf.size() < bs: return
    batch = failure_buf.sample(bs)
    if batch is None: return
    obs, masks, _, _, _, _ = batch; obs, masks = obs.to(device), masks.to(device)
    with torch.no_grad():
        cur_out = model({'observation': obs, 'action_mask': masks})
        cur_logits = cur_out[0] if isinstance(cur_out, tuple) else cur_out
    for _ in range(5):
        exp_logits, _ = exp_model({'observation': obs, 'action_mask': masks})
        exp_lp = F.log_softmax(exp_logits, dim=-1); exp_probs = F.softmax(exp_logits, dim=-1)
        cur_lp = F.log_softmax(cur_logits, dim=-1).detach()
        kl_gap = (exp_probs * (exp_lp - cur_lp)).sum(dim=-1).mean()
        with torch.no_grad():
            sl_out = sl_model({'observation': obs, 'action_mask': masks})
            sl_l = sl_out[0] if isinstance(sl_out, tuple) else sl_out
            sl_lp = F.log_softmax(sl_l, dim=-1)
        kl_to_sl = (exp_probs * (exp_lp - sl_lp.detach())).sum(dim=-1).mean()
        loss = -kl_gap + 0.5 * kl_to_sl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(exp_model.parameters(), 1.0); opt.step()

# ======================================================================
def evaluate(model, sl_model, exp_model, cfg, n=500):
    device = cfg['device']; wins = 0
    for g in range(n):
        sm = model if g%2==0 else sl_model; om = sl_model if g%2==0 else model
        models = [sm, om, om, om]
        env = MahjongGBEnv({'agent_clz':FeatureAgent,'duplicate':True,'variety':10000})
        obs_dict = env.reset(); agents = [FeatureAgent(i) for i in range(4)]
        for i in range(4): agents[i].request2obs('Wind %d'%(i%4))
        done = False; turns = 0
        while not done and turns < 500:
            actions = {}
            for name in env.agent_names:
                i = int(name.split('_')[1])-1; obs = obs_dict.get(name)
                if obs is not None:
                    ot = torch.from_numpy(np.expand_dims(obs['observation'],0)).float().to(device)
                    mt = torch.from_numpy(np.expand_dims(obs['action_mask'],0)).float().to(device)
                    with torch.no_grad():
                        logits,_ = models[i]({'observation':ot,'action_mask':mt})
                    actions[name] = int(logits[0].argmax().item())
            if not actions: break
            obs_dict, reward_dict, done_dict = env.step(actions); done = done_dict; turns += 1
        rl_won = (g%2==0 and any(reward_dict.get(f'player_{j+1}',0)>0 for j in [0])) or \
                 (g%2==1 and any(reward_dict.get(f'player_{j+1}',0)>0 for j in [1,2,3]))
        if rl_won: wins += 1
    return wins/n

# ======================================================================
def train():
    cfg = CFG; os.makedirs(cfg['ckpt_dir'], exist_ok=True)
    pool_dir = os.path.join(cfg['ckpt_dir'], 'policy_pool'); os.makedirs(pool_dir, exist_ok=True)
    device = cfg['device']
    print(f'[v17] 90% SL + 10% constrained RL. Device: {device}')

    sl = CNNModel().to(device)
    sl_chk = torch.load(cfg['sl_model_path'], map_location=device, weights_only=False)
    sl.load_state_dict(sl_chk, strict=False); sl.eval()
    for p in sl.parameters(): p.requires_grad = False

    model = CNNModel().to(device); model.load_state_dict(sl_chk, strict=False)
    exp_model = CNNModel().to(device); exp_model.load_state_dict(sl_chk, strict=False)
    exp_opt = torch.optim.Adam(exp_model.parameters(), lr=cfg['lr']*0.3)
    exp_pool = []  # List of state_dicts
    ema = CNNModel().to(device); ema.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr']); base_lr = cfg['lr']

    bc_buf = Buffer(cfg['bc_buf_cap']); dagger_buf = Buffer(cfg['dagger_buf_cap'])
    failure_buf = Buffer(cfg['failure_buf_cap']); rl_buf = Buffer(50000)

    best_sd = {k:v.cpu().clone() for k,v in ema.state_dict().items()}
    best_win, best_ep = 0.0, 0; eval_wins = deque(maxlen=10)
    kl_ema = 0.0; kl_spike_count = 0; rb = 0; sl_win = 0.0

    for ep in range(1, cfg['total_episodes']+1):
        r = run_game(model, sl, exp_pool, cfg)

        for s in r.get('samples', []):
            smp = {'obs':s['obs'],'mask':s['mask'],'act':s['act'],
                    'adv':s.get('adv',0),'tgt':s.get('tgt',0),'old_lp':s.get('old_lp',0)}
            rl_buf.push(smp)
            # BC buffer: always push SL action
            bc_buf.push({'obs':s['obs'],'mask':s['mask'],'act':s['sl_act'],'tier':'bc'})
            # DAgger buffer: current policy state + SL label
            dagger_buf.push({'obs':s['obs'],'mask':s['mask'],'act':s['sl_act'],'tier':'dagger'})
            # Hard Failure: big negative reward OR SL disagreement
            if s.get('adv',0) < -5.0 or s.get('sl_prob_chosen',1.0) < 0.1:
                failure_buf.push({'obs':s['obs'],'mask':s['mask'],'act':s['sl_act'],'tier':'failure'})

        # Build BC batch (50% of PPO batch from BC+DAgger)
        bc_batch = None
        n_bc = int(cfg['batch_size'] * 0.50)
        combo = []
        for buf, n in [(dagger_buf, int(n_bc*0.60)), (bc_buf, int(n_bc*0.30)), (failure_buf, int(n_bc*0.10))]:
            if buf.size() >= max(1, n//2):
                idx = np.random.choice(buf.size(), min(n, buf.size()), replace=False)
                for i in idx: combo.append((buf.b[i], 2.0 if buf.b[i].get('tier')=='failure' else 1.0))
        if len(combo) >= 32:
            np.random.shuffle(combo)
            samples = [c[0] for c in combo]; weights_list = [c[1] for c in combo]
            bc_obs = torch.from_numpy(np.stack([s['obs'] for s in samples]).astype(np.float32))
            bc_masks = torch.from_numpy(np.stack([s['mask'] for s in samples]).astype(np.float32))
            bc_acts = torch.tensor([s['act'] for s in samples], dtype=torch.long)
            bc_weights = torch.tensor(weights_list, dtype=torch.float32)
            bc_batch = (bc_obs, bc_masks, bc_acts, bc_weights)
        cfg['_bc_batch'] = bc_batch

        st = ppo_update(model, sl, opt, rl_buf, cfg, ep)

        # KL EMA safety net
        kl_ema = 0.9*kl_ema + 0.1*st['kl'] if kl_ema > 0 else st['kl']
        if kl_ema > cfg['kl_safety_threshold']: kl_spike_count += 1
        else: kl_spike_count = 0
        if kl_spike_count >= cfg['kl_safety_window']:
            bc_calibrate(model, bc_buf, dagger_buf, failure_buf, opt, cfg)
            kl_spike_count = 0
            print(f'  [KL SAFETY] EMA KL={kl_ema:.3f}>{cfg["kl_safety_threshold"]} → forced BC calibration')

        # EMA update
        with torch.no_grad():
            for ep_p, p in zip(ema.parameters(), model.parameters()):
                ep_p.data = cfg['ema_decay']*ep_p.data + (1-cfg['ema_decay'])*p.data

        # Exploiter: clone + train every 200ep, maintain pool
        if ep % cfg['exploiter_train_every'] == 0 and ep > 0 and rl_buf.size() >= 1024:
            exp_model.load_state_dict(model.state_dict())  # Clone student
            for _ in range(cfg['exploiter_train_games']):
                train_exploiter(exp_model, model, sl, failure_buf if failure_buf.size()>0 else rl_buf, exp_opt, cfg)
            exp_sd = {k: v.cpu().clone() for k, v in exp_model.state_dict().items()}
            exp_pool.append(exp_sd)
            if len(exp_pool) > cfg['exploiter_pool_size']:
                exp_pool.pop(0)
            print(f'  [EXPLOITER] ep={ep} pool={len(exp_pool)}')

        # BC calibration
        if ep % cfg['bc_calibrate_every'] == 0:
            bc_loss = bc_calibrate(model, bc_buf, dagger_buf, failure_buf, opt, cfg)
            print(f'  [BC CAL] ep={ep} loss={bc_loss:.4f}')

        # Eval
        if ep % cfg['eval_interval'] == 0:
            win = evaluate(ema, sl, exp_model, cfg, n=cfg['eval_games'])
            eval_wins.append(win)
            rolling = np.mean(eval_wins) if eval_wins else win

            if win > best_win:
                best_win = win; best_ep = ep
                best_sd = {k:v.cpu().clone() for k,v in ema.state_dict().items()}
                torch.save({'sd':best_sd,'ep':ep,'sl_win':best_win}, os.path.join(cfg['ckpt_dir'],'best_model.pt'))
            # Policy pool
            if win >= cfg['pool_save_threshold']:
                pool_files = sorted([f for f in os.listdir(pool_dir) if f.endswith('.pt')])
                if len(pool_files) >= cfg['pool_max_size']:
                    os.remove(os.path.join(pool_dir, pool_files[0]))
                torch.save({'sd':{k:v.cpu().clone() for k,v in ema.state_dict().items()},'ep':ep,'sl_win':win},
                           os.path.join(pool_dir, f'policy_ep{ep}_win{win:.3f}.pt'))

            # Rollback
            should_rb = False
            if win < best_win - cfg['rollback_margin'] and len(eval_wins) >= cfg['rollback_declines']+1:
                recent = list(eval_wins)[-cfg['rollback_declines']-1:]
                if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                    should_rb = True
            if should_rb:
                model.load_state_dict(best_sd); ema.load_state_dict(best_sd)
                new_lr = max(1e-6, base_lr*0.5)
                for g in opt.param_groups: g['lr'] = new_lr
                opt.state.clear(); rb += 1
                print(f'  [ROLLBACK #{rb}] win={win:.3f} < best={best_win:.3f} + 3↓ → lr={new_lr:.1e}')

            print(f'[Eval ep{ep}] SL_win={win:.3f} rolling={rolling:.3f} best={best_win:.3f}@{best_ep} KL={st["kl"]:.4f} KL_ema={kl_ema:.4f} ent={st["ent"]:.3f} bc={st["bc"]:.3f} rb={rb}')

    torch.save({'sd':best_sd,'ep':best_ep,'sl_win':best_win}, os.path.join(cfg['ckpt_dir'],'final_model.pt'))
    print(f'[v17] Done. best={best_win:.3f}@{best_ep} rb={rb} pool={len(os.listdir(pool_dir))}')


if __name__ == '__main__':
    train()
