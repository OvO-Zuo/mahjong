"""Cloud Enhanced RL Training — 7 improvements + NPU support."""
import os, sys, time, json, copy, numpy as np
from collections import defaultdict
import torch, torch.nn.functional as F
from torch.distributions import Categorical
import torch_npu

WORK = '/home/ma-user/work'
os.chdir(WORK); sys.path.insert(0, WORK)
from model_var import CNNModelVar
from env import MahjongGBEnv
from feature_agent_ext import make_agent_cls
from MahjongGB import MahjongFanCalculator

device = 'npu:0'; torch.npu.set_device(device)
print(f'Device: {device}', flush=True)

N_EP = 20000
EVAL_EVERY = 500
EVAL_GAMES = 30
LR = 5e-5
CLIP = 0.1
ANCHOR_COEFF = 0.05
GAMMA = 0.98; LAM = 0.95
VC = 1.0; EC = 0.01; PPE = 4; BS = 256

OUT = os.path.join(WORK, 'enhanced_out')
os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════
# SL Anchor (frozen, NPU)
# ═══════════════════════════════════════════════════════
class SLAnchor(torch.nn.Module):
    """Frozen SL model for KL anchor — matches SL model.py architecture."""
    def __init__(self):
        super().__init__()
        self._tower = torch.nn.Sequential(
            torch.nn.Conv2d(6, 64, 3, 1, 1, bias=False), torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False), torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False), torch.nn.ReLU(True),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 36, 256), torch.nn.ReLU(True),
            torch.nn.Linear(256, 235),
        )
    def forward(self, obs, mask):
        x = self._tower(obs)
        inf = torch.clamp(torch.log(mask + 1e-8), -1e38, 1e38)
        return x + inf

sl_anchor = SLAnchor().to(device)
sl_path = os.path.join(WORK, 'model_20.pt')
sd = torch.load(sl_path, map_location='cpu', weights_only=True)
sl_anchor.load_state_dict(sd)
sl_anchor.eval()
for p in sl_anchor.parameters(): p.requires_grad = False
print('SL anchor loaded', flush=True)

# ═══════════════════════════════════════════════════════
# Model Pool
# ═══════════════════════════════════════════════════════
class ModelPool:
    def __init__(self, max_size=5):
        self.max_size = max_size; self.models = []
    def push(self, sd):
        self.models.append({k: v.cpu().clone() for k, v in sd.items()})
        if len(self.models) > self.max_size: self.models.pop(0)
    def size(self): return len(self.models)

# ═══════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════
def evaluate(model, AgentCls, n_games=30):
    model.eval()
    env = MahjongGBEnv(config={'agent_clz': AgentCls})
    hu_count = 0; max_fans = []; lengths = []
    for _ in range(n_games):
        obs_dict = env.reset(); done = False; ep_len = 0; term_r = None
        while not done:
            actions = {}
            for a in obs_dict:
                ot = torch.tensor(obs_dict[a]['observation'], dtype=torch.float).unsqueeze(0).to(device)
                mt = torch.tensor(obs_dict[a]['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = model({'observation': ot, 'action_mask': mt})
                    act = logits.argmax(dim=1).item()
                actions[a] = act
            next_obs, rewards, done = env.step(actions)
            if done: term_r = rewards
            obs_dict = next_obs; ep_len += 1
        lengths.append(ep_len)
        if term_r: rv = list(term_r.values()); hu = max(rv) > 0; mr = max(rv)
        else: hu = False; mr = 0
        if hu: hu_count += 1
        max_fans.append(mr)
    return {'hu_rate': hu_count / n_games * 100, 'avg_max_fan': float(np.mean(max_fans)),
            'avg_length': float(np.mean(lengths))}

# ═══════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════
torch.manual_seed(42); np.random.seed(42)
model = CNNModelVar(in_channels=6)
sd_rl = torch.load(sl_path, map_location='cpu', weights_only=True)
model.load_sl_tower(sd_rl, 6)
model = model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

AgentCls = make_agent_cls(6)
pool = ModelPool(5)
pool.push(model.state_dict())

eval_hist = []; best_hu = 0; best_ep = 0
t0 = time.time()

for ep in range(N_EP):
    env = MahjongGBEnv(config={'agent_clz': AgentCls})
    obs_dict = env.reset(); names = env.agent_names
    traj = {a: {'obs': [], 'mask': [], 'act': [], 'rew': [], 'val': [], 'sl_logits': []} for a in names}
    done = False; term_r = None

    while not done:
        actions, values = {}, {}
        for a in obs_dict:
            oa = obs_dict[a]
            traj[a]['obs'].append(oa['observation']); traj[a]['mask'].append(oa['action_mask'])
            ot = torch.tensor(oa['observation'], dtype=torch.float).unsqueeze(0).to(device)
            mt = torch.tensor(oa['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                logits, value = model({'observation': ot, 'action_mask': mt})
                sl_logits = sl_anchor(ot, mt)
                act = Categorical(logits=logits).sample().item()
            actions[a] = act; values[a] = value.item()
            traj[a]['act'].append(act); traj[a]['val'].append(value.item())
            traj[a]['sl_logits'].append(sl_logits.detach().cpu())
        next_obs, rewards, done = env.step(actions)
        for a in rewards: traj[a]['rew'].append(rewards[a])
        if done: term_r = rewards
        obs_dict = next_obs

    # PPO + Anchor
    all_o, all_m, all_a, all_adv, all_tgt = [], [], [], [], []
    for a in names:
        d = traj[a]
        if not d['act']: continue
        n = len(d['act']); rw = d['rew'][:n] if len(d['rew']) >= n else (d['rew'] + [0])[:n]
        vl = d['val'][:n]; nv = d['val'][1:] + [0]
        td = np.array(rw) + GAMMA * np.array(nv); tdd = td - np.array(vl)
        advs = []; adv = 0.0
        for delta in reversed(tdd): adv = GAMMA * LAM * adv + delta; advs.append(adv)
        advs = np.array(advs[::-1], dtype=np.float32)
        all_o.append(np.stack(d['obs'])); all_m.append(np.stack(d['mask']))
        all_a.append(np.array(d['act'], dtype=np.int64)); all_adv.append(advs)
        all_tgt.append(td.astype(np.float32))

    if not all_o: continue
    oa = np.concatenate(all_o); ma = np.concatenate(all_m); aa = np.concatenate(all_a)
    dva = np.concatenate(all_adv); ta = np.concatenate(all_tgt)
    dva = (dva - dva.mean()) / (dva.std() + 1e-8)

    idx = np.random.permutation(len(aa))
    for s in range(0, len(aa), BS):
        ix = idx[s:s+BS]
        ob = torch.tensor(oa[ix], dtype=torch.float).to(device)
        mb = torch.tensor(ma[ix], dtype=torch.float).to(device)
        ab = torch.tensor(aa[ix]).unsqueeze(-1).to(device)
        db = torch.tensor(dva[ix], dtype=torch.float).to(device)
        tb = torch.tensor(ta[ix], dtype=torch.float).to(device)
        model.train()
        with torch.no_grad():
            ol, _ = model({'observation': ob, 'action_mask': mb})
            olp = torch.log(F.softmax(ol, dim=1).gather(1, ab) + 1e-8)
        for _ in range(PPE):
            logits, values = model({'observation': ob, 'action_mask': mb})
            probs = F.softmax(logits, dim=1)
            lp = torch.log(probs.gather(1, ab) + 1e-8); ratio = torch.exp(lp - olp)
            s1 = ratio * db; s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * db
            pl = -torch.mean(torch.min(s1, s2))
            vl = torch.mean(F.mse_loss(values.squeeze(-1), tb))
            el = -torch.mean(Categorical(probs=probs).entropy())
            # Anchor KL
            with torch.no_grad():
                sl_l = sl_anchor(ob, mb); sl_p = F.softmax(sl_l, dim=1).detach()
            kl = torch.sum(probs * (torch.log(probs + 1e-8) - torch.log(sl_p + 1e-8)), dim=1).mean()
            al = kl * ANCHOR_COEFF
            loss = pl + VC * vl + EC * el + al
            opt.zero_grad(); loss.backward(); opt.step()

    if (ep + 1) % 200 == 0:
        e = time.time() - t0
        print(f'Ep{ep+1}/{N_EP} | al={al.item():.4f} | {e:.0f}s', flush=True)

    # Eval + Checkpoint
    if (ep + 1) % EVAL_EVERY == 0:
        ev = evaluate(model, AgentCls, EVAL_GAMES)
        ev['ep'] = ep + 1; eval_hist.append(ev)
        print(f'  EVAL Ep{ep+1}: hu={ev["hu_rate"]:.1f}% fan={ev["avg_max_fan"]:.1f}', flush=True)
        if ev['hu_rate'] >= best_hu:
            best_hu = ev['hu_rate']; best_ep = ep + 1
            torch.save({'model': model.state_dict(), 'ep': ep+1, 'eval_hu_rate': best_hu},
                       os.path.join(OUT, 'best_model.pt'))
            print(f'  >>> BEST: hu={best_hu:.1f}%', flush=True)
        pool.push(model.state_dict())

    # Checkpoint every 2000
    if (ep + 1) % 2000 == 0:
        ckpt_path = os.path.join(OUT, f'ckpt_ep{ep+1}.pt')
        torch.save({'model': model.state_dict(), 'ep': ep + 1, 'eval_hist': eval_hist}, ckpt_path)

# Final
elapsed = time.time() - t0
final_ev = evaluate(model, AgentCls, 100)
report = {'best_ep': best_ep, 'best_hu': best_hu, 'final_100game': final_ev,
          'eval_hist': eval_hist, 'elapsed_sec': elapsed, 'n_ep': N_EP}
with open(os.path.join(OUT, 'report.json'), 'w') as f: json.dump(report, f, indent=2)
print(f'DONE. Best Ep{best_ep} hu={best_hu:.1f}%. Final 100g hu={final_ev["hu_rate"]:.1f}%', flush=True)
