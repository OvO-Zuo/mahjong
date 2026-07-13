"""
Adaptive Entropy Regularization (Suphx technique + SL anchor + eval + checkpoint).
Compares with enhanced baseline to isolate this technique's effect.
"""
import os, sys, time, json, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '../SL')
import torch, torch.nn.functional as F
from torch.distributions import Categorical

from env import MahjongGBEnv
from model_var import CNNModelVar
from feature_agent_ext import make_agent_cls

device = 'cuda'; SL_PATH = '../SL/model/checkpoint/model_20.pt'
OUT = 'adaptive_entropy_out'; os.makedirs(OUT, exist_ok=True)

# SL Anchor
class SLAnchor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._tower = torch.nn.Sequential(
            torch.nn.Conv2d(6,64,3,1,1,bias=False), torch.nn.ReLU(True),
            torch.nn.Conv2d(64,64,3,1,1,bias=False), torch.nn.ReLU(True),
            torch.nn.Conv2d(64,64,3,1,1,bias=False), torch.nn.ReLU(True),
            torch.nn.Flatten(), torch.nn.Linear(64*36,256), torch.nn.ReLU(True),
            torch.nn.Linear(256,235),
        )
    def forward(self, obs, mask):
        return self._tower(obs) + torch.clamp(torch.log(mask+1e-8), -1e38, 1e38)

sl_anchor = SLAnchor().to(device)
sl_anchor.load_state_dict(torch.load(SL_PATH, map_location='cpu', weights_only=True))
sl_anchor.eval()
for p in sl_anchor.parameters(): p.requires_grad = False

# Config — identical to enhanced baseline except adaptive entropy
N_EP = 2000; EVAL_EVERY = 200; EVAL_GAMES = 30
LR = 5e-5; CLIP = 0.1; ANCHOR_COEFF = 0.05; GAMMA = 0.98; LAM = 0.95
VC = 1.0; PPE = 4; BS = 256
TARGET_ENTROPY = 0.5; ENTROPY_STEP = 0.002

AgentCls = make_agent_cls(6)

torch.manual_seed(42); np.random.seed(42)
model = CNNModelVar(in_channels=6)
sd = torch.load(SL_PATH, map_location='cpu', weights_only=True)
model.load_sl_tower(sd, 6)
model = model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

eval_hist = []; best_hu = 0; best_ep = 0
entropy_coeff = 0.01
t0 = time.time()

for ep in range(N_EP):
    env = MahjongGBEnv(config={'agent_clz': AgentCls})
    obs_dict = env.reset(); names = env.agent_names
    traj = {a: {'obs':[],'mask':[],'act':[],'rew':[],'val':[]} for a in names}
    done = False; term_r = None; total_ent = 0; n_act = 0

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
                dist = Categorical(logits=logits)
                act = dist.sample().item(); total_ent += dist.entropy().item(); n_act += 1
            actions[a] = act; values[a] = value.item()
            traj[a]['act'].append(act); traj[a]['val'].append(value.item())
        next_obs, rewards, done = env.step(actions)
        for a in rewards: traj[a]['rew'].append(rewards[a])
        if done: term_r = rewards
        obs_dict = next_obs

    # Adaptive entropy (Suphx Eq.3)
    empir_ent = total_ent / max(n_act, 1)
    if empir_ent < TARGET_ENTROPY:
        entropy_coeff = min(0.1, entropy_coeff + ENTROPY_STEP)
    else:
        entropy_coeff = max(0.001, entropy_coeff - ENTROPY_STEP)

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

    idx = np.random.permutation(len(aa)); total_al = 0
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
            probs = F.softmax(logits, dim=1); lp = torch.log(probs.gather(1, ab) + 1e-8)
            ratio = torch.exp(lp - olp); s1 = ratio * db; s2 = torch.clamp(ratio, 1-CLIP, 1+CLIP) * db
            pl = -torch.mean(torch.min(s1, s2)); vl = torch.mean(F.mse_loss(values.squeeze(-1), tb))
            el = -torch.mean(Categorical(probs=probs).entropy())
            with torch.no_grad():
                sl_l = sl_anchor(ob, mb); sl_p = F.softmax(sl_l, dim=1).detach()
            kl = torch.sum(probs * (torch.log(probs+1e-8) - torch.log(sl_p+1e-8)), dim=1).mean()
            al = kl * ANCHOR_COEFF; total_al += al.item()
            loss = pl + VC * vl + entropy_coeff * el + al
            opt.zero_grad(); loss.backward(); opt.step()

    if (ep+1) % 100 == 0:
        e = time.time() - t0
        print(f'Ep{ep+1}/{N_EP} | al={total_al:.4f} ec={entropy_coeff:.3f} ent={empir_ent:.2f} | {e:.0f}s', flush=True)

    # Eval
    if (ep+1) % EVAL_EVERY == 0:
        env_eval = MahjongGBEnv(config={'agent_clz': AgentCls})
        hu = 0; fans = []
        for _ in range(EVAL_GAMES):
            od = env_eval.reset(); d = False; tr = None
            while not d:
                ac = {}
                for a in od:
                    ot = torch.tensor(od[a]['observation'], dtype=torch.float).unsqueeze(0).to(device)
                    mt = torch.tensor(od[a]['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits, _ = model({'observation': ot, 'action_mask': mt})
                        ac[a] = logits.argmax(dim=1).item()
                nd, rw, d = env_eval.step(ac)
                if d: tr = rw
                od = nd
            if tr: rv = list(tr.values()); h = max(rv) > 0; mr = max(rv)
            else: h = False; mr = 0
            if h: hu += 1
            fans.append(mr)
        ev_hu = hu / EVAL_GAMES * 100; ev_fan = float(np.mean(fans))
        eval_hist.append({'ep': ep+1, 'hu_rate': ev_hu, 'avg_fan': ev_fan})
        print(f'  EVAL Ep{ep+1}: hu={ev_hu:.1f}% fan={ev_fan:.1f} ec={entropy_coeff:.3f}', flush=True)
        if ev_hu >= best_hu: best_hu = ev_hu; best_ep = ep+1
        torch.save({'model': model.state_dict(), 'ep': ep+1, 'hu': ev_hu}, os.path.join(OUT, f'best_model.pt'))

    torch.save(model.state_dict(), os.path.join(OUT, 'final_model.pt'))

# 100-game final eval
env_f = MahjongGBEnv(config={'agent_clz': AgentCls})
hu_f = 0; fans_f = []
for _ in range(100):
    od = env_f.reset(); d = False; tr = None
    while not d:
        ac = {}
        for a in od:
            ot = torch.tensor(od[a]['observation'], dtype=torch.float).unsqueeze(0).to(device)
            mt = torch.tensor(od[a]['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model({'observation': ot, 'action_mask': mt})
                ac[a] = logits.argmax(dim=1).item()
        nd, rw, d = env_f.step(ac)
        if d: tr = rw
        od = nd
    if tr: rv = list(tr.values()); h = max(rv) > 0; mr = max(rv)
    else: h = False; mr = 0
    if h: hu_f += 1
    fans_f.append(mr)

final_hu = hu_f / 100 * 100
report = {
    'technique': 'adaptive_entropy',
    'target_entropy': TARGET_ENTROPY,
    'best_ep': best_ep, 'best_hu': best_hu,
    'final_100game_hu': final_hu, 'final_100game_fan': float(np.mean(fans_f)),
    'eval_hist': eval_hist,
}
with open(os.path.join(OUT, 'report.json'), 'w') as f: json.dump(report, f, indent=2)
print(f'\nDONE. Best Ep{best_ep} hu={best_hu:.1f}%. Final 100g hu={final_hu:.1f}%', flush=True)
