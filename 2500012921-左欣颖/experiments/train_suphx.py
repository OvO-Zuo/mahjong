"""
Suphx-inspired optimizations:
  1. Oracle Guiding — Train with perfect info (21ch), gradually dropout → 6ch
  2. Adaptive Entropy — Dynamic entropy_coeff to maintain target entropy
  3. Separate Action Heads — Structured output matching game semantics
"""
import os, sys, time, json, numpy as np
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '../SL')

import torch, torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from env import MahjongGBEnv
from feature_agent_ext import make_agent_cls
try: from MahjongGB import MahjongFanCalculator
except: raise SystemExit('MahjongGB required!')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
SL_PATH = '../SL/model/checkpoint/model_20.pt'
OUT = 'suphx_output'; os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════
# MODEL WITH SEPARATE ACTION HEADS
# ═══════════════════════════════════════════════════════
class SuphxModel(nn.Module):
    """CNN + separate heads for each action type."""
    def __init__(self, in_channels=21):
        super().__init__()
        self.in_channels = in_channels
        self.tower = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 36, 256), nn.ReLU(True),
        )
        # Separate action heads (match the action space structure)
        self.head_pass_hu = nn.Linear(256, 2)    # action 0-1
        self.head_discard = nn.Linear(256, 34)    # action 2-35
        self.head_chi = nn.Linear(256, 63)        # action 36-98
        self.head_peng = nn.Linear(256, 34)       # action 99-132
        self.head_gang = nn.Linear(256, 34)       # action 133-166
        self.head_angang = nn.Linear(256, 34)     # action 167-200
        self.head_bugang = nn.Linear(256, 34)     # action 201-234
        # Value head
        self.value_head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, 1))
        # Init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        h = self.tower(obs)
        # Concatenate all action heads
        logits = torch.cat([
            self.head_pass_hu(h),
            self.head_discard(h),
            self.head_chi(h),
            self.head_peng(h),
            self.head_gang(h),
            self.head_angang(h),
            self.head_bugang(h),
        ], dim=1)
        inf = torch.clamp(torch.log(mask + 1e-8), -1e38, 1e38)
        masked = logits + inf
        value = self.value_head(h)
        return masked, value

# ═══════════════════════════════════════════════════════
# SL ANCHOR
# ═══════════════════════════════════════════════════════
class SLAnchor(nn.Module):
    def __init__(self):
        super().__init__()
        self._tower = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64*36, 256), nn.ReLU(True),
            nn.Linear(256, 235),
        )
    def forward(self, obs, mask):
        x = self._tower(obs)
        return x + torch.clamp(torch.log(mask + 1e-8), -1e38, 1e38)

# Load SL anchor
sl_anchor = SLAnchor().to(device)
sl_sd = torch.load(SL_PATH, map_location='cpu', weights_only=True)
sl_anchor.load_state_dict(sl_sd)
sl_anchor.eval()
for p in sl_anchor.parameters(): p.requires_grad = False
print('SL anchor loaded', flush=True)

# ═══════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════
N_EP = 2000; EVAL_EVERY = 200; EVAL_GAMES = 30
LR = 5e-5; CLIP = 0.1; GAMMA = 0.98; LAM = 0.95; BS = 256
ANCHOR_COEFF = 0.05; VC = 1.0; PPE = 4
TARGET_ENTROPY = 0.5
ENTROPY_LR = 0.002  # adaptive entropy (faster adjustment)

# Suphx-lite: use 6ch baseline (no oracle guiding at this scale)
# Oracle guiding needs ~1.5M games; we adapt by keeping the other 2 techniques
full_agent_cls = make_agent_cls(6)
eval_agent_cls = make_agent_cls(6)
IN_CHANNELS = 6

torch.manual_seed(42); np.random.seed(42)

# Init model with 6ch (matches SL, no oracle guiding at our scale)
model = SuphxModel(in_channels=IN_CHANNELS)

# Load SL tower weights directly
own_sd = model.state_dict()
for k, v in sl_sd.items():
    own_k = k.replace('_tower.', 'tower.')
    if own_k in own_sd and v.shape == own_sd[own_k].shape:
        own_sd[own_k] = v.clone()
model.load_state_dict(own_sd, strict=False)
model = model.to(device)
print(f'Model: SuphxModel {IN_CHANNELS}ch, params={sum(p.numel() for p in model.parameters()):,}', flush=True)

opt = torch.optim.Adam(model.parameters(), lr=LR)

# Metrics
eval_hist = []; best_hu = 0; best_ep = 0
entropy_coeff = 0.01
t0 = time.time()

for ep in range(N_EP):
    env = MahjongGBEnv(config={'agent_clz': full_agent_cls})
    obs_dict = env.reset(); names = env.agent_names
    traj = {a: {'obs':[], 'mask':[], 'act':[], 'rew':[], 'val':[]} for a in names}
    done = False; term_r = None; total_entropy = 0; n_actions = 0

    while not done:
        actions, values = {}, {}
        for a in obs_dict:
            oa = obs_dict[a]
            traj[a]['obs'].append(oa['observation'])
            traj[a]['mask'].append(oa['action_mask'])

            ot = torch.tensor(oa['observation'], dtype=torch.float).unsqueeze(0).to(device)
            mt = torch.tensor(oa['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                logits, value = model(ot, mt)
                dist = Categorical(logits=logits)
                act = dist.sample().item()
                total_entropy += dist.entropy().item()
                n_actions += 1
            actions[a] = act; values[a] = value.item()
            traj[a]['act'].append(act); traj[a]['val'].append(value.item())
        next_obs, rewards, done = env.step(actions)
        for a in rewards: traj[a]['rew'].append(rewards[a])
        if done: term_r = rewards
        obs_dict = next_obs

    # Adaptive entropy adjustment
    empir_entropy = total_entropy / max(n_actions, 1)
    if empir_entropy < TARGET_ENTROPY:
        entropy_coeff = min(0.1, entropy_coeff + ENTROPY_LR)
    else:
        entropy_coeff = max(0.001, entropy_coeff - ENTROPY_LR)

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
    total_al = 0
    for s in range(0, len(aa), BS):
        ix = idx[s:s+BS]
        ob = torch.tensor(oa[ix], dtype=torch.float).to(device)
        mb = torch.tensor(ma[ix], dtype=torch.float).to(device)
        ab = torch.tensor(aa[ix]).unsqueeze(-1).to(device)
        db = torch.tensor(dva[ix], dtype=torch.float).to(device)
        tb = torch.tensor(ta[ix], dtype=torch.float).to(device)
        model.train()
        with torch.no_grad():
            ol, _ = model(ob, mb)
            olp = torch.log(F.softmax(ol, dim=1).gather(1, ab) + 1e-8)
        for _ in range(PPE):
            logits, values = model(ob, mb)
            probs = F.softmax(logits, dim=1); lp = torch.log(probs.gather(1, ab) + 1e-8)
            ratio = torch.exp(lp - olp); s1 = ratio * db; s2 = torch.clamp(ratio, 1-CLIP, 1+CLIP) * db
            pl = -torch.mean(torch.min(s1, s2)); vl = torch.mean(F.mse_loss(values.squeeze(-1), tb))
            el = -torch.mean(Categorical(probs=probs).entropy())
            # Anchor KL (both models are 6ch)
            with torch.no_grad():
                sl_l = sl_anchor(ob, mb); sl_p = F.softmax(sl_l, dim=1).detach()
            kl = torch.sum(probs * (torch.log(probs + 1e-8) - torch.log(sl_p + 1e-8)), dim=1).mean()
            al = kl * ANCHOR_COEFF; total_al += al.item()
            loss = pl + VC * vl + entropy_coeff * el + al
            opt.zero_grad(); loss.backward(); opt.step()

    if (ep+1) % 50 == 0:
        e = time.time() - t0
        print(f'Ep{ep+1}/{N_EP} | al={total_al:.4f} ec={entropy_coeff:.3f} '
              f'ent={empir_entropy:.2f} | {e:.0f}s', flush=True)

    # Eval (6ch agent, deterministic argmax)
    if (ep+1) % EVAL_EVERY == 0:
        env_eval = MahjongGBEnv(config={'agent_clz': eval_agent_cls})
        hu_count = 0; fans = []
        for _ in range(EVAL_GAMES):
            obs_dict = env_eval.reset(); done = False; term_r = None
            while not done:
                actions = {}
                for a in obs_dict:
                    ot = torch.tensor(obs_dict[a]['observation'], dtype=torch.float).unsqueeze(0).to(device)
                    mt = torch.tensor(obs_dict[a]['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits, _ = model(ot, mt)
                        act = logits.argmax(dim=1).item()
                    actions[a] = act
                next_obs, rewards, done = env_eval.step(actions)
                if done: term_r = rewards
                obs_dict = next_obs
            if term_r:
                rv = list(term_r.values()); hu = max(rv) > 0; mr = max(rv)
            else: hu = False; mr = 0
            if hu: hu_count += 1
            fans.append(mr)
        ev_hu = hu_count / EVAL_GAMES * 100; ev_fan = float(np.mean(fans))
        eval_hist.append({'ep': ep+1, 'hu_rate': ev_hu, 'avg_fan': ev_fan})
        print(f'  EVAL Ep{ep+1}: hu={ev_hu:.1f}% fan={ev_fan:.1f} (6ch, NO oracle)', flush=True)
        if ev_hu >= best_hu:
            best_hu = ev_hu; best_ep = ep+1
            torch.save({'model': model.state_dict(), 'ep': ep+1, 'hu': best_hu},
                       os.path.join(OUT, 'best_model.pt'))
            print(f'  >>> BEST: hu={best_hu:.1f}%', flush=True)

# Final eval (100 games)
env_final = MahjongGBEnv(config={'agent_clz': eval_agent_cls})
hu_f = 0; fans_f = []
for _ in range(100):
    obs_dict = env_final.reset(); done = False; term_r = None
    while not done:
        actions = {}
        for a in obs_dict:
            ot = torch.tensor(obs_dict[a]['observation'], dtype=torch.float).unsqueeze(0).to(device)
            mt = torch.tensor(obs_dict[a]['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(ot, mt); act = logits.argmax(dim=1).item()
            actions[a] = act
        next_obs, rewards, done = env_final.step(actions)
        if done: term_r = rewards
        obs_dict = next_obs
    if term_r:
        rv = list(term_r.values()); hu = max(rv) > 0; mr = max(rv)
    else: hu = False; mr = 0
    if hu: hu_f += 1
    fans_f.append(mr)

final_hu = hu_f / 100 * 100; final_fan = float(np.mean(fans_f))
report = {
    'techniques': ['adaptive_entropy', 'separate_heads', 'sl_anchor'],
    'best_ep': best_ep, 'best_hu': best_hu,
    'final_100game_hu': final_hu, 'final_100game_fan': final_fan,
    'eval_hist': eval_hist,
    'target_entropy': TARGET_ENTROPY,
}
with open(os.path.join(OUT, 'report.json'), 'w') as f: json.dump(report, f, indent=2)
print(f'\nDONE. Best Ep{best_ep} hu={best_hu:.1f}%. Final 100g hu={final_hu:.1f}% fan={final_fan:.1f}', flush=True)

if __name__ == '__main__':
    pass  # training runs at module level
