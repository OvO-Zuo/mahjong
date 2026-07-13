"""Final best config: lr=1e-5, anchor=0.05, 2000 episodes, checkpoint selection."""
import os, sys, time, json, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '../SL')
import torch, torch.nn.functional as F
from torch.distributions import Categorical
from env import MahjongGBEnv
from model_var import CNNModelVar
from feature_agent_ext import make_agent_cls

device = 'cuda'; SL_PATH = '../SL/model/checkpoint/model_20.pt'
OUT = 'final_best'
os.makedirs(OUT, exist_ok=True)

# SL Anchor
class SA(torch.nn.Module):
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
        return self._tower(obs) + torch.clamp(torch.log(mask + 1e-8), -1e38, 1e38)

sl_anchor = SA().to(device)
sl_anchor.load_state_dict(torch.load(SL_PATH, map_location='cpu', weights_only=True))
sl_anchor.eval()
for p in sl_anchor.parameters():
    p.requires_grad = False

AgentCls = make_agent_cls(6)
N_EP = 2000
EVAL_EVERY = 200
EVAL_GAMES = 30
LR = 1e-5
CLIP = 0.1
ANCHOR_COEFF = 0.05
GAMMA = 0.98
LAM = 0.95
VC = 1.0
EC = 0.01
PPE = 4
BS = 256

torch.manual_seed(42)
np.random.seed(42)

model = CNNModelVar(in_channels=6)
sd = torch.load(SL_PATH, map_location='cpu', weights_only=True)
model.load_sl_tower(sd, 6)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

eval_hist = []
best_hu = 0
best_ep = 0
t0 = time.time()

for ep in range(N_EP):
    env = MahjongGBEnv(config={'agent_clz': AgentCls})
    obs_dict = env.reset()
    agent_names = env.agent_names

    traj = {name: {'obs': [], 'mask': [], 'act': [], 'rew': [], 'val': []}
            for name in agent_names}
    done = False
    term_r = None

    while not done:
        actions = {}
        values = {}
        for name in obs_dict:
            oa = obs_dict[name]
            traj[name]['obs'].append(oa['observation'])
            traj[name]['mask'].append(oa['action_mask'])
            ot = torch.tensor(oa['observation'], dtype=torch.float).unsqueeze(0).to(device)
            mt = torch.tensor(oa['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                logits, value = model({'observation': ot, 'action_mask': mt})
                act = Categorical(logits=logits).sample().item()
            actions[name] = act
            values[name] = value.item()
            traj[name]['act'].append(act)
            traj[name]['val'].append(value.item())
        next_obs, rewards, done = env.step(actions)
        for name in rewards:
            traj[name]['rew'].append(rewards[name])
        if done:
            term_r = rewards
        obs_dict = next_obs

    # PPO + Anchor
    all_obs, all_mask, all_act, all_adv, all_tgt = [], [], [], [], []
    for name in agent_names:
        d = traj[name]
        if not d['act']:
            continue
        n = len(d['act'])
        rews = d['rew'][:n] if len(d['rew']) >= n else (d['rew'] + [0])[:n]
        vals = d['val'][:n]
        next_vals = d['val'][1:] + [0]
        td = np.array(rews) + GAMMA * np.array(next_vals)
        td_delta = td - np.array(vals)
        advs = []
        adv = 0.0
        for delta in reversed(td_delta):
            adv = GAMMA * LAM * adv + delta
            advs.append(adv)
        advs = np.array(advs[::-1], dtype=np.float32)
        all_obs.append(np.stack(d['obs']))
        all_mask.append(np.stack(d['mask']))
        all_act.append(np.array(d['act'], dtype=np.int64))
        all_adv.append(advs)
        all_tgt.append(td.astype(np.float32))

    if not all_obs:
        continue

    X_obs = np.concatenate(all_obs)
    X_mask = np.concatenate(all_mask)
    X_act = np.concatenate(all_act)
    X_adv = np.concatenate(all_adv)
    X_tgt = np.concatenate(all_tgt)
    X_adv = (X_adv - X_adv.mean()) / (X_adv.std() + 1e-8)

    indices = np.random.permutation(len(X_act))
    for s in range(0, len(X_act), BS):
        ix = indices[s:s + BS]
        ob = torch.tensor(X_obs[ix], dtype=torch.float).to(device)
        mb = torch.tensor(X_mask[ix], dtype=torch.float).to(device)
        ab = torch.tensor(X_act[ix]).unsqueeze(-1).to(device)
        db = torch.tensor(X_adv[ix], dtype=torch.float).to(device)
        tb = torch.tensor(X_tgt[ix], dtype=torch.float).to(device)
        model.train()
        with torch.no_grad():
            old_logits, _ = model({'observation': ob, 'action_mask': mb})
            old_lp = torch.log(F.softmax(old_logits, dim=1).gather(1, ab) + 1e-8)
        for _ in range(PPE):
            logits, values = model({'observation': ob, 'action_mask': mb})
            probs = F.softmax(logits, dim=1)
            lp = torch.log(probs.gather(1, ab) + 1e-8)
            ratio = torch.exp(lp - old_lp)
            s1 = ratio * db
            s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * db
            policy_loss = -torch.mean(torch.min(s1, s2))
            value_loss = torch.mean(F.mse_loss(values.squeeze(-1), tb))
            entropy_loss = -torch.mean(Categorical(probs=probs).entropy())
            with torch.no_grad():
                sl_l = sl_anchor(ob, mb)
                sl_p = F.softmax(sl_l, dim=1).detach()
            kl = torch.sum(probs * (torch.log(probs + 1e-8) - torch.log(sl_p + 1e-8)), dim=1).mean()
            anchor_loss = kl * ANCHOR_COEFF
            loss = policy_loss + VC * value_loss + EC * entropy_loss + anchor_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Eval
    if (ep + 1) % EVAL_EVERY == 0:
        eval_env = MahjongGBEnv(config={'agent_clz': AgentCls})
        hu_count = 0
        fans = []
        for _ in range(EVAL_GAMES):
            od = eval_env.reset()
            d = False
            tr = None
            while not d:
                ac = {}
                for a in od:
                    ot = torch.tensor(od[a]['observation'], dtype=torch.float).unsqueeze(0).to(device)
                    mt = torch.tensor(od[a]['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits, _ = model({'observation': ot, 'action_mask': mt})
                        ac[a] = logits.argmax(dim=1).item()
                nd, rw, d = eval_env.step(ac)
                if d:
                    tr = rw
                od = nd
            if tr:
                rv = list(tr.values())
                h = max(rv) > 0
                mr = max(rv)
            else:
                h = False
                mr = 0
            if h:
                hu_count += 1
            fans.append(mr)
        ev_hu = hu_count / EVAL_GAMES * 100
        ev_fan = float(np.mean(fans))
        eval_hist.append({'ep': ep + 1, 'hu_rate': ev_hu, 'avg_fan': ev_fan})
        elapsed = time.time() - t0
        print(f'EVAL Ep{ep + 1}: hu={ev_hu:.1f}% fan={ev_fan:.1f} | {elapsed:.0f}s', flush=True)
        if ev_hu >= best_hu:
            best_hu = ev_hu
            best_ep = ep + 1
            torch.save({'model': model.state_dict(), 'ep': ep + 1, 'hu': best_hu},
                       os.path.join(OUT, 'best_model.pt'))
            print(f'  >>> BEST: hu={best_hu:.1f}%', flush=True)

# Final 100-game eval
eval_env = MahjongGBEnv(config={'agent_clz': AgentCls})
hu_100 = 0
fans_100 = []
for _ in range(100):
    od = eval_env.reset()
    d = False
    tr = None
    while not d:
        ac = {}
        for a in od:
            ot = torch.tensor(od[a]['observation'], dtype=torch.float).unsqueeze(0).to(device)
            mt = torch.tensor(od[a]['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model({'observation': ot, 'action_mask': mt})
                ac[a] = logits.argmax(dim=1).item()
        nd, rw, d = eval_env.step(ac)
        if d:
            tr = rw
        od = nd
    if tr:
        rv = list(tr.values())
        h = max(rv) > 0
        mr = max(rv)
    else:
        h = False
        mr = 0
    if h:
        hu_100 += 1
    fans_100.append(mr)

final_hu = hu_100 / 100 * 100
final_fan = float(np.mean(fans_100))
print(f'\nFINAL: Best Ep{best_ep} hu={best_hu:.1f}%. 100-game hu={final_hu:.1f}% fan={final_fan:.1f}', flush=True)

report = {
    'best_ep': best_ep, 'best_hu': best_hu,
    'final_100game_hu': final_hu, 'final_100game_fan': final_fan,
    'eval_hist': eval_hist, 'lr': LR, 'clip': CLIP, 'anchor_coeff': ANCHOR_COEFF,
}
with open(os.path.join(OUT, 'report.json'), 'w') as f:
    json.dump(report, f, indent=2)
