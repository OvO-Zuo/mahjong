"""
Hyperparameter optimization: test anchor_coeff, entropy, lr, annealing.
Runs 1500ep each, eval every 300ep, records best eval hu_rate.
"""
import os, sys, time, json, copy, numpy as np
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '../SL')

import torch, torch.nn.functional as F
from torch.distributions import Categorical

from env import MahjongGBEnv
from model_var import CNNModelVar
from feature_agent_ext import make_agent_cls
try: from MahjongGB import MahjongFanCalculator
except: raise SystemExit('MahjongGB required!')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
SL_PATH = '../SL/model/checkpoint/model_20.pt'
OUT_DIR = 'optimize_output'; os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════
# SL Anchor
# ═══════════════════════════════════════════════
class SLAnchor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._tower = torch.nn.Sequential(
            torch.nn.Conv2d(6,64,3,1,1,bias=False), torch.nn.ReLU(True),
            torch.nn.Conv2d(64,64,3,1,1,bias=False), torch.nn.ReLU(True),
            torch.nn.Conv2d(64,64,3,1,1,bias=False), torch.nn.ReLU(True),
            torch.nn.Flatten(),
            torch.nn.Linear(64*36,256), torch.nn.ReLU(True),
            torch.nn.Linear(256,235),
        )
    def forward(self, obs, mask):
        x = self._tower(obs)
        inf = torch.clamp(torch.log(mask + 1e-8), -1e38, 1e38)
        return x + inf

sl_anchor = SLAnchor().to(device)
sd = torch.load(SL_PATH, map_location='cpu', weights_only=True)
# Fix key mismatch: SL model.pt has _tower.X keys, SLAnchor has same
sl_anchor.load_state_dict(sd)
sl_anchor.eval()
for p in sl_anchor.parameters(): p.requires_grad = False

# ═══════════════════════════════════════════════
# Shared
# ═══════════════════════════════════════════════
AgentCls = make_agent_cls(6)
GAMMA=0.98; LAM=0.95; CLIP=0.1; VC=1.0; PPE=4; BS=256
N_EP=1000; EVAL_EVERY=200; EVAL_GAMES=30

def evaluate(model):
    model.eval()
    env = MahjongGBEnv(config={'agent_clz': AgentCls})
    hu=0; fans=[]; lengths=[]
    for _ in range(EVAL_GAMES):
        obs_dict=env.reset(); done=False; ep_len=0; term_r=None
        while not done:
            actions={}
            for a in obs_dict:
                ot=torch.tensor(obs_dict[a]['observation'],dtype=torch.float).unsqueeze(0).to(device)
                mt=torch.tensor(obs_dict[a]['action_mask'],dtype=torch.float).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits,_=model({'observation':ot,'action_mask':mt})
                    act=logits.argmax(dim=1).item()
                actions[a]=act
            next_obs,rewards,done=env.step(actions)
            if done: term_r=rewards
            obs_dict=next_obs; ep_len+=1
        lengths.append(ep_len)
        if term_r: rv=list(term_r.values()); h=max(rv)>0; mr=max(rv)
        else: h=False; mr=0
        if h: hu+=1
        fans.append(mr)
    return {'hu_rate':hu/EVAL_GAMES*100,'avg_fan':float(np.mean(fans)),'avg_len':float(np.mean(lengths))}

# ═══════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════
def run_experiment(name, cfg):
    print(f'\n{"="*50}\n{name}\ncfg={cfg}', flush=True)
    torch.manual_seed(42); np.random.seed(42)

    anchor_coeff = cfg.get('anchor_coeff', 0.05)
    entropy_coeff = cfg.get('entropy_coeff', 0.01)
    lr = cfg.get('lr', 5e-5)
    anneal_anchor = cfg.get('anneal_anchor', False)
    anneal_entropy = cfg.get('anneal_entropy', False)
    init_anchor = cfg.get('init_anchor', 0.1)
    init_entropy = cfg.get('init_entropy', 0.05)

    model = CNNModelVar(in_channels=6)
    sd_rl = torch.load(SL_PATH, map_location='cpu', weights_only=True)
    model.load_sl_tower(sd_rl, 6)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_hu = 0; best_ep = 0; eval_hist = []
    t0 = time.time()

    for ep in range(N_EP):
        # Annealing
        if anneal_anchor:
            cur_anchor = init_anchor * (1 - ep / N_EP) + 0.01 * (ep / N_EP)
        else:
            cur_anchor = anchor_coeff
        if anneal_entropy:
            cur_entropy = init_entropy * (1 - ep / N_EP) + 0.005 * (ep / N_EP)
        else:
            cur_entropy = entropy_coeff

        env = MahjongGBEnv(config={'agent_clz': AgentCls})
        obs_dict = env.reset(); names = env.agent_names
        traj = {a: {'obs':[],'mask':[],'act':[],'rew':[],'val':[]} for a in names}
        done=False; term_r=None

        while not done:
            actions,values={},{}
            for a in obs_dict:
                oa=obs_dict[a]
                traj[a]['obs'].append(oa['observation']); traj[a]['mask'].append(oa['action_mask'])
                ot=torch.tensor(oa['observation'],dtype=torch.float).unsqueeze(0).to(device)
                mt=torch.tensor(oa['action_mask'],dtype=torch.float).unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    logits,value=model({'observation':ot,'action_mask':mt})
                    act=Categorical(logits=logits).sample().item()
                actions[a]=act; values[a]=value.item()
                traj[a]['act'].append(act); traj[a]['val'].append(value.item())
            next_obs,rewards,done=env.step(actions)
            for a in rewards: traj[a]['rew'].append(rewards[a])
            if done: term_r=rewards
            obs_dict=next_obs

        # PPO
        all_o,all_m,all_a,all_adv,all_tgt=[],[],[],[],[]
        for a in names:
            d=traj[a]
            if not d['act']: continue
            n=len(d['act']); rw=d['rew'][:n] if len(d['rew'])>=n else (d['rew']+[0])[:n]
            vl=d['val'][:n]; nv=d['val'][1:]+[0]
            td=np.array(rw)+GAMMA*np.array(nv); tdd=td-np.array(vl)
            advs=[]; adv=0.0
            for delta in reversed(tdd): adv=GAMMA*LAM*adv+delta; advs.append(adv)
            advs=np.array(advs[::-1],dtype=np.float32)
            all_o.append(np.stack(d['obs'])); all_m.append(np.stack(d['mask']))
            all_a.append(np.array(d['act'],dtype=np.int64)); all_adv.append(advs)
            all_tgt.append(td.astype(np.float32))
        if not all_o: continue

        oa=np.concatenate(all_o); ma=np.concatenate(all_m); aa=np.concatenate(all_a)
        dva=np.concatenate(all_adv); ta=np.concatenate(all_tgt)
        dva=(dva-dva.mean())/(dva.std()+1e-8)

        idx=np.random.permutation(len(aa))
        for s in range(0,len(aa),BS):
            ix=idx[s:s+BS]
            ob=torch.tensor(oa[ix],dtype=torch.float).to(device)
            mb=torch.tensor(ma[ix],dtype=torch.float).to(device)
            ab=torch.tensor(aa[ix]).unsqueeze(-1).to(device)
            db=torch.tensor(dva[ix],dtype=torch.float).to(device)
            tb=torch.tensor(ta[ix],dtype=torch.float).to(device)
            model.train()
            with torch.no_grad():
                ol,_=model({'observation':ob,'action_mask':mb})
                olp=torch.log(F.softmax(ol,dim=1).gather(1,ab)+1e-8)
            for _ in range(PPE):
                logits,values=model({'observation':ob,'action_mask':mb})
                probs=F.softmax(logits,dim=1); lp=torch.log(probs.gather(1,ab)+1e-8)
                ratio=torch.exp(lp-olp); s1=ratio*db; s2=torch.clamp(ratio,1-CLIP,1+CLIP)*db
                pl=-torch.mean(torch.min(s1,s2)); vl=torch.mean(F.mse_loss(values.squeeze(-1),tb))
                el=-torch.mean(Categorical(probs=probs).entropy())
                with torch.no_grad():
                    sl_l=sl_anchor(ob,mb); sl_p=F.softmax(sl_l,dim=1).detach()
                kl=torch.sum(probs*(torch.log(probs+1e-8)-torch.log(sl_p+1e-8)),dim=1).mean()
                al=kl*cur_anchor
                loss=pl+VC*vl+cur_entropy*el+al
                opt.zero_grad(); loss.backward(); opt.step()

        if (ep+1)%EVAL_EVERY==0:
            ev=evaluate(model); ev['ep']=ep+1; eval_hist.append(ev)
            print(f'  EVAL Ep{ep+1}: hu={ev["hu_rate"]:.1f}% fan={ev["avg_fan"]:.1f} al={cur_anchor:.4f}',flush=True)
            if ev['hu_rate']>=best_hu: best_hu=ev['hu_rate']; best_ep=ep+1

    # Final 100-game eval for best checkpoint
    final_ev = evaluate(model)
    result = {'name':name,'cfg':cfg,'best_ep':best_ep,'best_hu':best_hu,
              'final_hu':final_ev['hu_rate'],'eval_hist':eval_hist,
              'time':time.time()-t0}
    print(f'  RESULT: best_hu={best_hu:.1f}% final_hu={final_ev["hu_rate"]:.1f}%',flush=True)
    return result

# ═══════════════════════════════════════════════
# Experiments
# ═══════════════════════════════════════════════
experiments = [
    ('A_baseline',           {'anchor_coeff':0.05, 'entropy_coeff':0.01, 'lr':5e-5}),
    ('B_anchor_0.01',        {'anchor_coeff':0.01, 'entropy_coeff':0.01, 'lr':5e-5}),
    ('C_anchor_0.1',         {'anchor_coeff':0.1,  'entropy_coeff':0.01, 'lr':5e-5}),
    ('D_anchor_0.5',         {'anchor_coeff':0.5,  'entropy_coeff':0.01, 'lr':5e-5}),
    ('E_entropy_0.05',       {'anchor_coeff':0.05, 'entropy_coeff':0.05, 'lr':5e-5}),
    ('F_lr_1e-5',            {'anchor_coeff':0.05, 'entropy_coeff':0.01, 'lr':1e-5}),
    ('G_anneal_anchor',      {'anneal_anchor':True, 'init_anchor':0.1, 'entropy_coeff':0.01, 'lr':5e-5}),
    ('H_anneal_entropy',     {'anneal_entropy':True, 'init_entropy':0.05, 'anchor_coeff':0.05, 'lr':5e-5}),
]

if __name__=='__main__':
    results=[]
    for name,cfg in experiments:
        r=run_experiment(name,cfg)
        results.append(r)

    with open(os.path.join(OUT_DIR,'optimize_results.json'),'w') as f: json.dump(results,f,indent=2)

    print(f'\n{"="*70}')
    print(f'{"Experiment":<25} {"best_hu":>8} {"final_hu":>8} {"best_ep":>8}')
    print(f'{"="*70}')
    for r in results:
        print(f'{r["name"]:<25} {r["best_hu"]:>7.1f}% {r["final_hu"]:>7.1f}% {r["best_ep"]:>8}')
    print(f'{"="*70}')
