"""
Test all features individually: each feature is a module with obs channels + optional reward.

Features: ting_obs, ting_reward, discard, meld, shown, wall, discard+meld
500 episodes each, SL init, compare hu_rate curves.
"""
import os, sys, time, json, numpy as np
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '../SL')

import torch, torch.nn.functional as F
from torch.distributions import Categorical

from env import MahjongGBEnv
from model_var import CNNModelVar
from agent import MahjongGBAgent

try:
    from MahjongGB import MahjongFanCalculator
except ImportError:
    raise SystemExit('MahjongGB required!')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EP = 1000
SL_PATH = '../SL/model/checkpoint/model_20.pt'
TILE_LIST = [*(f'W{i}' for i in range(1,10)), *(f'T{i}' for i in range(1,10)),
             *(f'B{i}' for i in range(1,10)), *(f'F{i}' for i in range(1,5)),
             *(f'J{i}' for i in range(1,4))]
T2I = {c:i for i,c in enumerate(TILE_LIST)}
ACT_OFF = {'Pass':0,'Hu':1,'Play':2,'Chi':36,'Peng':99,'Gang':133,'AnGang':167,'BuGang':201}


# ═══════════════════════════════════════════════════════════
# FEATURE MODULES
# ═══════════════════════════════════════════════════════════

FEATURES = {
    'ting_obs': {
        'name': 'ting_obs', 'channels': 1, 'reward': False,
        'desc': 'Ting indicator channel: which discards lead to waiting state',
    },
    'ting_reward': {
        'name': 'ting_reward', 'channels': 0, 'reward': True,
        'desc': '+2 intermediate reward when reaching ting state',
    },
    'discard': {
        'name': 'discard', 'channels': 4, 'reward': False,
        'desc': 'Opponent discard history (1 channel per player)',
    },
    'meld': {
        'name': 'meld', 'channels': 4, 'reward': False,
        'desc': 'Opponent visible melds (1 channel per player)',
    },
    'shown': {
        'name': 'shown', 'channels': 1, 'reward': False,
        'desc': 'Global shown tile counts (normalized)',
    },
    'wall': {
        'name': 'wall', 'channels': 1, 'reward': False,
        'desc': 'Remaining tiles in wall (normalized)',
    },
}

BASE_CH = 6  # seat wind + prevalent wind + hand×4


# ═══════════════════════════════════════════════════════════
# CONFIGURABLE AGENT
# ═══════════════════════════════════════════════════════════

def make_feature_agent(feature_names):
    """Create an agent class with specified features enabled."""
    features = [FEATURES[f] for f in feature_names if f in FEATURES]
    extra_ch = sum(f['channels'] for f in features)
    total_ch = BASE_CH + extra_ch
    has_reward = any(f['reward'] for f in features)

    # Map features to channel offsets
    offsets = {}
    ch = BASE_CH
    for f in features:
        if f['channels'] > 0:
            offsets[f['name']] = ch
            ch += f['channels']

    class FeatureAgent(MahjongGBAgent):
        observation_space = None
        action_space = None

        def __init__(self, seatWind):
            self.seatWind = seatWind
            self.total_ch = total_ch
            self.features = features
            self.offsets = offsets
            self.packs = [[] for _ in range(4)]
            self.history = [[] for _ in range(4)]
            self.tileWall = [21]*4
            self.shownTiles = defaultdict(int)
            self.wallLast = False; self.isAboutKong = False
            self.obs = np.zeros((total_ch, 36))
            self.obs[0][T2I['F%d'%(seatWind+1)]] = 1
            self._was_ting = False  # for ting_reward

        # ── Game protocol ────────────────────────────
        def request2obs(self, request):
            t = request.split()
            if t[0]=='Wind':
                self.prevalentWind=int(t[1])
                self.obs[1][T2I['F%d'%(self.prevalentWind+1)]]=1; return
            if t[0]=='Deal': self.hand=t[1:]; self._upd_hand(); return
            if t[0]=='Huang': self.valid=[]; return self._obs()
            if t[0]=='Draw': return self._handle_draw(t[1])
            p=(int(t[1])+4-self.seatWind)%4
            if t[2]=='Draw': self.tileWall[p]-=1; self.wallLast=self.tileWall[(p+1)%4]==0; return
            if t[2] in ('Invalid','Hu'): self.valid=[]; return self._obs()
            if t[2]=='Play': return self._handle_play(p, t[3])
            if t[2]=='Chi': return self._handle_chi(p, t[3])
            if t[2] in ('UnChi','UnPeng'): return
            if t[2]=='Peng': return self._handle_peng(p)
            if t[2]=='Gang': return self._handle_gang(p)
            if t[2]=='AnGang': return self._handle_angang(p, t[3] if len(t)>3 else None)
            if t[2]=='BuGang': return self._handle_bugang(p, t[3])
            raise NotImplementedError(f'Unknown: {request}')

        def _handle_draw(self, tile):
            self.tileWall[0]-=1; self.wallLast=self.tileWall[1]==0
            self.isAboutKong=False; self.hand.append(tile); self._upd_hand()
            self.valid=[]
            if self._check_mj(tile,True,self.isAboutKong): self.valid.append(1)
            for t_ in set(self.hand):
                self.valid.append(2+T2I[t_])
                if self.hand.count(t_)==4 and not self.wallLast and self.tileWall[0]>0:
                    self.valid.append(167+T2I[t_])
            if not self.wallLast and self.tileWall[0]>0:
                for pt,t_,_ in self.packs[0]:
                    if pt=='PENG' and t_ in self.hand: self.valid.append(201+T2I[t_])
            self._fill_features()
            return self._obs()

        def _handle_play(self, p, tile):
            self.tileFrom=p; self.curTile=tile
            self.shownTiles[tile]+=1; self.history[p].append(tile)
            if p==0: self.hand.remove(tile); self._upd_hand(); return
            else:
                self.valid=[]
                if self._check_mj(tile): self.valid.append(1)
                if not self.wallLast:
                    if self.hand.count(tile)>=2:
                        self.valid.append(99+T2I[tile])
                        if self.hand.count(tile)==3 and self.tileWall[0]:
                            self.valid.append(133+T2I[tile])
                    if p==3 and tile[0] in 'WTB':
                        num=int(tile[1])
                        for i,off in [(-2,2),(-1,1),(0,0)]:
                            tmp=[tile[0]+str(num+j) for j in range(i,i+3)]
                            if all(x in self.hand for x in tmp[:2]+tmp[1:]):
                                c='WTB'.index(tile[0])
                                self.valid.append(36+c*21+(num-3+i)*3+(2-off))
                self.valid.append(0)
                self._fill_features(); return self._obs()

        def _handle_chi(self, p, tile):
            c,n=tile[0],int(tile[1])
            self.packs[p].append(('CHI',tile,int(self.curTile[1])-n+2))
            self.shownTiles[self.curTile]-=1
            for i in range(-1,2): self.shownTiles[c+str(n+i)]+=1
            self.wallLast=self.tileWall[(p+1)%4]==0
            if p==0:
                self.valid=[]; self.hand.append(self.curTile)
                for i in range(-1,2): self.hand.remove(c+str(n+i))
                self._upd_hand()
                for t_ in set(self.hand): self.valid.append(2+T2I[t_])
                return self._obs()
            return

        def _handle_peng(self, p):
            self.packs[p].append(('PENG',self.curTile,(4+p-self.tileFrom)%4))
            self.shownTiles[self.curTile]+=2; self.wallLast=self.tileWall[(p+1)%4]==0
            if p==0:
                self.valid=[]; [self.hand.remove(self.curTile) for _ in range(2)]
                self._upd_hand()
                for t_ in set(self.hand): self.valid.append(2+T2I[t_])
                return self._obs()
            return

        def _handle_gang(self, p):
            self.packs[p].append(('GANG',self.curTile,(4+p-self.tileFrom)%4))
            self.shownTiles[self.curTile]+=3
            if p==0: [self.hand.remove(self.curTile) for _ in range(3)]; self._upd_hand(); self.isAboutKong=True
            return

        def _handle_angang(self, p, tile):
            t='CONCEALED' if p else tile
            self.packs[p].append(('GANG',t,0))
            if p==0: self.isAboutKong=True; [self.hand.remove(tile) for _ in range(4)]
            else: self.isAboutKong=False
            return

        def _handle_bugang(self, p, tile):
            for i in range(len(self.packs[p])):
                if tile==self.packs[p][i][1]: self.packs[p][i]=('GANG',tile,self.packs[p][i][2]); break
            self.shownTiles[tile]+=1
            if p==0: self.hand.remove(tile); self._upd_hand(); self.isAboutKong=True; return
            else:
                self.valid=[]
                if self._check_mj(tile,False,True): self.valid.append(1)
                self.valid.append(0); return self._obs()

        def action2response(self, a):
            if a<1: return 'Pass'
            if a<2: return 'Hu'
            if a<36: return 'Play '+TILE_LIST[a-2]
            if a<99: t=(a-36)//3; return 'Chi '+('WTB'[t//7])+str(t%7+2)
            if a<133: return 'Peng'
            if a<167: return 'Gang'
            if a<201: return 'Gang '+TILE_LIST[a-167]
            return 'BuGang '+TILE_LIST[a-201]

        def response2action(self, r):
            t=r.split()
            if t[0]=='Pass': return 0
            if t[0]=='Hu': return 1
            if t[0]=='Play': return 2+T2I[t[1]]
            if t[0]=='Chi': return 36+'WTB'.index(t[1][0])*21+(int(t[2][1])-2)*3+int(t[1][1])-int(t[2][1])+1
            if t[0]=='Peng': return 99+T2I[t[1]]
            if t[0]=='Gang': return 133+T2I[t[1]]
            if t[0]=='AnGang': return 167+T2I[t[1]]
            if t[0]=='BuGang': return 201+T2I[t[1]]
            return 0

        def _obs(self):
            m=np.zeros(235); [m.__setitem__(a,1) for a in self.valid]
            return {'observation':self.obs.reshape((self.total_ch,4,9)).copy(),'action_mask':m}

        def _upd_hand(self):
            self.obs[2:6]=0; d=defaultdict(int)
            for t in self.hand: d[t]+=1
            for t,c in d.items(): self.obs[2:2+c,T2I[t]]=1

        # ── Feature computation ──────────────────────
        def _fill_features(self):
            for f in self.features:
                name = f['name']
                if name=='ting_obs': self._feat_ting_obs()
                elif name=='discard': self._feat_discard()
                elif name=='meld': self._feat_meld()
                elif name=='shown': self._feat_shown()
                elif name=='wall': self._feat_wall()

        def _feat_ting_obs(self):
            ch = self.offsets['ting_obs']; self.obs[ch]=0
            if len(self.hand) not in (2,5,8,11,14): return
            for ut in set(self.hand):
                th=list(self.hand); th.remove(ut)
                for wt in set(th):
                    if wt[0] in 'FJ': continue
                    try:
                        fans=MahjongFanCalculator(pack=tuple(self.packs[0]),hand=tuple(th),
                            winTile=wt,flowerCount=0,isSelfDrawn=True,
                            is4thTile=(self.shownTiles.get(wt,0))==3,
                            isAboutKong=False,isWallLast=self.wallLast,
                            seatWind=self.seatWind,prevalentWind=self.prevalentWind,verbose=True)
                        if sum(fp*c for fp,c,_,_ in fans)>=8:
                            self.obs[ch][T2I[ut]]=1; break
                    except: pass

        def _feat_discard(self):
            ch=self.offsets['discard']
            for pi in range(4):
                self.obs[ch+pi]=0
                for t in self.history[pi]: self.obs[ch+pi][T2I[t]]=min(1.0,self.obs[ch+pi][T2I[t]]+0.1)

        def _feat_meld(self):
            ch=self.offsets['meld']
            for pi in range(4):
                self.obs[ch+pi]=0
                for pt,t,_ in self.packs[pi]:
                    if t in T2I: self.obs[ch+pi][T2I[t]]=1

        def _feat_shown(self):
            ch=self.offsets['shown']; self.obs[ch]=0
            for t,c in self.shownTiles.items():
                if t in T2I: self.obs[ch][T2I[t]]=min(c/4.0,1.0)

        def _feat_wall(self):
            ch=self.offsets['wall']
            self.obs[ch].fill(min(sum(self.tileWall)/144.0,1.0))

        def is_ting(self):
            if 'ting_obs' in self.offsets:
                return self.obs[self.offsets['ting_obs']].sum()>0
            # If no ting_obs, compute on demand
            if len(self.hand) not in (2,5,8,11,14): return False
            for ut in set(self.hand):
                th=list(self.hand); th.remove(ut)
                for wt in set(th):
                    if wt[0] in 'FJ': continue
                    try:
                        fans=MahjongFanCalculator(pack=tuple(self.packs[0]),hand=tuple(th),
                            winTile=wt,flowerCount=0,isSelfDrawn=True,
                            is4thTile=(self.shownTiles.get(wt,0))==3,
                            isAboutKong=False,isWallLast=self.wallLast,
                            seatWind=self.seatWind,prevalentWind=self.prevalentWind,verbose=True)
                        if sum(fp*c for fp,c,_,_ in fans)>=8: return True
                    except: pass
            return False

        def get_ting_reward(self):
            """+2 if just reached ting state."""
            is_t = self.is_ting()
            r = 0.0
            if is_t and not self._was_ting: r = 2.0
            self._was_ting = is_t
            return r

        def _check_mj(self, wt, sd=False, ak=False):
            try:
                fans=MahjongFanCalculator(pack=tuple(self.packs[0]),hand=tuple(self.hand),
                    winTile=wt,flowerCount=0,isSelfDrawn=sd,
                    is4thTile=(self.shownTiles.get(wt,0)+sd)==4,
                    isAboutKong=ak,isWallLast=self.wallLast,
                    seatWind=self.seatWind,prevalentWind=self.prevalentWind,verbose=True)
                if sum(fp*c for fp,c,_,_ in fans)<8: raise Exception
            except: return False
            return True

    return FeatureAgent


# ═══════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════

def run_experiment(name, feature_names, n_ep=N_EP):
    AgentCls = make_feature_agent(feature_names)
    obs_size = BASE_CH + sum(FEATURES[f]['channels'] for f in feature_names if f in FEATURES)
    has_ting_reward = 'ting_reward' in feature_names

    print(f'\n{"="*50}', flush=True)
    print(f'{name} | ch={obs_size} | features={feature_names} | {n_ep} eps', flush=True)

    torch.manual_seed(42); np.random.seed(42)
    model = CNNModelVar(in_channels=obs_size)
    if os.path.exists(SL_PATH):
        sd = torch.load(SL_PATH, map_location='cpu', weights_only=True)
        model.load_sl_tower(sd, obs_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    env = MahjongGBEnv(config={'agent_clz': AgentCls})
    names = env.agent_names
    gamma, lam, clip = 0.98, 0.95, 0.2
    vc, ec, ppe, bs = 1.0, 0.01, 4, 256

    hu_rates, max_fans, lengths = [], [], []
    t0 = time.time()

    for ep in range(n_ep):
        obs_dict = env.reset()
        traj = {a:{'obs':[],'mask':[],'act':[],'rew':[],'val':[]} for a in names}
        done, ep_len = False, 0; term_r = None

        while not done:
            actions, values = {}, {}
            for a in obs_dict:
                traj[a]['obs'].append(obs_dict[a]['observation'])
                traj[a]['mask'].append(obs_dict[a]['action_mask'])
                ot = torch.tensor(obs_dict[a]['observation'], dtype=torch.float).unsqueeze(0).to(device)
                mt = torch.tensor(obs_dict[a]['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    logits, value = model({'observation':ot,'action_mask':mt})
                    dist = Categorical(logits=logits); act = dist.sample().item()
                actions[a]=act; values[a]=value.item()
                traj[a]['act'].append(act); traj[a]['val'].append(value.item())

            next_obs, rewards, done = env.step(actions)
            for a in rewards:
                r = rewards[a]
                if has_ting_reward:
                    for ag in env.agents:
                        if hasattr(ag, 'get_ting_reward'):
                            r += ag.get_ting_reward(); break
                traj[a]['rew'].append(r)
            if done: term_r = rewards
            obs_dict = next_obs; ep_len += 1

        lengths.append(ep_len)
        if term_r:
            rv=list(term_r.values()); hu=max(rv)>0; mr=max(rv)
        else: hu=False; mr=0
        hu_rates.append(1 if hu else 0); max_fans.append(mr)

        # PPO update
        all_o,all_m,all_a,all_adv,all_tgt=[],[],[],[],[]
        for a in names:
            d=traj[a]
            if not d['act']: continue
            n=len(d['act'])
            rw=d['rew'][:n] if len(d['rew'])>=n else (d['rew']+[0])[:n]
            vl=d['val'][:n]; nv=d['val'][1:]+[0]
            td=np.array(rw)+gamma*np.array(nv); tdd=td-np.array(vl)
            advs=[]; adv=0.0
            for delta in reversed(tdd): adv=gamma*lam*adv+delta; advs.append(adv)
            advs=np.array(advs[::-1],dtype=np.float32)
            all_o.append(np.stack(d['obs'])); all_m.append(np.stack(d['mask']))
            all_a.append(np.array(d['act'],dtype=np.int64))
            all_adv.append(advs); all_tgt.append(td.astype(np.float32))

        if not all_o: continue
        oa=np.concatenate(all_o); ma=np.concatenate(all_m)
        aa=np.concatenate(all_a); dva=np.concatenate(all_adv); ta=np.concatenate(all_tgt)
        dva=(dva-dva.mean())/(dva.std()+1e-8)

        total_loss=0; nu=0
        idx=np.random.permutation(len(aa))
        for s in range(0,len(aa),bs):
            ix=idx[s:s+bs]
            ob=torch.tensor(oa[ix],dtype=torch.float).to(device)
            mb=torch.tensor(ma[ix],dtype=torch.float).to(device)
            ab=torch.tensor(aa[ix]).unsqueeze(-1).to(device)
            db=torch.tensor(dva[ix],dtype=torch.float).to(device)
            tb=torch.tensor(ta[ix],dtype=torch.float).to(device)
            model.train()
            with torch.no_grad():
                ol,_=model({'observation':ob,'action_mask':mb})
                olp=torch.log(F.softmax(ol,dim=1).gather(1,ab)+1e-8)
            for _ in range(ppe):
                l,vs=model({'observation':ob,'action_mask':mb})
                lp=torch.log(F.softmax(l,dim=1).gather(1,ab)+1e-8)
                ratio=torch.exp(lp-olp)
                s1=ratio*db; s2=torch.clamp(ratio,1-clip,1+clip)*db
                pl=-torch.mean(torch.min(s1,s2))
                vl=torch.mean(F.mse_loss(vs.squeeze(-1),tb))
                el=-torch.mean(Categorical(logits=l).entropy())
                loss=pl+vc*vl+ec*el
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss+=loss.item(); nu+=1
        avg_loss=total_loss/max(nu,1)

        if (ep+1)%50==0 or ep==0:
            e=time.time()-t0; hr=np.mean(hu_rates[-50:])*100; af=np.mean(max_fans[-50:])
            print(f'Ep{ep+1}/{n_ep} | loss={avg_loss:.4f} | hu={hr:.0f}% | fan={af:.0f} | {e:.0f}s', flush=True)

    e=time.time()-t0
    r={
        'name':name,'feature_names':feature_names,'obs_size':obs_size,
        'episodes':n_ep,'elapsed_sec':e,
        'hu_rates':hu_rates,'max_fans':max_fans,'lengths':lengths,
        'final_hu_rate_50':float(np.mean(hu_rates[-50:])*100),
        'final_avg_fan_50':float(np.mean(max_fans[-50:])),
    }
    print(f'Done. hu={r["final_hu_rate_50"]:.0f}% fan={r["final_avg_fan_50"]:.0f} | {e:.0f}s', flush=True)
    return r


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__=='__main__':
    experiments = [
        ('SL-RL baseline (6ch)', []),
        ('SL-RL +meld (10ch)', ['meld']),
        ('SL-RL +wall (7ch)', ['wall']),
        ('SL-RL +meld+wall (11ch)', ['meld','wall']),
        ('SL-RL +meld+discard (14ch)', ['meld','discard']),
        ('SL-RL +wall+discard (11ch)', ['wall','discard']),
        ('SL-RL +meld+wall+discard (15ch)', ['meld','wall','discard']),
    ]

    results = {}
    for name, feats in experiments:
        r = run_experiment(name, feats)
        results[r['name']] = r

    with open('all_features_result.json','w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"="*70}')
    print(f'{"Experiment":<35} {"hu_rate":>8} {"avg_fan":>8} {"ch":>4}')
    print(f'{"="*70}')
    for k, r in results.items():
        hu = r['final_hu_rate_50']; fan = r['final_avg_fan_50']; ch = r['obs_size']
        print(f'{r["name"]:<35} {hu:>7.0f}% {fan:>8.0f} {ch:>4}')
    print(f'{"="*70}')
