"""
Feature ablation: SL→RL fine-tuning with 3 feature sets.

Design:
  - SL model trained once, frozen as init
  - RL fine-tunes all weights (policy + new value head)
  - 3 feature groups: baseline(6ch) / +ting(7ch) / full(21ch)
  - 500 episodes each, record hu_rate curve
  - Baselines: SL-only eval, RL-random-init
"""
import os, sys, time, json, numpy as np
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '../SL')

import torch, torch.nn.functional as F
from torch.distributions import Categorical

from env import MahjongGBEnv
from model_var import CNNModelVar
from feature_agent_ext import make_agent_cls, TILE_LIST, OFFSET_TILE, OFFSET_ACT
try:
    from MahjongGB import MahjongFanCalculator
except ImportError:
    raise SystemExit('MahjongGB required!')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EP = 500
SL_PATH = '../SL/model/checkpoint/model_20.pt'


# ═══════════════════════════════════════════════════════════
# AGENT FACTORIES
# ═══════════════════════════════════════════════════════════

def make_baseline_agent():
    """6-channel baseline agent (seat wind, prev wind, hand×4)."""
    return make_agent_cls(6)


def make_ting_agent():
    """7-channel agent: baseline + ting indicator (channel 6)."""
    from agent import MahjongGBAgent

    class TingAgent(MahjongGBAgent):
        def __init__(self, seatWind):
            self.seatWind = seatWind
            self.packs = [[] for _ in range(4)]
            self.history = [[] for _ in range(4)]
            self.tileWall = [21] * 4
            self.shownTiles = defaultdict(int)
            self.wallLast = False
            self.isAboutKong = False
            self.obs = np.zeros((7, 36))
            self.obs[0][OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1

        def request2obs(self, request):
            t = request.split()
            if t[0] == 'Wind':
                self.prevalentWind = int(t[1])
                self.obs[1][OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
                return
            if t[0] == 'Deal':
                self.hand = t[1:]; self._hand_upd(); return
            if t[0] == 'Huang': self.valid = []; return self._obs()
            if t[0] == 'Draw':
                self.tileWall[0] -= 1; self.wallLast = self.tileWall[1] == 0
                self.isAboutKong = False; self.hand.append(t[1]); self._hand_upd()
                self.valid = []
                if self._check_mj(t[1], True, self.isAboutKong): self.valid.append(1)
                for t_ in set(self.hand):
                    self.valid.append(2 + OFFSET_TILE[t_])
                    if self.hand.count(t_) == 4 and not self.wallLast and self.tileWall[0] > 0:
                        self.valid.append(167 + OFFSET_TILE[t_])
                if not self.wallLast and self.tileWall[0] > 0:
                    for pt, t_, _ in self.packs[0]:
                        if pt == 'PENG' and t_ in self.hand:
                            self.valid.append(201 + OFFSET_TILE[t_])
                self._fill_ting()
                return self._obs()
            p = (int(t[1]) + 4 - self.seatWind) % 4
            if t[2] == 'Draw': self.tileWall[p] -= 1; self.wallLast = self.tileWall[(p+1)%4] == 0; return
            if t[2] in ('Invalid','Hu'): self.valid = []; return self._obs()
            if t[2] == 'Play':
                self.tileFrom = p; self.curTile = t[3]
                self.shownTiles[self.curTile] += 1; self.history[p].append(self.curTile)
                if p == 0: self.hand.remove(self.curTile); self._hand_upd(); return
                else:
                    self.valid = []
                    if self._check_mj(self.curTile): self.valid.append(1)
                    if not self.wallLast:
                        if self.hand.count(self.curTile) >= 2:
                            self.valid.append(99 + OFFSET_TILE[self.curTile])
                            if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                                self.valid.append(133 + OFFSET_TILE[self.curTile])
                        if p == 3 and self.curTile[0] in 'WTB':
                            num = int(self.curTile[1])
                            for i, off in [(-2,2), (-1,1), (0,0)]:
                                tmp = [self.curTile[0]+str(num+j) for j in range(i,i+3)]
                                if all(x in self.hand for x in tmp[:2]+tmp[1:]):
                                    c = 'WTB'.index(self.curTile[0])
                                    self.valid.append(36 + c*21 + (num-3+i)*3 + (2-off))
                    self.valid.append(0)
                    self._fill_ting()
                    return self._obs()
            if t[2] == 'Chi':
                tile = t[3]; c, n = tile[0], int(tile[1])
                self.packs[p].append(('CHI', tile, int(self.curTile[1])-n+2))
                self.shownTiles[self.curTile] -= 1
                for i in range(-1,2): self.shownTiles[c+str(n+i)] += 1
                self.wallLast = self.tileWall[(p+1)%4] == 0
                if p == 0:
                    self.valid = []; self.hand.append(self.curTile)
                    for i in range(-1,2): self.hand.remove(c+str(n+i))
                    self._hand_upd()
                    for t_ in set(self.hand): self.valid.append(2+OFFSET_TILE[t_])
                    return self._obs()
                return
            if t[2] in ('UnChi','UnPeng'): return
            if t[2] == 'Peng':
                self.packs[p].append(('PENG',self.curTile,(4+p-self.tileFrom)%4))
                self.shownTiles[self.curTile] += 2
                self.wallLast = self.tileWall[(p+1)%4] == 0
                if p == 0:
                    self.valid = []; [self.hand.remove(self.curTile) for _ in range(2)]
                    self._hand_upd()
                    for t_ in set(self.hand): self.valid.append(2+OFFSET_TILE[t_])
                    return self._obs()
                return
            if t[2] == 'Gang':
                self.packs[p].append(('GANG',self.curTile,(4+p-self.tileFrom)%4))
                self.shownTiles[self.curTile] += 3
                if p == 0: [self.hand.remove(self.curTile) for _ in range(3)]; self._hand_upd(); self.isAboutKong = True
                return
            if t[2] == 'AnGang':
                tile = 'CONCEALED' if p else t[3]
                self.packs[p].append(('GANG',tile,0))
                if p == 0: self.isAboutKong = True; [self.hand.remove(tile) for _ in range(4)]
                else: self.isAboutKong = False
                return
            if t[2] == 'BuGang':
                tile = t[3]
                for i in range(len(self.packs[p])):
                    if tile == self.packs[p][i][1]: self.packs[p][i] = ('GANG',tile,self.packs[p][i][2]); break
                self.shownTiles[tile] += 1
                if p == 0: self.hand.remove(tile); self._hand_upd(); self.isAboutKong = True; return
                else:
                    self.valid = []; [self.valid.append(1) if self._check_mj(tile,False,True) else None, self.valid.append(0)]
                    return self._obs()
            raise NotImplementedError(f'Unknown: {request}')

        def action2response(self, a):
            if a < 1: return 'Pass'
            if a < 2: return 'Hu'
            if a < 36: return 'Play '+TILE_LIST[a-2]
            if a < 99: t=(a-36)//3; return 'Chi '+('WTB'[t//7])+str(t%7+2)
            if a < 133: return 'Peng'
            if a < 167: return 'Gang'
            if a < 201: return 'Gang '+TILE_LIST[a-167]
            return 'BuGang '+TILE_LIST[a-201]

        def response2action(self, r):
            t = r.split()
            if t[0]=='Pass': return 0
            if t[0]=='Hu': return 1
            if t[0]=='Play': return 2+OFFSET_TILE[t[1]]
            if t[0]=='Chi': return 36+'WTB'.index(t[1][0])*21+(int(t[2][1])-2)*3+int(t[1][1])-int(t[2][1])+1
            if t[0]=='Peng': return 99+OFFSET_TILE[t[1]]
            if t[0]=='Gang': return 133+OFFSET_TILE[t[1]]
            if t[0]=='AnGang': return 167+OFFSET_TILE[t[1]]
            if t[0]=='BuGang': return 201+OFFSET_TILE[t[1]]
            return 0

        def _obs(self):
            m = np.zeros(235); [m.__setitem__(a,1) for a in self.valid]
            return {'observation': self.obs.reshape((7,4,9)).copy(), 'action_mask': m}

        def _hand_upd(self):
            self.obs[2:6] = 0; d = defaultdict(int)
            for t in self.hand: d[t] += 1
            for t,c in d.items(): self.obs[2:2+c, OFFSET_TILE[t]] = 1

        def _fill_ting(self):
            self.obs[6] = 0
            if len(self.hand) not in (2,5,8,11,14): return
            for ut in set(self.hand):
                th = list(self.hand); th.remove(ut)
                for wt in set(th):
                    if wt[0] in 'FJ': continue
                    try:
                        fans = MahjongFanCalculator(
                            pack=tuple(self.packs[0]), hand=tuple(th), winTile=wt,
                            flowerCount=0, isSelfDrawn=True,
                            is4thTile=(self.shownTiles.get(wt,0))==3,
                            isAboutKong=False, isWallLast=self.wallLast,
                            seatWind=self.seatWind, prevalentWind=self.prevalentWind,
                            verbose=True)
                        if sum(fp*c for fp,c,_,_ in fans) >= 8:
                            self.obs[6][OFFSET_TILE[ut]] = 1; break
                    except: pass

        def _check_mj(self, wt, sd=False, ak=False):
            try:
                fans = MahjongFanCalculator(
                    pack=tuple(self.packs[0]), hand=tuple(self.hand), winTile=wt,
                    flowerCount=0, isSelfDrawn=sd,
                    is4thTile=(self.shownTiles.get(wt,0)+sd)==4,
                    isAboutKong=ak, isWallLast=self.wallLast,
                    seatWind=self.seatWind, prevalentWind=self.prevalentWind, verbose=True)
                if sum(fp*c for fp,c,_,_ in fans) < 8: raise Exception
            except: return False
            return True

    TingAgent.observation_space = None
    TingAgent.action_space = None
    return TingAgent


def make_full_agent():
    """21-channel agent: baseline + discards + melds + shown + opponent hands + wall + phase + ting."""
    from agent import MahjongGBAgent

    class FullAgent(MahjongGBAgent):
        def __init__(self, seatWind):
            self.seatWind = seatWind; self.obs_size = 21
            self.packs = [[] for _ in range(4)]
            self.history = [[] for _ in range(4)]
            self.tileWall = [21] * 4
            self.shownTiles = defaultdict(int)
            self.wallLast = False; self.isAboutKong = False
            self.obs = np.zeros((21, 36))
            self.obs[0][OFFSET_TILE['F%d'%(self.seatWind+1)]] = 1

        def request2obs(self, request):
            t = request.split()
            if t[0] == 'Wind':
                self.prevalentWind=int(t[1])
                self.obs[1][OFFSET_TILE['F%d'%(self.prevalentWind+1)]]=1; return
            if t[0] == 'Deal': self.hand=t[1:]; self._upd(); return
            if t[0] == 'Huang': self.valid=[]; return self._obs()
            if t[0] == 'Draw':
                self.tileWall[0]-=1; self.wallLast=self.tileWall[1]==0
                self.isAboutKong=False; self.hand.append(t[1]); self._upd()
                self.valid=[]
                if self._check_mj(t[1],True,self.isAboutKong): self.valid.append(1)
                for t_ in set(self.hand):
                    self.valid.append(2+OFFSET_TILE[t_])
                    if self.hand.count(t_)==4 and not self.wallLast and self.tileWall[0]>0:
                        self.valid.append(167+OFFSET_TILE[t_])
                if not self.wallLast and self.tileWall[0]>0:
                    for pt,t_,_ in self.packs[0]:
                        if pt=='PENG' and t_ in self.hand: self.valid.append(201+OFFSET_TILE[t_])
                self._fill_extra()
                return self._obs()
            p=(int(t[1])+4-self.seatWind)%4
            if t[2]=='Draw': self.tileWall[p]-=1; self.wallLast=self.tileWall[(p+1)%4]==0; return
            if t[2] in ('Invalid','Hu'): self.valid=[]; return self._obs()
            if t[2]=='Play':
                self.tileFrom=p; self.curTile=t[3]
                self.shownTiles[self.curTile]+=1; self.history[p].append(self.curTile)
                if p==0: self.hand.remove(self.curTile); self._upd(); return
                else:
                    self.valid=[]
                    if self._check_mj(self.curTile): self.valid.append(1)
                    if not self.wallLast:
                        if self.hand.count(self.curTile)>=2:
                            self.valid.append(99+OFFSET_TILE[self.curTile])
                            if self.hand.count(self.curTile)==3 and self.tileWall[0]:
                                self.valid.append(133+OFFSET_TILE[self.curTile])
                        if p==3 and self.curTile[0] in 'WTB':
                            num=int(self.curTile[1])
                            for i,off in [(-2,2),(-1,1),(0,0)]:
                                tmp=[self.curTile[0]+str(num+j) for j in range(i,i+3)]
                                if all(x in self.hand for x in tmp[:2]+tmp[1:]):
                                    c='WTB'.index(self.curTile[0])
                                    self.valid.append(36+c*21+(num-3+i)*3+(2-off))
                    self.valid.append(0); self._fill_extra()
                    return self._obs()
            if t[2]=='Chi':
                tile=t[3]; c,n=tile[0],int(tile[1])
                self.packs[p].append(('CHI',tile,int(self.curTile[1])-n+2))
                self.shownTiles[self.curTile]-=1
                for i in range(-1,2): self.shownTiles[c+str(n+i)]+=1
                self.wallLast=self.tileWall[(p+1)%4]==0
                if p==0:
                    self.valid=[]; self.hand.append(self.curTile)
                    for i in range(-1,2): self.hand.remove(c+str(n+i))
                    self._upd()
                    for t_ in set(self.hand): self.valid.append(2+OFFSET_TILE[t_])
                    return self._obs()
                return
            if t[2] in ('UnChi','UnPeng'): return
            if t[2]=='Peng':
                self.packs[p].append(('PENG',self.curTile,(4+p-self.tileFrom)%4))
                self.shownTiles[self.curTile]+=2
                self.wallLast=self.tileWall[(p+1)%4]==0
                if p==0:
                    self.valid=[]; [self.hand.remove(self.curTile) for _ in range(2)]
                    self._upd()
                    for t_ in set(self.hand): self.valid.append(2+OFFSET_TILE[t_])
                    return self._obs()
                return
            if t[2]=='Gang':
                self.packs[p].append(('GANG',self.curTile,(4+p-self.tileFrom)%4))
                self.shownTiles[self.curTile]+=3
                if p==0: [self.hand.remove(self.curTile) for _ in range(3)]; self._upd(); self.isAboutKong=True
                return
            if t[2]=='AnGang':
                tile='CONCEALED' if p else t[3]
                self.packs[p].append(('GANG',tile,0))
                if p==0: self.isAboutKong=True; [self.hand.remove(tile) for _ in range(4)]
                else: self.isAboutKong=False
                return
            if t[2]=='BuGang':
                tile=t[3]
                for i in range(len(self.packs[p])):
                    if tile==self.packs[p][i][1]: self.packs[p][i]=('GANG',tile,self.packs[p][i][2]); break
                self.shownTiles[tile]+=1
                if p==0: self.hand.remove(tile); self._upd(); self.isAboutKong=True; return
                else:
                    self.valid=[]; [self.valid.append(1) if self._check_mj(tile,False,True) else None, self.valid.append(0)]
                    return self._obs()
            raise NotImplementedError(f'Unknown: {request}')

        def action2response(self,a):
            if a<1: return 'Pass'
            if a<2: return 'Hu'
            if a<36: return 'Play '+TILE_LIST[a-2]
            if a<99: t=(a-36)//3; return 'Chi '+('WTB'[t//7])+str(t%7+2)
            if a<133: return 'Peng'
            if a<167: return 'Gang'
            if a<201: return 'Gang '+TILE_LIST[a-167]
            return 'BuGang '+TILE_LIST[a-201]
        def response2action(self,r):
            t=r.split()
            if t[0]=='Pass': return 0
            if t[0]=='Hu': return 1
            if t[0]=='Play': return 2+OFFSET_TILE[t[1]]
            if t[0]=='Chi': return 36+'WTB'.index(t[1][0])*21+(int(t[2][1])-2)*3+int(t[1][1])-int(t[2][1])+1
            if t[0]=='Peng': return 99+OFFSET_TILE[t[1]]
            if t[0]=='Gang': return 133+OFFSET_TILE[t[1]]
            if t[0]=='AnGang': return 167+OFFSET_TILE[t[1]]
            if t[0]=='BuGang': return 201+OFFSET_TILE[t[1]]
            return 0
        def _obs(self):
            m=np.zeros(235); [m.__setitem__(a,1) for a in self.valid]
            return {'observation':self.obs.reshape((21,4,9)).copy(),'action_mask':m}
        def _upd(self):
            self.obs[2:6]=0; d=defaultdict(int)
            for t in self.hand: d[t]+=1
            for t,c in d.items(): self.obs[2:2+c,OFFSET_TILE[t]]=1
        def _fill_extra(self):
            Ch=self.obs_size
            if Ch>=10:
                for pi in range(4):
                    self.obs[6+pi]=0
                    for t in self.history[pi]: self.obs[6+pi][OFFSET_TILE[t]]=min(1.0,self.obs[6+pi][OFFSET_TILE[t]]+0.1)
            if Ch>=14:
                for pi in range(4):
                    self.obs[10+pi]=0
                    for pt,t,_ in self.packs[pi]:
                        if t in OFFSET_TILE: self.obs[10+pi][OFFSET_TILE[t]]=1
            if Ch>=15:
                self.obs[14]=0
                for t,c in self.shownTiles.items():
                    if t in OFFSET_TILE: self.obs[14][OFFSET_TILE[t]]=min(c/4.0,1.0)
            if Ch>=19: self.obs[18].fill(min(sum(self.tileWall)/144.0,1.0))
            if Ch>=20:
                self.obs[19]=0
                if self.valid and 1 in self.valid: self.obs[19].fill(1.0)
            if Ch>=21:
                self.obs[20]=0
                if self.valid:
                    self.obs[20].fill(1.0)
        def _check_mj(self,wt,sd=False,ak=False):
            try:
                fans=MahjongFanCalculator(pack=tuple(self.packs[0]),hand=tuple(self.hand),winTile=wt,
                    flowerCount=0,isSelfDrawn=sd,is4thTile=(self.shownTiles.get(wt,0)+sd)==4,
                    isAboutKong=ak,isWallLast=self.wallLast,seatWind=self.seatWind,
                    prevalentWind=self.prevalentWind,verbose=True)
                if sum(fp*c for fp,c,_,_ in fans)<8: raise Exception
            except: return False
            return True
    FullAgent.observation_space = None
    FullAgent.action_space = None
    return FullAgent


# ═══════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════

def create_model(obs_size, sl_path=None):
    """Create CNNModelVar, optionally loading SL tower weights."""
    model = CNNModelVar(in_channels=obs_size)
    if sl_path and os.path.exists(sl_path):
        sd = torch.load(sl_path, map_location='cpu', weights_only=True)
        model.load_sl_tower(sd, obs_size)
    return model.to(device)


# ═══════════════════════════════════════════════════════════
# TRAINING + EVAL
# ═══════════════════════════════════════════════════════════

def run_experiment(name, agent_factory, obs_size, sl_init=True, n_ep=N_EP):
    print(f'\n{"="*50}', flush=True)
    print(f'{name} | obs={obs_size} | SL_init={sl_init} | {n_ep} eps', flush=True)

    torch.manual_seed(42); np.random.seed(42)

    model = create_model(obs_size, SL_PATH if sl_init else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # agent_factory can be a class or a callable returning a class
    if isinstance(agent_factory, type):
        AgentCls = agent_factory
    else:
        AgentCls = agent_factory()
    env = MahjongGBEnv(config={'agent_clz': AgentCls})
    names = env.agent_names

    gamma, lam, clip = 0.98, 0.95, 0.2
    vc, ec, ppe, bs = 1.0, 0.01, 4, 256

    hu_rates, max_fans, lengths = [], [], []
    t0 = time.time()

    for ep in range(n_ep):
        obs_dict = env.reset()
        traj = {a: {'obs':[],'mask':[],'act':[],'rew':[],'val':[]} for a in names}
        done, ep_len = False, 0
        term_r = None

        while not done:
            actions, values = {}, {}
            for a in obs_dict:
                traj[a]['obs'].append(obs_dict[a]['observation'])
                traj[a]['mask'].append(obs_dict[a]['action_mask'])
                ot = torch.tensor(obs_dict[a]['observation'], dtype=torch.float).unsqueeze(0).to(device)
                mt = torch.tensor(obs_dict[a]['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    logits, value = model({'observation': ot, 'action_mask': mt})
                    dist = Categorical(logits=logits)
                    act = dist.sample().item()
                actions[a] = act; values[a] = value.item()
                traj[a]['act'].append(act); traj[a]['val'].append(value.item())

            next_obs, rewards, done = env.step(actions)
            for a in rewards: traj[a]['rew'].append(rewards[a])
            if done: term_r = rewards
            obs_dict = next_obs; ep_len += 1

        lengths.append(ep_len)
        if term_r:
            rv = list(term_r.values()); hu = max(rv) > 0; mr = max(rv)
        else:
            hu = False; mr = 0
        hu_rates.append(1 if hu else 0); max_fans.append(mr)

        # PPO update
        all_o, all_m, all_a, all_adv, all_tgt = [], [], [], [], []
        for a in names:
            d = traj[a]
            if not d['act']: continue
            n = len(d['act'])
            rw = d['rew'][:n] if len(d['rew'])>=n else (d['rew']+[0])[:n]
            vl = d['val'][:n]
            nv = d['val'][1:]+[0]
            td = np.array(rw) + gamma * np.array(nv)
            tdd = td - np.array(vl)
            advs = []; adv = 0.0
            for delta in reversed(tdd): adv = gamma*lam*adv+delta; advs.append(adv)
            advs = np.array(advs[::-1], dtype=np.float32)
            all_o.append(np.stack(d['obs'])); all_m.append(np.stack(d['mask']))
            all_a.append(np.array(d['act'], dtype=np.int64))
            all_adv.append(advs); all_tgt.append(td.astype(np.float32))

        if not all_o: continue
        oa = np.concatenate(all_o); ma = np.concatenate(all_m)
        aa = np.concatenate(all_a); dva = np.concatenate(all_adv)
        ta = np.concatenate(all_tgt)
        dva = (dva - dva.mean()) / (dva.std() + 1e-8)

        total_loss = 0; nu = 0
        idx = np.random.permutation(len(aa))
        for s in range(0, len(aa), bs):
            ix = idx[s:s+bs]
            ob = torch.tensor(oa[ix], dtype=torch.float).to(device)
            mb = torch.tensor(ma[ix], dtype=torch.float).to(device)
            ab = torch.tensor(aa[ix]).unsqueeze(-1).to(device)
            db = torch.tensor(dva[ix], dtype=torch.float).to(device)
            tb = torch.tensor(ta[ix], dtype=torch.float).to(device)
            model.train()
            with torch.no_grad():
                ol,_ = model({'observation':ob,'action_mask':mb})
                olp = torch.log(F.softmax(ol,dim=1).gather(1,ab)+1e-8)
            for _ in range(ppe):
                l,vs = model({'observation':ob,'action_mask':mb})
                lp = torch.log(F.softmax(l,dim=1).gather(1,ab)+1e-8)
                ratio = torch.exp(lp-olp)
                s1 = ratio*db; s2 = torch.clamp(ratio,1-clip,1+clip)*db
                pl = -torch.mean(torch.min(s1,s2))
                vl = torch.mean(F.mse_loss(vs.squeeze(-1),tb))
                el = -torch.mean(Categorical(logits=l).entropy())
                loss = pl + vc*vl + ec*el
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item(); nu += 1
        avg_loss = total_loss/max(nu,1)

        if (ep+1)%50 == 0 or ep == 0:
            e = time.time()-t0; hr=np.mean(hu_rates[-50:])*100; af=np.mean(max_fans[-50:])
            print(f'Ep{ep+1}/{n_ep} | loss={avg_loss:.4f} | hu={hr:.0f}% | fan={af:.0f} | len={ep_len} | {e:.0f}s', flush=True)

    e = time.time()-t0
    r = {
        'name': name, 'obs_size': obs_size, 'sl_init': sl_init,
        'episodes': n_ep, 'elapsed_sec': e,
        'hu_rates': hu_rates, 'max_fans': max_fans, 'lengths': lengths,
        'final_hu_rate_50': float(np.mean(hu_rates[-50:])*100),
        'final_avg_fan_50': float(np.mean(max_fans[-50:])),
    }
    print(f'Done. hu={r["final_hu_rate_50"]:.0f}% fan={r["final_avg_fan_50"]:.0f} | {e:.0f}s', flush=True)
    return r


def evaluate_sl(n_ep=50):
    """Evaluate frozen SL model (no RL fine-tuning)."""
    print(f'\n{"="*50}', flush=True)
    print(f'SL-only eval | {n_ep} eps', flush=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location("sl_model", '../SL/model.py')
    sl_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl_mod)
    model = sl_mod.CNNModel().to(device)
    model.load_state_dict(torch.load(SL_PATH, map_location=device, weights_only=True))
    model.eval()

    AgentCls = make_agent_cls(6)
    env = MahjongGBEnv(config={'agent_clz': AgentCls})
    names = env.agent_names

    hu_rates, max_fans, lengths = [], [], []
    for ep in range(n_ep):
        obs_dict = env.reset()
        done, ep_len = False, 0; term_r = None
        while not done:
            actions = {}
            for a in obs_dict:
                ot = torch.tensor(obs_dict[a]['observation'], dtype=torch.float).unsqueeze(0).to(device)
                mt = torch.tensor(obs_dict[a]['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model({'is_training':False, 'obs':{'observation':ot,'action_mask':mt}})
                    act = Categorical(logits=logits).sample().item()
                actions[a] = act
            next_obs, rewards, done = env.step(actions)
            if done: term_r = rewards
            obs_dict = next_obs; ep_len += 1
        lengths.append(ep_len)
        if term_r:
            rv=list(term_r.values()); hu=max(rv)>0; mr=max(rv)
        else: hu=False; mr=0
        hu_rates.append(1 if hu else 0); max_fans.append(mr)
    r = {
        'name': 'SL-only', 'episodes': n_ep,
        'hu_rates': hu_rates, 'max_fans': max_fans, 'lengths': lengths,
        'avg_hu_rate': float(np.mean(hu_rates)*100),
        'avg_max_fan': float(np.mean(max_fans)),
    }
    print(f'Done. avg_hu={r["avg_hu_rate"]:.0f}% avg_fan={r["avg_max_fan"]:.0f}', flush=True)
    return r


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    results = {}

    # Baseline 1: SL-only eval
    results['sl_only'] = evaluate_sl(50)

    # Baseline 2: RL-random-init (already done, load from file)
    bp = 'rl_baseline_result.json'
    if os.path.exists(bp):
        with open(bp) as f: results['rl_random'] = json.load(f)

    # Experiment 1: SL→RL baseline (6ch)
    results['sl_rl_base'] = run_experiment(
        'SL→RL baseline (6ch)', make_agent_cls(6), obs_size=6, sl_init=True)

    # Experiment 2: SL→RL +ting (7ch)
    results['sl_rl_ting'] = run_experiment(
        'SL→RL +ting (7ch)', make_ting_agent, obs_size=7, sl_init=True)

    # Experiment 3: SL→RL full (21ch)
    results['sl_rl_full'] = run_experiment(
        'SL→RL full (21ch)', make_full_agent, obs_size=21, sl_init=True)

    # Save
    with open('ablation_final.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f'\n{"="*60}')
    print('FEATURE ABLATION SUMMARY')
    print(f'{"="*60}')
    for k, r in results.items():
        name = r['name']
        hu = r.get('final_hu_rate_50', r.get('avg_hu_rate', 0))
        fan = r.get('final_avg_fan_50', r.get('avg_max_fan', 0))
        print(f'  {name}: hu_rate={hu:.0f}%  avg_fan={fan:.0f}')
