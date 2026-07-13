"""Cloud training v2: conservative PPO to prevent policy drift."""
import os, sys, time, json, numpy as np
from collections import defaultdict
import torch, torch.nn.functional as F
from torch.distributions import Categorical
import torch_npu

WORK = '/home/ma-user/work'
os.chdir(WORK); sys.path.insert(0, WORK)
from model_var import CNNModelVar
from env import MahjongGBEnv
from agent import MahjongGBAgent
from MahjongGB import MahjongFanCalculator

device = 'npu:0'; torch.npu.set_device(device)

TILE_LIST = [*(f'W{i}' for i in range(1,10)), *(f'T{i}' for i in range(1,10)),
             *(f'B{i}' for i in range(1,10)), *(f'F{i}' for i in range(1,5)),
             *(f'J{i}' for i in range(1,4))]
T2I = {c:i for i,c in enumerate(TILE_LIST)}

class MeldWallAgent(MahjongGBAgent):
    observation_space = None; action_space = None
    def __init__(self, seatWind):
        self.seatWind=seatWind; self.ch=11
        self.packs=[[]for _ in range(4)]; self.history=[[]for _ in range(4)]
        self.tileWall=[21]*4; self.shownTiles=defaultdict(int)
        self.wallLast=False; self.isAboutKong=False
        self.obs=np.zeros((11,36)); self.obs[0][T2I['F%d'%(seatWind+1)]]=1

    def request2obs(self,req):
        t=req.split()
        if t[0]=='Wind':self.prevalentWind=int(t[1]);self.obs[1][T2I['F%d'%(self.prevalentWind+1)]]=1;return
        if t[0]=='Deal':self.hand=t[1:];self._upd();return
        if t[0]=='Huang':self.valid=[];return self._obs()
        if t[0]=='Draw':return self._draw(t[1])
        p=(int(t[1])+4-self.seatWind)%4
        if t[2]=='Draw':self.tileWall[p]-=1;self.wallLast=self.tileWall[(p+1)%4]==0;return
        if t[2]in('Invalid','Hu'):self.valid=[];return self._obs()
        if t[2]=='Play':return self._play(p,t[3])
        if t[2]=='Chi':return self._chi(p,t[3])
        if t[2]in('UnChi','UnPeng'):return
        if t[2]=='Peng':return self._peng(p)
        if t[2]=='Gang':return self._gang(p)
        if t[2]=='AnGang':return self._angang(p,t[3]if len(t)>3 else None)
        if t[2]=='BuGang':return self._bugang(p,t[3])
        raise NotImplementedError(f'Unknown:{req}')

    def _draw(self,tile):
        self.tileWall[0]-=1;self.wallLast=self.tileWall[1]==0;self.isAboutKong=False
        self.hand.append(tile);self._upd();self.valid=[]
        if self._mj(tile,True,self.isAboutKong):self.valid.append(1)
        for t_ in set(self.hand):
            self.valid.append(2+T2I[t_])
            if self.hand.count(t_)==4 and not self.wallLast and self.tileWall[0]>0:self.valid.append(167+T2I[t_])
        if not self.wallLast and self.tileWall[0]>0:
            for pt,t_,_ in self.packs[0]:
                if pt=='PENG'and t_ in self.hand:self.valid.append(201+T2I[t_])
        self._feat();return self._obs()

    def _play(self,p,tile):
        self.tileFrom=p;self.curTile=tile;self.shownTiles[tile]+=1;self.history[p].append(tile)
        if p==0:self.hand.remove(tile);self._upd();return
        self.valid=[]
        if self._mj(tile):self.valid.append(1)
        if not self.wallLast:
            if self.hand.count(tile)>=2:
                self.valid.append(99+T2I[tile])
                if self.hand.count(tile)==3 and self.tileWall[0]:self.valid.append(133+T2I[tile])
            if p==3 and tile[0]in'WTB':
                num=int(tile[1])
                for i,off in[(-2,2),(-1,1),(0,0)]:
                    tmp=[tile[0]+str(num+j)for j in range(i,i+3)]
                    if all(x in self.hand for x in tmp[:2]+tmp[1:]):
                        c='WTB'.index(tile[0]);self.valid.append(36+c*21+(num-3+i)*3+(2-off))
        self.valid.append(0);self._feat();return self._obs()

    def _chi(self,p,tile):
        c,n=tile[0],int(tile[1]);self.packs[p].append(('CHI',tile,int(self.curTile[1])-n+2))
        self.shownTiles[self.curTile]-=1
        for i in range(-1,2):self.shownTiles[c+str(n+i)]+=1
        self.wallLast=self.tileWall[(p+1)%4]==0
        if p==0:
            self.valid=[];self.hand.append(self.curTile)
            for i in range(-1,2):self.hand.remove(c+str(n+i))
            self._upd()
            for t_ in set(self.hand):self.valid.append(2+T2I[t_])
            return self._obs()
        return
    def _peng(self,p):
        self.packs[p].append(('PENG',self.curTile,(4+p-self.tileFrom)%4))
        self.shownTiles[self.curTile]+=2;self.wallLast=self.tileWall[(p+1)%4]==0
        if p==0:self.valid=[];[self.hand.remove(self.curTile)for _ in range(2)];self._upd()
        for t_ in set(self.hand):self.valid.append(2+T2I[t_])
        return self._obs()if p==0 else None
    def _gang(self,p):
        self.packs[p].append(('GANG',self.curTile,(4+p-self.tileFrom)%4));self.shownTiles[self.curTile]+=3
        if p==0:[self.hand.remove(self.curTile)for _ in range(3)];self._upd();self.isAboutKong=True
    def _angang(self,p,tile):
        t='CONCEALED'if p else tile;self.packs[p].append(('GANG',t,0))
        if p==0:self.isAboutKong=True;[self.hand.remove(tile)for _ in range(4)]
        else:self.isAboutKong=False
    def _bugang(self,p,tile):
        for i in range(len(self.packs[p])):
            if tile==self.packs[p][i][1]:self.packs[p][i]=('GANG',tile,self.packs[p][i][2]);break
        self.shownTiles[tile]+=1
        if p==0:self.hand.remove(tile);self._upd();self.isAboutKong=True;return
        self.valid=[]
        if self._mj(tile,False,True):self.valid.append(1)
        self.valid.append(0);return self._obs()

    def action2response(self,a):
        if a<1:return'Pass'
        if a<2:return'Hu'
        if a<36:return'Play '+TILE_LIST[a-2]
        if a<99:t=(a-36)//3;return'Chi '+('WTB'[t//7])+str(t%7+2)
        if a<133:return'Peng'
        if a<167:return'Gang'
        if a<201:return'Gang '+TILE_LIST[a-167]
        return'BuGang '+TILE_LIST[a-201]
    def response2action(self,r):
        t=r.split()
        if t[0]=='Pass':return 0
        if t[0]=='Hu':return 1
        if t[0]=='Play':return 2+T2I[t[1]]
        if t[0]=='Chi':return 36+'WTB'.index(t[1][0])*21+(int(t[2][1])-2)*3+int(t[1][1])-int(t[2][1])+1
        if t[0]=='Peng':return 99+T2I[t[1]]
        if t[0]=='Gang':return 133+T2I[t[1]]
        if t[0]=='AnGang':return 167+T2I[t[1]]
        if t[0]=='BuGang':return 201+T2I[t[1]]
        return 0
    def _obs(self):
        m=np.zeros(235);[m.__setitem__(a,1)for a in self.valid]
        return{'observation':self.obs.reshape((11,4,9)).copy(),'action_mask':m}
    def _upd(self):
        self.obs[2:6]=0;d=defaultdict(int)
        for t in self.hand:d[t]+=1
        for t,c in d.items():self.obs[2:2+c,T2I[t]]=1
    def _feat(self):
        for pi in range(4):self.obs[6+pi]=0
        for pi in range(4):
            for pt,t,_ in self.packs[pi]:
                if t in T2I:self.obs[6+pi][T2I[t]]=1
        self.obs[10].fill(min(sum(self.tileWall)/144.0,1.0))
    def _mj(self,wt,sd=False,ak=False):
        try:
            fans=MahjongFanCalculator(pack=tuple(self.packs[0]),hand=tuple(self.hand),
                winTile=wt,flowerCount=0,isSelfDrawn=sd,
                is4thTile=(self.shownTiles.get(wt,0)+sd)==4,
                isAboutKong=ak,isWallLast=self.wallLast,
                seatWind=self.seatWind,prevalentWind=self.prevalentWind,verbose=True)
            if sum(fp*c for fp,c,_,_ in fans)<8:raise Exception
        except:return False
        return True

def main():
    N_EP=8000;CKPT=1000
    print(f'Cloud v2: meld+wall 11ch, clip=0.1, lr=5e-5, {N_EP} eps',flush=True)
    torch.manual_seed(42);np.random.seed(42)
    model=CNNModelVar(in_channels=11)
    sl=os.path.join(WORK,'model_20.pt')
    if os.path.exists(sl):
        sd=torch.load(sl,map_location='cpu',weights_only=True)
        model.load_sl_tower(sd,11);print('SL loaded',flush=True)
    model=model.to(device)
    opt=torch.optim.Adam(model.parameters(),lr=5e-5)
    env=MahjongGBEnv(config={'agent_clz':MeldWallAgent});names=env.agent_names
    gamma,lam,clip=0.98,0.95,0.1
    vc,ec,ppe,bs=1.0,0.01,4,256
    hu_rates,max_fans=[],[]
    t0=time.time()
    for ep in range(N_EP):
        obs_dict=env.reset()
        traj={a:{'obs':[],'mask':[],'act':[],'rew':[],'val':[]}for a in names}
        done,ep_len=False,0;term_r=None
        while not done:
            actions,values={},{}
            for a in obs_dict:
                traj[a]['obs'].append(obs_dict[a]['observation'])
                traj[a]['mask'].append(obs_dict[a]['action_mask'])
                ot=torch.tensor(obs_dict[a]['observation'],dtype=torch.float).unsqueeze(0).to(device)
                mt=torch.tensor(obs_dict[a]['action_mask'],dtype=torch.float).unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    logits,value=model({'observation':ot,'action_mask':mt})
                    act=Categorical(logits=logits).sample().item()
                actions[a]=act;values[a]=value.item()
                traj[a]['act'].append(act);traj[a]['val'].append(value.item())
            next_obs,rewards,done=env.step(actions)
            for a in rewards:traj[a]['rew'].append(rewards[a])
            if done:term_r=rewards
            obs_dict=next_obs;ep_len+=1
        if term_r:rv=list(term_r.values());hu=max(rv)>0;mr=max(rv)
        else:hu=False;mr=0
        hu_rates.append(1 if hu else 0);max_fans.append(mr)
        all_o,all_m,all_a,all_adv,all_tgt=[],[],[],[],[]
        for a in names:
            d=traj[a]
            if not d['act']:continue
            n=len(d['act']);rw=d['rew'][:n]if len(d['rew'])>=n else(d['rew']+[0])[:n]
            vl=d['val'][:n];nv=d['val'][1:]+[0]
            td=np.array(rw)+gamma*np.array(nv);tdd=td-np.array(vl)
            advs=[];adv=0.0
            for delta in reversed(tdd):adv=gamma*lam*adv+delta;advs.append(adv)
            advs=np.array(advs[::-1],dtype=np.float32)
            all_o.append(np.stack(d['obs']));all_m.append(np.stack(d['mask']))
            all_a.append(np.array(d['act'],dtype=np.int64));all_adv.append(advs)
            all_tgt.append(td.astype(np.float32))
        if not all_o:continue
        oa=np.concatenate(all_o);ma=np.concatenate(all_m);aa=np.concatenate(all_a)
        dva=np.concatenate(all_adv);ta=np.concatenate(all_tgt)
        dva=(dva-dva.mean())/(dva.std()+1e-8)
        total_loss=0;nu=0;idx=np.random.permutation(len(aa))
        for s in range(0,len(aa),bs):
            ix=idx[s:s+bs]
            ob=torch.tensor(oa[ix],dtype=torch.float).to(device)
            mb=torch.tensor(ma[ix],dtype=torch.float).to(device)
            ab=torch.tensor(aa[ix]).unsqueeze(-1).to(device)
            db=torch.tensor(dva[ix],dtype=torch.float).to(device)
            tb=torch.tensor(ta[ix],dtype=torch.float).to(device)
            model.train()
            with torch.no_grad():ol,_=model({'observation':ob,'action_mask':mb});olp=torch.log(F.softmax(ol,dim=1).gather(1,ab)+1e-8)
            for _ in range(ppe):
                l,vs=model({'observation':ob,'action_mask':mb})
                lp=torch.log(F.softmax(l,dim=1).gather(1,ab)+1e-8)
                ratio=torch.exp(lp-olp);s1=ratio*db;s2=torch.clamp(ratio,1-clip,1+clip)*db
                pl=-torch.mean(torch.min(s1,s2));vl=torch.mean(F.mse_loss(vs.squeeze(-1),tb))
                el=-torch.mean(Categorical(logits=l).entropy())
                loss=pl+vc*vl+ec*el
                opt.zero_grad();loss.backward();opt.step()
                total_loss+=loss.item();nu+=1
        avg_loss=total_loss/max(nu,1)
        if(ep+1)%200==0 or ep==0:
            e=time.time()-t0;hr=np.mean(hu_rates[-200:])*100;af=np.mean(max_fans[-200:])
            print(f'Ep{ep+1}/{N_EP}|loss={avg_loss:.4f}|hu={hr:.0f}%|fan={af:.0f}|{e:.0f}s',flush=True)
        if(ep+1)%CKPT==0:
            ckpt=os.path.join(WORK,f'v2_ckpt_ep{ep+1}.pt')
            torch.save({'model':model.state_dict(),'hu_rates':hu_rates,'max_fans':max_fans,'ep':ep+1},ckpt)
            print(f'CKPT:{ckpt}',flush=True)
    result={'hu_rates':hu_rates,'max_fans':max_fans,'elapsed':time.time()-t0,'final_hu_rate_200':float(np.mean(hu_rates[-200:])*100)}
    with open(os.path.join(WORK,'cloud_v2_result.json'),'w')as f:json.dump(result,f)
    torch.save(model.state_dict(),os.path.join(WORK,'v2_final_model.pt'))
    final_hu = result['final_hu_rate_200']
    print(f'Done. hu={final_hu:.0f}%',flush=True)

if __name__=='__main__':main()
