"""
v15 — Aggressive rollback + stabilization phase + EMA + permanent BC + tight KL.
"""
import os, sys, argparse, json, time, torch, random
import torch.nn.functional as F
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RL'))
from model import CNNModel
from feature import FeatureAgent
from env import MahjongGBEnv

CFG = {
    'lr': 5e-5, 'clip': 0.1, 'gamma': 0.98, 'gae_lambda': 0.95,
    'value_coeff': 0.5, 'ppo_epochs': 4, 'batch_size': 256,
    'eval_interval': 200, 'eval_games': 80,
    'print_interval': 50, 'ckpt_interval': 300,
    'kl_coeff': 0.05, 'bc_buf_size': 10000, 'bc_batch_ratio': 0.30, 'bc_weight': 1.0,
    'exploiter_train_every': 100, 'exploiter_buf_size': 5000,
    'exploiter_batch_ratio': 0.15, 'exploiter_priority': 2.0,
    'entropy_coeff': 0.03, 'entropy_low': 0.06, 'entropy_high': 0.10,
    'sl_opponent_ratio': 0.7,
    'sl_model_path': os.path.join(os.path.dirname(__file__), '..', 'SL', 'model',
                                   'checkpoint', 'model_20.pt'),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

class SimpleBuffer:
    def __init__(self, max_size=50000): self.buf = deque(maxlen=max_size)
    def push(self, s): self.buf.append(s)
    def size(self): return len(self.buf)
    def sample(self, bs):
        if len(self.buf) < bs: return None
        idx = np.random.choice(len(self.buf), bs, replace=False)
        batch = [self.buf[i] for i in idx]
        obs = torch.from_numpy(np.stack([b['obs'] for b in batch]).astype(np.float32))
        masks = torch.from_numpy(np.stack([b['mask'] for b in batch]).astype(np.float32))
        acts = torch.tensor([b['act'] for b in batch], dtype=torch.long)
        advs = torch.tensor([b.get('adv',0) for b in batch], dtype=torch.float32)
        tgts = torch.tensor([b.get('tgt',0) for b in batch], dtype=torch.float32)
        old_lp = torch.tensor([b.get('old_lp',0) for b in batch], dtype=torch.float32)
        return obs, masks, acts, advs, tgts, old_lp

def run_game(model, sl_model, cfg):
    device = cfg['device']
    opp = [sl_model]*3 if np.random.random()<cfg['sl_opponent_ratio'] else [model]*3
    models = [model]+opp
    env = MahjongGBEnv({'agent_clz':FeatureAgent,'duplicate':True,'variety':10000})
    obs_dict = env.reset()
    agents = [FeatureAgent(i) for i in range(4)]
    for i in range(4): agents[i].request2obs('Wind %d'%(i%4))
    traj=[]; done=False; turns=0
    while not done and turns<500:
        actions={}
        for name in env.agent_names:
            i=int(name.split('_')[1])-1; obs=obs_dict.get(name)
            if obs is not None:
                ot=torch.from_numpy(np.expand_dims(obs['observation'],0)).float().to(device)
                mt=torch.from_numpy(np.expand_dims(obs['action_mask'],0)).float().to(device)
                with torch.no_grad():
                    logits,_ = models[i]({'observation':ot,'action_mask':mt})
                probs=F.softmax(logits,dim=-1)
                action=int(logits[0].argmax().item()) if i!=0 else int(torch.multinomial(probs,1).item())
                if i==0:
                    lp=F.log_softmax(logits,dim=-1)[0,action].item()
                    _,val=model({'observation':ot,'action_mask':mt})
                    sl_out=sl_model({'observation':ot,'action_mask':mt})
                    sl_act=int(sl_out[0].argmax().item()) if isinstance(sl_out,tuple) else int(sl_out.argmax().item())
                    traj.append({'obs':obs['observation'].copy(),'mask':obs['action_mask'].copy(),
                                 'act':action,'sl_act':sl_act,'val':val.item(),'old_lp':lp,'r':0.0})
                actions[name]=action
        if not actions: break
        obs_dict,reward_dict,done_dict=env.step(actions)
        if traj: traj[-1]['r']=reward_dict.get('player_1',0)
        done=done_dict; turns+=1
    ws=None
    for n,r in reward_dict.items():
        if r>0: ws=int(n.split('_')[1])-1; break
    fan=0
    if ws is not None:
        r=reward_dict.get(f'player_{ws+1}',0)
        if r>0: fan=max(0,int(r/3-8))
    if traj:
        gae=0.0
        for t in reversed(range(len(traj))):
            s=traj[t]; nv=traj[t+1]['val'] if t+1<len(traj) else 0.0
            d=s['r']+cfg['gamma']*nv-s['val']
            gae=d+cfg['gamma']*cfg['gae_lambda']*gae
            s['adv']=gae; s['tgt']=gae+s['val']
    return {'samples':traj,'ws':ws,'fan':fan,'turns':turns,'hu':ws is not None}

def ppo_update(model,sl_model,opt,buf,bc_buf,exp_buf,cfg):
    bs=cfg['batch_size']; device=cfg['device']
    if buf.size()<bs: return {'pl':0,'vl':0,'ent':0,'kl':0,'bc':0}
    batch=buf.sample(bs)
    if batch is None: return {'pl':0,'vl':0,'ent':0,'kl':0,'bc':0}
    obs,masks,acts,advs,tgts,old_lp=batch
    obs,masks=obs.to(device),masks.to(device); acts=acts.to(device)
    advs=advs.to(device); tgts=tgts.to(device); old_lp=old_lp.to(device)
    advs=(advs-advs.mean())/(advs.std()+1e-8)
    bc_obs=bc_masks=bc_acts=None
    if bc_buf.size()>=32:
        n_bc=int(bs*cfg['bc_batch_ratio'])
        bc_batch=bc_buf.sample(min(n_bc,bc_buf.size()))
        if bc_batch: bc_obs,bc_masks,bc_acts,_,_,_=bc_batch
    exp_obs=exp_masks=exp_acts=None
    if exp_buf.size()>=16:
        n_exp=int(bs*cfg['exploiter_batch_ratio'])
        exp_batch=exp_buf.sample(min(n_exp,exp_buf.size()))
        if exp_batch: exp_obs,exp_masks,exp_acts,_,_,_=exp_batch
    tp,tv,te,tk,tbc=0,0,0,0,0
    for _ in range(cfg['ppo_epochs']):
        logits,values=model({'observation':obs,'action_mask':masks})
        probs=F.softmax(logits,dim=-1); logp=F.log_softmax(logits,dim=-1)
        sel_lp=logp.gather(1,acts.unsqueeze(1)).squeeze(1)
        ratio=torch.exp(sel_lp-old_lp)
        clip_adv=torch.clamp(ratio,1-cfg['clip'],1+cfg['clip'])*advs
        policy_loss=-torch.min(ratio*advs,clip_adv).mean()
        value_loss=F.mse_loss(values.squeeze(1),tgts)
        entropy=-(probs*logp).sum(dim=-1).mean()
        kl=torch.tensor(0.0,device=device)
        if sl_model is not None:
            with torch.no_grad():
                sl_out=sl_model({'observation':obs,'action_mask':masks})
                sl_l=sl_out[0] if isinstance(sl_out,tuple) else sl_out
            sl_lp=F.log_softmax(sl_l,dim=-1)
            valid=(masks>0.5).float()
            kl=(valid*probs*(logp-sl_lp.detach())).sum(dim=-1).mean()
        bc_loss=torch.tensor(0.0,device=device)
        if bc_obs is not None:
            bc_obs_d,bc_masks_d=bc_obs.to(device),bc_masks.to(device)
            bc_acts_d=bc_acts.to(device)
            bc_logits,_=model({'observation':bc_obs_d,'action_mask':bc_masks_d})
            bc_loss=F.cross_entropy(bc_logits,bc_acts_d)
        exp_loss=torch.tensor(0.0,device=device)
        if exp_obs is not None:
            exp_obs_d,exp_masks_d=exp_obs.to(device),exp_masks.to(device)
            exp_acts_d=exp_acts.to(device)
            exp_logits,_=model({'observation':exp_obs_d,'action_mask':exp_masks_d})
            exp_loss=F.cross_entropy(exp_logits,exp_acts_d)
        kl_item=kl.item(); ent_item=entropy.item()
        kl_scale=2.0 if kl_item>0.10 else (0.5 if kl_item<0.05 else 1.0)
        ent_scale=2.0 if ent_item<cfg['entropy_low'] else (0.5 if ent_item>cfg['entropy_high'] else 1.0)
        loss=(policy_loss+cfg['value_coeff']*value_loss
              +kl_scale*cfg['kl_coeff']*kl+cfg['bc_weight']*bc_loss
              +cfg['exploiter_priority']*exp_loss-cfg['entropy_coeff']*entropy*ent_scale)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tp+=policy_loss.item(); tv+=value_loss.item(); te+=ent_item; tk+=kl_item; tbc+=bc_loss.item()
    n=cfg['ppo_epochs']
    return {'pl':tp/n,'vl':tv/n,'ent':te/n,'kl':tk/n,'bc':tbc/n}

def train_exploiter(exp_model,model,sl_model,opt,cfg):
    device=cfg['device']; bs=128; d=SimpleBuffer(max_size=5000)
    for _ in range(5):
        r=run_game(model,sl_model,cfg)
        for s in r.get('samples',[]):
            d.push({'obs':s['obs'],'mask':s['mask'],'act':s['act'],'adv':0,'tgt':0,'old_lp':0})
    batch=d.sample(min(bs,d.size()))
    if batch is None: return
    obs,masks,_,_,_,_=batch; obs,masks=obs.to(device),masks.to(device)
    with torch.no_grad():
        cur_out=model({'observation':obs,'action_mask':masks})
        cur_logits=cur_out[0] if isinstance(cur_out,tuple) else cur_out
        cur_probs=F.softmax(cur_logits,dim=-1)
    for _ in range(5):
        exp_logits,_=exp_model({'observation':obs,'action_mask':masks})
        exp_probs=F.softmax(exp_logits,dim=-1); exp_lp=F.log_softmax(exp_logits,dim=-1)
        cur_lp=F.log_softmax(cur_logits,dim=-1).detach()
        kl_gap=(exp_probs*(exp_lp-cur_lp)).sum(dim=-1).mean()
        with torch.no_grad():
            sl_out=sl_model({'observation':obs,'action_mask':masks})
            sl_l=sl_out[0] if isinstance(sl_out,tuple) else sl_out
            sl_lp=F.log_softmax(sl_l,dim=-1)
        kl_to_sl=(exp_probs*(exp_lp-sl_lp.detach())).sum(dim=-1).mean()
        loss=-kl_gap+0.5*kl_to_sl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(exp_model.parameters(),1.0); opt.step()

def evaluate(model,sl_model,cfg,n=50):
    device=cfg['device']; w,tf,tt=0,0,0
    for g in range(n):
        r=run_game(model,sl_model,cfg)
        rl_won=(g%2==0 and r['ws']==0) or (g%2==1 and r['ws']!=0)
        if rl_won: w+=1
        if r['hu']: tf+=r['fan']
        tt+=r['turns']
    return {'win':w/n,'fan':tf/max(w,1),'turns':tt/n}

def train_one_seed(seed, total_episodes, output_dir):
    cfg = dict(CFG)
    cfg['total_episodes'] = total_episodes; cfg['ckpt_dir'] = output_dir; cfg['seed'] = seed
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = cfg['device']
    print(f'[v15] seed={seed} ep={total_episodes} device={device}')

    sl = CNNModel().to(device)
    sl_chk = torch.load(cfg['sl_model_path'], map_location=device, weights_only=False)
    sl.load_state_dict(sl_chk, strict=False); sl.eval()
    for p in sl.parameters(): p.requires_grad = False

    model = CNNModel().to(device); model.load_state_dict(sl_chk, strict=False)
    ema_model = CNNModel().to(device); ema_model.load_state_dict(model.state_dict())
    ema_decay = 0.995

    exp = CNNModel().to(device); exp.load_state_dict(sl_chk, strict=False)
    exp_opt = torch.optim.Adam(exp.parameters(), lr=cfg['lr']*0.3)

    buf = SimpleBuffer(max_size=50000); bc_buf = SimpleBuffer(max_size=cfg['bc_buf_size'])
    exp_buf = SimpleBuffer(max_size=cfg['exploiter_buf_size'])
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    base_lr = cfg['lr']

    best_sd = {k: v.cpu().clone() for k, v in ema_model.state_dict().items()}
    best_win, best_ep = 0.0, 0
    eval_wins, no_improve = [], 0
    tg, rb = 0, 0
    in_stable_phase = False

    for ep in range(1, total_episodes+1):
        # Stabilization phase at rolling >= 0.55
        rolling = np.mean(eval_wins[-3:]) if len(eval_wins) >= 3 else 0
        if rolling >= 0.55 and not in_stable_phase:
            in_stable_phase = True
            cfg['ppo_epochs'] = max(1, cfg['ppo_epochs']//2)
            cfg['entropy_coeff'] = cfg['entropy_coeff'] * 0.5
            cfg['clip'] = 0.1
            cfg['bc_weight'] = 1.0
            cfg['sl_opponent_ratio'] = 0.70
            for g in opt.param_groups: g['lr'] = base_lr * 0.5
            print(f'  [STABLE] rolling={rolling:.3f}>=0.55: half PPO/lr/entropy, BC=1.0, SL=70%')

        r = run_game(model, sl, cfg); tg += 1
        for s in r.get('samples', []):
            smp = {'obs':s['obs'],'mask':s['mask'],'act':s['act'],
                    'adv':s.get('adv',0),'tgt':s.get('tgt',0),'old_lp':s.get('old_lp',0)}
            buf.push(smp)
            bc_buf.push({'obs':s['obs'],'mask':s['mask'],'act':s['sl_act'],'adv':0,'tgt':0,'old_lp':0})

        if ep % cfg['exploiter_train_every'] == 0 and buf.size() >= 512:
            train_exploiter(exp, model, sl, exp_opt, cfg)
            for _ in range(3):
                er = run_game(exp, sl, cfg)
                for s in er.get('samples', []):
                    exp_buf.push({'obs':s['obs'],'mask':s['mask'],'act':s['act'],
                                   'adv':-abs(s.get('adv',0)),'tgt':s.get('tgt',0),'old_lp':s.get('old_lp',0)})

        # PPO with tight KL gate (>0.12 skip)
        st = ppo_update(model, sl, opt, buf, bc_buf, exp_buf, cfg)
        if st['kl'] > 0.12:
            # Skip this PPO step, KL too high
            if best_sd is not None and best_win > 0:
                model.load_state_dict(best_sd); ema_model.load_state_dict(best_sd)
                opt.state.clear(); rb += 1
                print(f'  [KL-SKIP+ROLLBACK] KL={st["kl"]:.3f}>0.12 rb=#{rb}')
            st = {'pl':0,'vl':0,'ent':0,'kl':st['kl'],'bc':0}

        # EMA
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data = ema_decay * ema_p.data + (1-ema_decay) * p.data

        # Eval (use EMA)
        if ep % cfg['eval_interval'] == 0:
            ev = evaluate(ema_model, sl, cfg, n=cfg['eval_games'])
            eval_wins.append(ev['win']); rolling = np.mean(eval_wins[-3:]) if len(eval_wins)>=3 else ev['win']

            # Peak checkpoint (save EMA)
            if ev['win'] > best_win:
                best_win = ev['win']; best_ep = ep
                best_sd = {k: v.cpu().clone() for k, v in ema_model.state_dict().items()}
                torch.save({'sd': best_sd, 'ep': ep, 'sl_win': best_win},
                           os.path.join(output_dir, 'best_model.pt'))
                no_improve = 0
            else:
                no_improve += 1

            # Aggressive rollback: best-current > 0.03 or 2 consecutive declines
            should_rollback = False
            if ev['win'] < best_win - 0.03:
                should_rollback = True
            elif len(eval_wins) >= 3:
                recent = eval_wins[-3:]
                if recent[0] > recent[1] > recent[2]:
                    should_rollback = True

            if should_rollback:
                model.load_state_dict(best_sd); ema_model.load_state_dict(best_sd)
                new_lr = max(1e-6, opt.param_groups[0]['lr'] * 0.5)
                for g in opt.param_groups: g['lr'] = new_lr
                opt.state.clear(); rb += 1
                print(f'  [ROLLBACK] best-win>{ev["win"]:.3f}+0.03, restore ep{best_ep} lr->{new_lr:.1e} rb=#{rb}')

            # Early stop: rolling >= 0.58 + 5 no-improve
            if rolling >= 0.58 and no_improve >= 5:
                print(f'[EARLY STOP] rolling={rolling:.3f}>=0.58 no_improve={no_improve}')
                break

            print(f'[Eval s={seed} ep={ep}] SL_win={ev["win"]:.3f} roll={rolling:.3f} '
                  f'best={best_win:.3f}@{best_ep} KL={st["kl"]:.4f} ni={no_improve} rb={rb} stable={in_stable_phase}')

    torch.save({'sd': best_sd, 'ep': best_ep, 'sl_win': best_win},
               os.path.join(output_dir, 'best_model.pt'))
    final_eval = evaluate(ema_model, sl, cfg, n=100)
    result = {'seed':seed,'peak_sl_win':best_win,'peak_ep':best_ep,'final_sl_win':final_eval['win'],
              'rollbacks':rb,'stable_phase':in_stable_phase,
              'mean_rolling':np.mean(eval_wins) if eval_wins else 0}
    with open(os.path.join(output_dir,'result.json'),'w') as f: json.dump(result,f,indent=2)
    print(f'[v15 DONE] seed={seed} peak={best_win:.3f}@{best_ep} final={final_eval["win"]:.3f} rolling={result["mean_rolling"]:.3f} rb={rb} stable={in_stable_phase}')
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--episodes', type=int, default=3000)
    p.add_argument('--output', type=str, default='./results')
    args = p.parse_args()
    train_one_seed(args.seed, args.episodes, args.output)
