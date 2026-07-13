"""
v16 — Champion Refinement Phase.
Fork from s0 peak (0.650) → N trust-region branches → league Elo → best model.
"""
import os, sys, argparse, json, torch, random, copy
import torch.nn.functional as F
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RL'))
from model import CNNModel
from feature import FeatureAgent
from env import MahjongGBEnv

# Base config (will be varied per branch)
BASE_CFG = {
    'lr': 2e-5, 'clip': 0.08, 'gamma': 0.98, 'gae_lambda': 0.95,
    'value_coeff': 0.5, 'ppo_epochs': 2, 'batch_size': 256,
    'eval_interval': 100, 'eval_games': 80,
    'print_interval': 25, 'ckpt_interval': 100,
    'kl_coeff': 0.05, 'bc_buf_size': 10000, 'bc_batch_ratio': 0.30, 'bc_weight': 1.0,
    'exploiter_train_every': 100, 'exploiter_buf_size': 5000,
    'exploiter_batch_ratio': 0.15, 'exploiter_priority': 2.0,
    'entropy_coeff': 0.03, 'entropy_low': 0.06, 'entropy_high': 0.10,
    'sl_opponent_ratio': 0.7,
    'kl_hard_limit': 0.06,          # Trust region: KL must stay below this
    'total_refine_ep': 1000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Branch hyperparameter variations
BRANCH_CONFIGS = [
    {'name': 'baseline',    'lr': 2e-5, 'kl_hard_limit': 0.06, 'bc_weight': 1.0},
    {'name': 'low_kl',      'lr': 1e-5, 'kl_hard_limit': 0.04, 'bc_weight': 1.2},
    {'name': 'high_bc',     'lr': 2e-5, 'kl_hard_limit': 0.06, 'bc_weight': 1.5},
    {'name': 'more_rl',     'lr': 3e-5, 'kl_hard_limit': 0.08, 'bc_weight': 0.8},
    {'name': 'tight_kl',    'lr': 1e-5, 'kl_hard_limit': 0.03, 'bc_weight': 1.0},
    {'name': 'entropy_up',  'lr': 2e-5, 'kl_hard_limit': 0.06, 'bc_weight': 1.0, 'entropy_coeff': 0.05},
    {'name': 'low_clip',    'lr': 2e-5, 'kl_hard_limit': 0.06, 'bc_weight': 1.0, 'clip': 0.05},
    {'name': 'high_epochs', 'lr': 2e-5, 'kl_hard_limit': 0.06, 'bc_weight': 1.0, 'ppo_epochs': 3},
]

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

def ppo_update(model,sl_model,opt,buf,bc_buf,cfg):
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
        kl_item=kl.item(); ent_item=entropy.item()
        kl_scale=2.0 if kl_item>0.08 else (0.5 if kl_item<0.03 else 1.0)
        ent_scale=2.0 if ent_item<cfg['entropy_low'] else (0.5 if ent_item>cfg['entropy_high'] else 1.0)
        loss=(policy_loss+cfg['value_coeff']*value_loss
              +kl_scale*cfg['kl_coeff']*kl+cfg['bc_weight']*bc_loss
              -cfg['entropy_coeff']*entropy*ent_scale)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tp+=policy_loss.item(); tv+=value_loss.item(); te+=ent_item; tk+=kl_item; tbc+=bc_loss.item()
    n=cfg['ppo_epochs']
    return {'pl':tp/n,'vl':tv/n,'ent':te/n,'kl':tk/n,'bc':tbc/n}

def evaluate(model,sl_model,cfg,n=50):
    device=cfg['device']; w,tf,tt=0,0,0
    for g in range(n):
        r=run_game(model,sl_model,cfg)
        rl_won=(g%2==0 and r['ws']==0) or (g%2==1 and r['ws']!=0)
        if rl_won: w+=1
        if r['hu']: tf+=r['fan']; tt+=r['turns']
    return {'win':w/n,'fan':tf/max(w,1),'turns':tt/n}

def train_exploiter_target(exp_model, target_model, sl_model, opt, cfg):
    """Train exploiter specifically against target (s0)."""
    device=cfg['device']; bs=128; d=SimpleBuffer(max_size=5000)
    for _ in range(5):
        r=run_game(target_model,sl_model,cfg)
        for s in r.get('samples',[]):
            d.push({'obs':s['obs'],'mask':s['mask'],'act':s['act'],'adv':0,'tgt':0,'old_lp':0})
    batch=d.sample(min(bs,d.size()))
    if batch is None: return
    obs,masks,_,_,_,_=batch; obs,masks=obs.to(device),masks.to(device)
    with torch.no_grad():
        cur_out=target_model({'observation':obs,'action_mask':masks})
        cur_logits=cur_out[0] if isinstance(cur_out,tuple) else cur_out
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

def elo_update(elo_dict, winner, loser, k=32):
    rw, rl = elo_dict.get(winner,1500), elo_dict.get(loser,1500)
    ew = 1/(1+10**((rl-rw)/400))
    elo_dict[winner] = rw + k*(1-ew)
    elo_dict[loser] = rl + k*(0-ew)
    return elo_dict

def run_branch(branch_cfg, s0_sd, sl_model, seed, output_dir):
    """Run one trust-region fine-tuning branch from s0 checkpoint."""
    cfg = dict(BASE_CFG)
    cfg.update(branch_cfg)
    cfg['seed'] = seed
    name = branch_cfg['name']
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = cfg['device']

    model = CNNModel().to(device)
    model.load_state_dict(s0_sd, strict=False)
    ema_model = CNNModel().to(device); ema_model.load_state_dict(model.state_dict())
    ema_decay = 0.995

    buf = SimpleBuffer(max_size=30000); bc_buf = SimpleBuffer(max_size=cfg['bc_buf_size'])
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    best_sd = {k:v.cpu().clone() for k,v in ema_model.state_dict().items()}
    best_win, best_ep = 0.0, 0
    total_ep = cfg['total_refine_ep']
    eval_wins = []

    for ep in range(1, total_ep+1):
        r = run_game(model, sl_model, cfg)
        for s in r.get('samples',[]):
            smp = {'obs':s['obs'],'mask':s['mask'],'act':s['act'],
                    'adv':s.get('adv',0),'tgt':s.get('tgt',0),'old_lp':s.get('old_lp',0)}
            buf.push(smp)
            bc_buf.push({'obs':s['obs'],'mask':s['mask'],'act':s['sl_act'],'adv':0,'tgt':0,'old_lp':0})

        st = ppo_update(model, sl_model, opt, buf, bc_buf, cfg)
        # Trust region: if KL exceeds hard limit, rollback to best
        if st['kl'] > cfg['kl_hard_limit'] and best_win > 0:
            model.load_state_dict(best_sd); ema_model.load_state_dict(best_sd)
            opt.state.clear()

        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data = ema_decay*ema_p.data + (1-ema_decay)*p.data

        if ep % cfg['eval_interval'] == 0:
            ev = evaluate(ema_model, sl_model, cfg, n=cfg['eval_games'])
            eval_wins.append(ev['win'])
            if ev['win'] > best_win:
                best_win = ev['win']; best_ep = ep
                best_sd = {k:v.cpu().clone() for k,v in ema_model.state_dict().items()}
                torch.save({'sd':best_sd,'ep':ep,'sl_win':best_win,'branch':name},
                           os.path.join(output_dir, 'best_model.pt'))
            if ev['win'] >= 0.62:
                torch.save({'sd':{k:v.cpu().clone() for k,v in ema_model.state_dict().items()},
                           'ep':ep,'sl_win':ev['win'],'branch':name},
                           os.path.join(output_dir, f'ckpt_ep{ep}_win{ev["win"]:.3f}.pt'))

    final_eval = evaluate(ema_model, sl_model, cfg, n=100)
    result = {'branch':name,'peak':best_win,'peak_ep':best_ep,'final':final_eval['win'],
              'eval_wins':eval_wins}
    with open(os.path.join(output_dir,'result.json'),'w') as f: json.dump(result,f)
    return result, best_sd, best_win

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--s0_ckpt', type=str, default='best_model_s0.pt')
    p.add_argument('--sl_ckpt', type=str, default=None)
    p.add_argument('--output', type=str, default='./refine_results')
    p.add_argument('--device', type=str, default='npu')
    p.add_argument('--num_branches', type=int, default=8)
    args = p.parse_args()

    # Set SL path
    if args.sl_ckpt is None:
        args.sl_ckpt = os.path.join(os.path.dirname(__file__), '..', 'SL', 'model',
                                     'checkpoint', 'model_20.pt')

    device = args.device
    print(f'[v16] Champion Refinement. Device: {device}')
    os.makedirs(args.output, exist_ok=True)

    # Load s0 peak checkpoint
    s0_ckpt = torch.load(args.s0_ckpt, map_location='cpu', weights_only=False)
    s0_sd = s0_ckpt['sd'] if 'sd' in s0_ckpt else s0_ckpt
    print(f'[v16] Loaded s0 peak: SL_win={s0_ckpt.get("sl_win","?")} ep={s0_ckpt.get("ep","?")}')

    # Load SL model
    sl = CNNModel().to(device)
    sl_chk = torch.load(args.sl_ckpt, map_location=device, weights_only=False)
    sl.load_state_dict(sl_chk, strict=False); sl.eval()
    for p in sl.parameters(): p.requires_grad = False

    # Train exploiter against s0
    print('[v16] Training exploiter against s0...')
    s0_model = CNNModel().to(device); s0_model.load_state_dict(s0_sd, strict=False); s0_model.eval()
    exp_model = CNNModel().to(device); exp_model.load_state_dict(sl_chk, strict=False)
    exp_opt = torch.optim.Adam(exp_model.parameters(), lr=2e-5)
    for _ in range(20):
        train_exploiter_target(exp_model, s0_model, sl, exp_opt, BASE_CFG)
    exp_sd = {k:v.cpu().clone() for k,v in exp_model.state_dict().items()}
    torch.save({'sd':exp_sd}, os.path.join(args.output, 'exploiter_vs_s0.pt'))
    print('[v16] Exploiter trained and saved.')

    # Run N branches
    branches = BRANCH_CONFIGS[:args.num_branches]
    results = {}; elo = {}
    for i, bcfg in enumerate(branches):
        name = bcfg['name']
        seed = 100 + i
        print(f'\n[v16] Branch {i+1}/{len(branches)}: {name}')
        result, best_sd, best_win = run_branch(bcfg, s0_sd, sl, seed,
                                                os.path.join(args.output, f'branch_{name}'))
        results[name] = result
        elo[name] = 1500.0
        print(f'  {name}: peak={result["peak"]:.3f}@{result["peak_ep"]} final={result["final"]:.3f}')

    # Cross-play: evaluate each branch vs others to compute Elo
    print('\n[v16] Cross-play Elo evaluation...')
    branch_models = {}
    for name, result in results.items():
        ckpt_path = os.path.join(args.output, f'branch_{name}', 'best_model.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            m = CNNModel().to(device); m.load_state_dict(ckpt['sd'], strict=False); m.eval()
            branch_models[name] = m

    for _ in range(5):  # 5 rounds of cross-play
        names = list(branch_models.keys())
        np.random.shuffle(names)
        for i in range(0, len(names)-1, 2):
            n1, n2 = names[i], names[i+1]
            # Play n1 vs n2 (n1 as seat0)
            cfg_tmp = dict(BASE_CFG); cfg_tmp['sl_opponent_ratio'] = 0.0
            # Quick eval: 10 games
            w1 = 0
            for g in range(10):
                r = run_game(branch_models[n1], branch_models[n2], cfg_tmp)
                if r['ws'] == 0: w1 += 1
            for _ in range(w1): elo = elo_update(elo, n1, n2, k=16)
            for _ in range(10-w1): elo = elo_update(elo, n2, n1, k=16)

    # Sort by Elo
    ranked = sorted(elo.items(), key=lambda x: x[1], reverse=True)
    print('\n=== Elo Rankings ===')
    for name, e in ranked:
        r = results.get(name, {})
        print(f'  {name}: Elo={e:.0f} peak={r.get("peak","?"):.3f} final={r.get("final","?"):.3f}')

    # Save #1 model
    best_name = ranked[0][0]
    best_branch_dir = os.path.join(args.output, f'branch_{best_name}')
    import shutil
    shutil.copy(os.path.join(best_branch_dir, 'best_model.pt'),
                os.path.join(args.output, 'elo_best_model.pt'))
    print(f'\n[v16] Elo #1: {best_name} (Elo={ranked[0][1]:.0f})')
    print(f'  Model saved: {os.path.join(args.output, "elo_best_model.pt")}')

    # Save all results
    final_results = {
        'branches': {name: {'elo': elo.get(name,1500), **results.get(name,{})} for name in elo},
        'elo_ranking': [(name, e) for name, e in ranked],
        'elo_best': best_name,
    }
    with open(os.path.join(args.output, 'league_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    print('[v16] Complete.')

if __name__ == '__main__':
    main()
