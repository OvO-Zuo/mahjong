"""
v11 — KL-constrained dual-head SL+RL mixture policy
       + adversarial exploiter replay + strict SL safety rollback.

Core:
  π = α·π_SL + (1-α)·π_RL,  α = clamp(KL / 0.04, 0.2, 0.8)
  KL ≤ 0.04: normal RL+SL update
  KL > 0.04: RL gradient frozen, SL projection only
  SL weight ≥ RL (1.5~3x)

Replay: SL expert (30%) + self-play RL (≤50%) + exploiter (≥20%, priority x2)
Exploiter: train every 50ep, max KL under win constraint
Safety: SL_win < 0.55 or rolling drop > 0.03 → rollback + freeze backbone
Entropy: tight band 0.06 ± 0.01
"""

import os, sys, json, time, math
import torch, torch.nn.functional as F
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RL'))
from feature import FeatureAgent
from env import MahjongGBEnv
from dual_head_model import DualHeadModel, compute_kl_sl_rl

# ======================================================================
CFG = {
    'lr': 3e-5, 'clip': 0.1,
    'gamma': 0.98, 'gae_lambda': 0.95,
    'value_coeff': 0.3, 'ppo_epochs': 3, 'batch_size': 256,
    'min_buffer_size': 512,
    'total_episodes': 2000, 'eval_interval': 200, 'eval_games': 80,
    'print_interval': 50, 'ckpt_interval': 300,

    # Dual-head mixing — curriculum phases
    'alpha_phase1': 0.50,            # ep 0-400: equal SL/RL, KL soft only
    'alpha_phase2': 0.60,            # ep 400-1000: KL gate ON
    'alpha_phase3_range': [0.50, 0.80],  # ep 1000+: adaptive by SL_win
    'kl_ref': 0.06,                  # KL reference (target center)
    'alpha_min': 0.50,               # Never below 50% SL
    'alpha_max': 0.80,               # Never above 80% SL
    'kl_hard_gate_phase1': 0.20,     # Phase 1: gate very loose (effectively off)
    'kl_hard_gate_phase2': 0.08,     # Phase 2: moderate gate
    'kl_hard_gate_phase3': 0.06,     # Phase 3: tight gate
    'kl_target_low': 0.03,           # Target KL band
    'kl_target_high': 0.08,

    # Loss weights
    'sl_weight': 1.5,                # SL ≥ RL, moderate ratio
    'rl_weight': 1.0,
    'exploiter_priority': 2.0,

    # Replay ratios
    'sl_expert_ratio': 0.30,
    'rl_selfplay_ratio': 0.50,
    'exploiter_replay_ratio': 0.20,

    # Exploiter
    'exploiter_train_interval': 50,
    'exploiter_kl_weight': 0.3,

    # KL soft penalty (phase 1) vs gate (phase 2-3)
    'kl_soft_coeff': 0.03,           # Phase 1: soft KL penalty only
    'kl_rollback': 0.15,             # Emergency rollback only
    'kl_freeze_rl': 0.08,            # Freeze RL when KL > this

    # Entropy
    'entropy_target': 0.06,
    'entropy_band': 0.01,
    'entropy_coeff': 0.05,

    # Safety (light)
    'safety_freeze_backbone': 50,

    # Buffer sizes
    'sl_expert_buf_size': 12000,
    'rl_selfplay_buf_size': 20000,
    'exploiter_buf_size': 8000,

    # Paths
    'sl_model_path': os.path.join(os.path.dirname(__file__), '..', 'SL', 'model',
                                   'checkpoint', 'model_20.pt'),
    'ckpt_dir': os.path.join(os.path.dirname(__file__), 'league_checkpoints_v11'),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# ======================================================================
class PrioritizedBuffer:
    def __init__(self, cfg):
        self.sl_expert = deque(maxlen=cfg['sl_expert_buf_size'])
        self.rl_selfplay = deque(maxlen=cfg['rl_selfplay_buf_size'])
        self.exploiter = deque(maxlen=cfg['exploiter_buf_size'])

    def push_sl(self, s): self.sl_expert.append(s)
    def push_rl(self, s): self.rl_selfplay.append(s)
    def push_exploiter(self, s): self.exploiter.append(s)

    def sample(self, bs, cfg):
        n_sl = int(bs * cfg['sl_expert_ratio'])
        n_exp = int(bs * cfg['exploiter_replay_ratio'])
        n_rl = bs - n_sl - n_exp

        samples = []
        for buf, n in [(self.sl_expert, n_sl), (self.exploiter, n_exp), (self.rl_selfplay, n_rl)]:
            if n > 0 and len(buf) > 0:
                idx = np.random.choice(len(buf), min(n, len(buf)), replace=False)
                for i in idx: samples.append(buf[i])
        if len(samples) < 32: return None

        np.random.shuffle(samples)
        obs = torch.from_numpy(np.stack([s['obs'] for s in samples]).astype(np.float32))
        masks = torch.from_numpy(np.stack([s['mask'] for s in samples]).astype(np.float32))
        acts = torch.tensor([s['act'] for s in samples], dtype=torch.long)
        advs = torch.tensor([s.get('adv',0) for s in samples], dtype=torch.float32)
        tgts = torch.tensor([s.get('tgt',0) for s in samples], dtype=torch.float32)
        old_lp = torch.tensor([s.get('old_lp',0) for s in samples], dtype=torch.float32)
        tiers = [s.get('tier','rl') for s in samples]
        return obs, masks, acts, advs, tgts, old_lp, tiers

    def total(self): return len(self.sl_expert)+len(self.rl_selfplay)+len(self.exploiter)


# ======================================================================
def model_infer(model, obs, agent, device, argmax=False):
    obs_t = torch.from_numpy(np.expand_dims(obs['observation'],0)).float().to(device)
    mask_t = torch.from_numpy(np.expand_dims(obs['action_mask'],0)).float().to(device)
    with torch.no_grad():
        logits, _, _, _ = model({'observation': obs_t, 'action_mask': mask_t}, 'mixture')
    if argmax: return int(logits[0].argmax().item())
    return int(torch.multinomial(F.softmax(logits,dim=-1),1).item())


def run_game(model, sl_model, cfg):
    device = cfg['device']
    opp_nets = [sl_model]*3 if sl_model else [model]*3
    models = [model] + opp_nets
    env = MahjongGBEnv({'agent_clz': FeatureAgent, 'duplicate': True, 'variety': 10000})
    obs_dict = env.reset()
    agents = [FeatureAgent(i) for i in range(4)]
    for i in range(4): agents[i].request2obs('Wind %d'%(i%4))

    traj = []; done = False; turns = 0
    while not done and turns < 500:
        actions = {}
        for name in env.agent_names:
            i = int(name.split('_')[1])-1
            obs = obs_dict.get(name)
            if obs is not None:
                action = model_infer(models[i], obs, agents[i], device, argmax=(i!=0))
                if i==0:
                    with torch.no_grad():
                        ot=torch.from_numpy(np.expand_dims(obs['observation'],0)).float().to(device)
                        mt=torch.from_numpy(np.expand_dims(obs['action_mask'],0)).float().to(device)
                        logits,val,sl_l,rl_l = model({'observation':ot,'action_mask':mt},'mixture')
                        lp = F.log_softmax(logits,dim=-1)[0,action].item()
                        sl_a = int(sl_l[0].argmax().item())
                    traj.append({'obs':obs['observation'].copy(),'mask':obs['action_mask'].copy(),
                                 'act':action,'sl_act':sl_a,'val':val.item(),'old_lp':lp,'r':0.0})
                actions[name]=action
        if not actions: break
        obs_dict, reward_dict, done_dict = env.step(actions)
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


# ======================================================================
def dual_head_update(model, sl_model, optimizer, buffer, exploiter_model, cfg, ep, sl_win_hist):
    batch = buffer.sample(cfg['batch_size'], cfg)
    if batch is None:
        return {'pl':0,'vl':0,'ent':0,'kl':0,'sl_loss':0,'rl_act':0,'alpha':model.current_alpha}

    obs,masks,acts,advs,tgts,old_lp,tiers = batch
    device=cfg['device']
    obs,masks=obs.to(device),masks.to(device); acts=acts.to(device)
    advs=advs.to(device); tgts=tgts.to(device); old_lp=old_lp.to(device)
    advs=(advs-advs.mean())/(advs.std()+1e-8)

    is_exp = torch.tensor([t=='exploiter' for t in tiers], device=device)
    is_sl = torch.tensor([t=='sl' for t in tiers], device=device)

    # KL check
    kl_val = compute_kl_sl_rl(model, obs, masks)

    # Curriculum: phase-dependent α and KL gate
    if ep <= 400:       # Phase 1: let RL head explore, soft KL only
        model.set_alpha_manual(cfg['alpha_phase1'])
        gate = cfg['kl_hard_gate_phase1']
        kl_penalty_only = True
    elif ep <= 1000:    # Phase 2: balanced, moderate KL gate
        model.set_alpha_manual(cfg['alpha_phase2'])
        gate = cfg['kl_hard_gate_phase2']
        kl_penalty_only = False
    else:               # Phase 3: adaptive by SL_win
        if len(sl_win_hist) >= 50:
            rw = np.mean(list(sl_win_hist)[-50:])
            # High win → can afford more RL (lower α)
            adapt_alpha = 0.80 - 0.30 * max(0, min(1, (rw - 0.4) / 0.2))
            adapt_alpha = max(0.50, min(0.80, adapt_alpha))
        else:
            adapt_alpha = 0.65
        model.set_alpha_manual(adapt_alpha)
        gate = cfg['kl_hard_gate_phase3']
        kl_penalty_only = False

    rl_frozen = (not kl_penalty_only) and (kl_val > gate)

    # Emergency rollback only on extreme KL
    if kl_val > cfg['kl_rollback']:
        return {'pl':0,'vl':0,'ent':0,'kl':kl_val,'sl_loss':0,'rl_act':0,
                'alpha':model.current_alpha,'rollback':True}

    total_pl,total_vl,total_ent,total_kl,total_sl=0,0,0,0,0
    rl_ct=0

    for _ in range(cfg['ppo_epochs']):
        logits,values,sl_logits,rl_logits = model({'observation':obs,'action_mask':masks},'mixture')
        probs=F.softmax(logits,dim=-1); log_probs=F.log_softmax(logits,dim=-1)
        sel_lp=log_probs.gather(1,acts.unsqueeze(1)).squeeze(1)
        entropy=-(probs*log_probs).sum(dim=-1).mean()

        # SL loss (primary, weight ≥ RL ×1.5)
        sl_targets=sl_logits.argmax(dim=-1).detach().clamp(0,234)
        w=torch.where(is_exp,cfg['exploiter_priority'],1.0)
        sl_loss=(w*F.cross_entropy(logits,sl_targets,reduction='none'))
        sl_loss=sl_loss[torch.isfinite(sl_loss)].mean() if torch.isfinite(sl_loss).any() else torch.tensor(0.0,device=device)

        # RL loss (frozen if KL > 0.04)
        if not rl_frozen:
            ratio=torch.exp(sel_lp-old_lp)
            clip_adv=torch.clamp(ratio,1-cfg['clip'],1+cfg['clip'])*advs
            policy_loss=-torch.min(ratio*advs,clip_adv).mean()
            value_loss=F.mse_loss(values.squeeze(1),tgts)
            rl_ct=len(acts)
        else:
            policy_loss=torch.tensor(0.0,device=device)
            value_loss=torch.tensor(0.0,device=device)

        # KL(RL || SL)
        rl_probs=F.softmax(rl_logits,dim=-1); rl_lp2=F.log_softmax(rl_logits,dim=-1)
        sl_lp2=F.log_softmax(sl_logits,dim=-1)
        valid=(masks>0.5).float()
        kl=(valid*rl_probs*(rl_lp2-sl_lp2.detach())).sum(dim=-1).mean()

        # Entropy
        ent_dev=abs(entropy.item()-cfg['entropy_target'])
        ent_bonus=2.0 if ent_dev>cfg['entropy_band'] else 1.0
        ent_sign=-1 if entropy.item()<cfg['entropy_target'] else 1

        # KL penalty: soft in phase 1, gate-based in phase 2-3
        kl_loss = cfg['kl_soft_coeff'] * kl if kl_penalty_only else torch.tensor(0.0, device=device)

        loss=(cfg['sl_weight']*sl_loss+
              cfg['value_coeff']*value_loss+
              cfg['entropy_coeff']*entropy*ent_bonus*ent_sign+
              kl_loss)
        if not rl_frozen:
            loss=loss+cfg['rl_weight']*policy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()

        total_pl+=policy_loss.item(); total_vl+=value_loss.item()
        total_ent+=entropy.item(); total_kl+=kl.item(); total_sl+=sl_loss.item()

    n=cfg['ppo_epochs']
    return {'pl':total_pl/n,'vl':total_vl/n,'ent':total_ent/n,'kl':total_kl/n,
            'sl_loss':total_sl/n,'rl_act':rl_ct,'alpha':model.current_alpha}


# ======================================================================
def train_exploiter(exp_model, model, sl_model, buffer, opt, cfg):
    """Train exploiter: max KL vs current policy under win constraint."""
    device=cfg['device']
    if buffer.total()<128: return
    batch=buffer.sample(min(128,buffer.total()),cfg)
    if batch is None: return
    obs,masks,_,_,_,_,_=batch; obs,masks=obs.to(device),masks.to(device)

    with torch.no_grad():
        _,_,sl_l,cur_rl=model({'observation':obs,'action_mask':masks},'mixture')
        cur_rl_probs=F.softmax(cur_rl,dim=-1)

    for _ in range(3):
        _,_,_,exp_rl=exp_model({'observation':obs,'action_mask':masks},'mixture')
        exp_probs=F.softmax(exp_rl,dim=-1); exp_lp=F.log_softmax(exp_rl,dim=-1)
        cur_lp=F.log_softmax(cur_rl,dim=-1).detach()
        kl_gap=(exp_probs*(exp_lp-cur_lp)).sum(dim=-1).mean()
        with torch.no_grad():
            _,_,sl_l2,_=exp_model({'observation':obs,'action_mask':masks},'mixture')
            sl_lp2=F.log_softmax(sl_l2,dim=-1)
            kl_to_sl=(exp_probs*(exp_lp-sl_lp2.detach())).sum(dim=-1).mean()
        loss=-kl_gap+cfg['exploiter_kl_weight']*kl_to_sl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(exp_model.parameters(),1.0); opt.step()


def evaluate_vs_sl(model,sl_model,cfg,n=50):
    device=cfg['device']; wins,tf,tt=0,0,0
    for g in range(n):
        sm=model if g%2==0 else sl_model
        r=run_game(sm,sl_model,cfg)
        rl_won=(g%2==0 and r['ws']==0) or (g%2==1 and r['ws']!=0)
        if rl_won: wins+=1
        if r['hu']: tf+=r['fan']
        tt+=r['turns']
    return {'win':wins/n,'fan':tf/max(wins,1),'turns':tt/n}


# ======================================================================
def train():
    cfg=CFG; os.makedirs(cfg['ckpt_dir'],exist_ok=True)
    device=cfg['device']
    print(f'[v11] Dual-Head KL-Constrained. Device:{device}')

    # SL anchor
    sl_model=DualHeadModel().to(device)
    sl_ckpt=torch.load(cfg['sl_model_path'],map_location=device,weights_only=False)
    sl_model.load_sl_checkpoint(sl_ckpt)
    sl_model.eval(); sl_model.freeze_sl()
    for p in sl_model.parameters(): p.requires_grad=False

    # RL model
    model=DualHeadModel().to(device)
    model.load_sl_checkpoint(sl_ckpt)
    model.freeze_sl()  # Freeze tower + SL head initially

    # Exploiter
    exp_model=DualHeadModel().to(device)
    exp_model.load_sl_checkpoint(sl_ckpt)
    exp_model.freeze_sl()
    exp_opt=torch.optim.Adam([p for p in exp_model.parameters() if p.requires_grad],lr=cfg['lr']*0.3)

    # Buffer
    buf=PrioritizedBuffer(cfg)

    # Optimizer (only RL head + value head)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'])

    # Safety
    best_ckpt={k:v.cpu().clone() for k,v in model.state_dict().items()}
    best_ckpt_ep=0; sl_frozen_until=0
    total_games,best_sl_win,rb=0,0.0,0
    sl_win_hist=deque(maxlen=200)

    for ep in range(1,cfg['total_episodes']+1):
        # 1. Game
        r=run_game(model,sl_model,cfg); total_games+=1

        # 2. Fill buffers
        for s in r.get('samples',[]):
            smp={'obs':s['obs'],'mask':s['mask'],'act':s['act'],
                  'adv':s.get('adv',0),'tgt':s.get('tgt',0),'old_lp':s.get('old_lp',0)}
            # SL expert (always)
            sl_smp=dict(smp); sl_smp['act']=s['sl_act']; sl_smp['tier']='sl'
            buf.push_sl(sl_smp)
            # RL self-play
            smp['tier']='rl'; buf.push_rl(smp)

        # 3. Exploiter generates adversarial samples
        if ep%cfg['exploiter_train_interval']==0 and buf.total()>=256:
            train_exploiter(exp_model,model,sl_model,buf,exp_opt,cfg)
            # Generate exploiter trajectories
            with torch.no_grad():
                exp_traj=run_game(exp_model,sl_model,cfg)
                for s in exp_traj.get('samples',[]):
                    es={'obs':s['obs'],'mask':s['mask'],'act':s['act'],
                        'adv':-abs(s.get('adv',0)),'tgt':s.get('tgt',0),
                        'old_lp':s.get('old_lp',0),'tier':'exploiter'}
                    buf.push_exploiter(es)
            print(f'  [EXPLOITER] trained, buf={buf.total()}')

        # 4. Update
        stats={'pl':0,'vl':0,'ent':0,'kl':0,'sl_loss':0,'rl_act':0,'alpha':model.current_alpha}
        if buf.total()>=cfg['min_buffer_size']:
            stats=dual_head_update(model,sl_model,opt,buf,exp_model,cfg,ep,sl_win_hist)

            if stats.get('rollback'):
                model.load_state_dict(best_ckpt); rb+=1
                model.freeze_sl(); sl_frozen_until=ep+cfg['safety_freeze_backbone']
                print(f'  [ROLLBACK] KL={stats["kl"]:.4f} → restored ep{best_ckpt_ep}, '
                      f'freeze tower until ep{sl_frozen_until}')
                continue

            if stats['kl']<cfg['kl_ref'] and ep>sl_frozen_until:
                best_ckpt={k:v.cpu().clone() for k,v in model.state_dict().items()}
                best_ckpt_ep=ep

        # 5. Eval
        if ep%cfg['eval_interval']==0:
            ev=evaluate_vs_sl(model,sl_model,cfg,n=cfg['eval_games'])
            sl_win_hist.append(ev['win'])
            rolling=np.mean(list(sl_win_hist)[-50:]) if len(sl_win_hist)>=50 else np.mean(sl_win_hist)

            if ev['win']>best_sl_win:
                best_sl_win=ev['win']
                torch.save({'sd':model.state_dict(),'ep':ep,'sl_win':best_sl_win},
                           os.path.join(cfg['ckpt_dir'],'best_model.pt'))

            print(f'\n[Eval ep{ep}] SL_win={ev["win"]:.3f} rolling={rolling:.3f} best={best_sl_win:.3f} '
                  f'KL={stats["kl"]:.4f} α={stats["alpha"]:.2f} fan={ev["fan"]:.0f}')

            if rolling>=0.58 and len(sl_win_hist)>=100 and stats['kl']<0.04:
                print(f'[v11] STOP: rolling={rolling:.3f} KL={stats["kl"]:.4f}')
                break

        # 6. Print
        if ep%cfg['print_interval']==0:
            a=stats['alpha']; kl_s='RB' if stats.get('rollback') else f'{stats["kl"]:.4f}'
            fz='F' if ep<=sl_frozen_until else ''
            print(f'[Ep{ep:5d}] g={total_games:5d} buf={buf.total():5d} '
                  f'hu={r["hu"]} fan={r["fan"]:2d} sl={stats["sl_loss"]:.3f} pl={stats["pl"]:.3f} '
                  f'kl={kl_s} ent={stats["ent"]:.3f} α={a:.2f}{fz} rl={stats["rl_act"]} rb={rb}')

        # 7. Ckpt
        if ep%cfg['ckpt_interval']==0:
            torch.save({'sd':model.state_dict(),'ep':ep,'sl_win':best_sl_win},
                       os.path.join(cfg['ckpt_dir'],f'ckpt_ep{ep}.pt'))

    torch.save({'sd':model.state_dict(),'ep':ep,'sl_win':best_sl_win},
               os.path.join(cfg['ckpt_dir'],'final_model.pt'))
    print(f'[v11] Done. best_sl_win={best_sl_win:.3f} games={total_games} rollbacks={rb}')


if __name__=='__main__':
    train()
