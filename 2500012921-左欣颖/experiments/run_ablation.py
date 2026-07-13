"""Feature ablation study: SL→RL with 3 feature sets.

Experiments:
  1. SL-only (no RL)
  2. RL-random-init (baseline features, no SL init)
  3. SL→RL baseline (6 channels)
  4. SL→RL key (15 channels)
  5. SL→RL full (21 channels)

Each RL run: short training (~80 episodes), record reward curve.
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# Project paths
SL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'SL')
RL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SL_DIR)
sys.path.insert(0, RL_DIR)

from model_var import CNNModelVar
from feature_agent_ext import make_agent_cls, FEATURE_SETS, ACT_SIZE
from env import MahjongGBEnv


def load_sl_init(model, sl_path, target_channels):
    """Load SL weights into RL model's tower, expanding first conv if needed."""
    sl_state = torch.load(sl_path, map_location='cpu', weights_only=True)
    model.load_sl_tower(sl_state, target_channels)
    return model


def run_rl_experiment(exp_name, agent_cls, obs_size, sl_model_path=None, n_episodes=80,
                      lr=1e-4, device='cuda', seed=42):
    """Single-process PPO training loop.

    Args:
        name: experiment name
        agent_cls: FeatureAgent class
        obs_size: number of observation channels
        sl_model_path: if given, load and expand SL weights for init
        n_episodes: number of episodes to run
        device: 'cuda' or 'cpu'
    """
    print(f'\n{"="*60}', flush=True)
    print(f'Experiment: {exp_name}', flush=True)
    print(f'  obs_size={obs_size}, episodes={n_episodes}, lr={lr}', flush=True)

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    model = CNNModelVar(in_channels=obs_size)

    if sl_model_path and os.path.exists(sl_model_path):
        print(f'  Loading SL init from {sl_model_path}', flush=True)
        model = load_sl_init(model, sl_model_path, obs_size)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create env (4-player self-play)
    env = MahjongGBEnv(config={'agent_clz': agent_cls})
    agent_names = env.agent_names

    # PPO hyperparams
    gamma = 0.98
    gae_lambda = 0.95
    clip_epsilon = 0.2
    value_coeff = 1.0
    entropy_coeff = 0.01
    ppo_epochs = 4
    batch_size = 256

    # Track metrics
    episode_rewards = []    # terminal reward per episode
    episode_lengths = []
    hu_episodes = []        # 1 if hu, 0 otherwise
    max_fan_rewards = []    # max positive reward (winner's fan-based score)
    loss_history = []

    t_start = time.time()

    for ep in range(n_episodes):
        # Self-play one episode
        obs_dict = env.reset()

        traj = {agent_name: {'obs': [], 'mask': [], 'act': [], 'rew': [], 'val': []}
                for agent_name in agent_names}

        done = False
        ep_len = 0
        terminal_rewards = None

        while not done:
            actions = {}
            values = {}
            for agent_name in obs_dict:
                traj[agent_name]['obs'].append(obs_dict[agent_name]['observation'])
                traj[agent_name]['mask'].append(obs_dict[agent_name]['action_mask'])

                obs_t = torch.tensor(obs_dict[agent_name]['observation'],
                                     dtype=torch.float).unsqueeze(0).to(device)
                mask_t = torch.tensor(obs_dict[agent_name]['action_mask'],
                                      dtype=torch.float).unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    logits, value = model({'observation': obs_t, 'action_mask': mask_t})
                    dist = Categorical(logits=logits)
                    action = dist.sample().item()
                    value = value.item()
                actions[agent_name] = action
                values[agent_name] = value
                traj[agent_name]['act'].append(action)
                traj[agent_name]['val'].append(value)

            next_obs, rewards, done = env.step(actions)
            for agent_name in rewards:
                traj[agent_name]['rew'].append(rewards[agent_name])
            if done:
                terminal_rewards = rewards

            obs_dict = next_obs
            ep_len += 1

        episode_lengths.append(ep_len)

        # Track terminal outcome
        if terminal_rewards:
            rew_values = list(terminal_rewards.values())
            ep_reward = sum(rew_values)
            max_r = max(rew_values)
            hu = max_r > 0  # positive reward = someone hu'd
        else:
            ep_reward = 0
            max_r = 0
            hu = False

        episode_rewards.append(ep_reward)
        hu_episodes.append(1 if hu else 0)
        max_fan_rewards.append(max_r)

        # Post-process trajectory (per agent)
        all_obs, all_mask, all_act, all_adv, all_target = [], [], [], [], []

        for agent_name in agent_names:
            data = traj[agent_name]
            if not data['act']:
                continue

            # Align lengths (reward may have 1 extra)
            n = len(data['act'])
            rews = data['rew'][:n] if len(data['rew']) >= n else (data['rew'] + [0])[:n]
            vals = data['val'][:n]
            next_vals = data['val'][1:] + [0]

            # Compute GAE advantages and TD targets
            td_targets = np.array(rews) + gamma * np.array(next_vals)
            td_deltas = td_targets - np.array(vals)

            advantages = []
            adv = 0.0
            for delta in reversed(td_deltas):
                adv = gamma * gae_lambda * adv + delta
                advantages.append(adv)
            advantages = np.array(advantages[::-1], dtype=np.float32)

            all_obs.append(np.stack(data['obs']))
            all_mask.append(np.stack(data['mask']))
            all_act.append(np.array(data['act'], dtype=np.int64))
            all_adv.append(advantages)
            all_target.append(td_targets.astype(np.float32))

        if not all_obs:
            continue

        # Merge across agents
        obs_arr = np.concatenate(all_obs)
        mask_arr = np.concatenate(all_mask)
        act_arr = np.concatenate(all_act)
        adv_arr = np.concatenate(all_adv)
        target_arr = np.concatenate(all_target)

        # Normalize advantages
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        # PPO update
        total_loss = 0
        n_updates = 0
        n_samples = len(act_arr)

        # Shuffle and batch
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = indices[start:start + batch_size]
            obs_b = torch.tensor(obs_arr[idx], dtype=torch.float).to(device)
            mask_b = torch.tensor(mask_arr[idx], dtype=torch.float).to(device)
            act_b = torch.tensor(act_arr[idx]).unsqueeze(-1).to(device)
            adv_b = torch.tensor(adv_arr[idx], dtype=torch.float).to(device)
            target_b = torch.tensor(target_arr[idx], dtype=torch.float).to(device)

            model.train()
            # Old log probs
            with torch.no_grad():
                old_logits, _ = model({'observation': obs_b, 'action_mask': mask_b})
                old_probs = F.softmax(old_logits, dim=1).gather(1, act_b)
                old_log_probs = torch.log(old_probs + 1e-8)

            for _ in range(ppo_epochs):
                logits, values = model({'observation': obs_b, 'action_mask': mask_b})
                dist = Categorical(logits=logits)
                probs = F.softmax(logits, dim=1).gather(1, act_b)
                log_probs = torch.log(probs + 1e-8)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv_b
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = torch.mean(F.mse_loss(values.squeeze(-1), target_b))
                entropy_loss = -torch.mean(dist.entropy())
                loss = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_updates += 1

        avg_loss = total_loss / max(n_updates, 1)
        loss_history.append(avg_loss)

        if (ep + 1) % 50 == 0 or ep == 0:
            elapsed = time.time() - t_start
            hu_rate = np.mean(hu_episodes[-50:]) * 100 if hu_episodes else 0
            avg_max_r = np.mean(max_fan_rewards[-50:]) if max_fan_rewards else 0
            print(f'  Ep {ep+1}/{n_episodes} | loss={avg_loss:.4f} | '
                  f'hu_rate={hu_rate:.0f}% | avg_max_fan={avg_max_r:.0f} | '
                  f'len={ep_len} | {elapsed:.0f}s', flush=True)

    elapsed = time.time() - t_start
    results = {
        'name': exp_name,
        'obs_size': obs_size,
        'episodes': n_episodes,
        'elapsed_sec': elapsed,
        'loss_history': loss_history,
        'episode_rewards': episode_rewards,
        'max_fan_rewards': max_fan_rewards,
        'hu_episodes': hu_episodes,
        'episode_lengths': episode_lengths,
        'final_hu_rate_50': float(np.mean(hu_episodes[-50:]) * 100 if hu_episodes else 0),
        'final_avg_max_fan_50': float(np.mean(max_fan_rewards[-50:]) if max_fan_rewards else 0),
        'reward_trend': float(np.mean(episode_rewards[-100:]) - np.mean(episode_rewards[:100])
                             if len(episode_rewards) >= 200 else 0),
    }
    return results


def evaluate_sl_policy(sl_model_path, agent_cls, n_episodes=50, device='cuda'):
    """Evaluate frozen SL policy (no RL) against itself."""
    print(f'\n{"="*60}', flush=True)
    print(f'Experiment: SL-only (evaluation)', flush=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location("sl_model",
        os.path.join(SL_DIR, 'model.py'))
    sl_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sl_mod)
    SLModel = sl_mod.CNNModel

    model = SLModel().to(device)
    model.load_state_dict(torch.load(sl_model_path, map_location=device, weights_only=True))
    model.eval()

    env = MahjongGBEnv(config={'agent_clz': agent_cls})
    agent_names = env.agent_names

    episode_rewards = []
    episode_lengths = []
    hu_rates = []

    for ep in range(n_episodes):
        obs_dict = env.reset()
        ep_reward = 0
        done = False
        ep_len = 0
        while not done:
            actions = {}
            for agent_name in obs_dict:
                obs_t = torch.tensor(obs_dict[agent_name]['observation'],
                                     dtype=torch.float).unsqueeze(0).to(device)
                mask_t = torch.tensor(obs_dict[agent_name]['action_mask'],
                                      dtype=torch.float).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model({'is_training': False, 'obs': {'observation': obs_t, 'action_mask': mask_t}})
                    dist = Categorical(logits=logits)
                    action = dist.sample().item()
                actions[agent_name] = action
            next_obs, rewards, done = env.step(actions)
            # Sum rewards across all players returned
            ep_reward += sum(rewards.values()) if rewards else 0
            obs_dict = next_obs
            ep_len += 1
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        if (ep + 1) % 10 == 0:
            print(f'  Ep {ep+1}/{n_episodes} | len={ep_len}', flush=True)

    return {
        'name': 'SL-only',
        'episodes': n_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_avg_reward_10': float(np.mean(episode_rewards[-10:])),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}', flush=True)

    sl_model_path = os.path.join(SL_DIR, 'model', 'checkpoint', 'model_20.pt')
    if not os.path.exists(sl_model_path):
        # Try other epoch
        for e in range(20, 0, -1):
            p = os.path.join(SL_DIR, 'model', 'checkpoint', f'model_{e}.pt')
            if os.path.exists(p):
                sl_model_path = p
                print(f'Found SL model: {p}', flush=True)
                break
        else:
            print('WARNING: No SL model found, using random init for all experiments',
                  flush=True)
            sl_model_path = None

    all_results = {}
    N = 500  # episodes per experiment

    # Experiment 1: RL-random-init (baseline features, no SL init)
    baseline_cls = make_agent_cls(6)
    rl_random = run_rl_experiment(
        'RL-random-init (6ch)', baseline_cls, obs_size=6,
        sl_model_path=None, n_episodes=N, device=device,
    )
    all_results['rl_random'] = rl_random

    # Experiment 2: SL→RL baseline (6 channels)
    sl_rl_base = run_rl_experiment(
        'SL→RL baseline (6ch)', baseline_cls, obs_size=6,
        sl_model_path=sl_model_path, n_episodes=N, device=device,
    )
    all_results['sl_rl_baseline'] = sl_rl_base

    # Experiment 3: SL→RL key (15 channels)
    key_cls = make_agent_cls(15)
    sl_rl_key = run_rl_experiment(
        'SL→RL key (15ch)', key_cls, obs_size=15,
        sl_model_path=sl_model_path, n_episodes=N, device=device,
    )
    all_results['sl_rl_key'] = sl_rl_key

    # Experiment 4: SL→RL full (21 channels)
    full_cls = make_agent_cls(21)
    sl_rl_full = run_rl_experiment(
        'SL→RL full (21ch)', full_cls, obs_size=21,
        sl_model_path=sl_model_path, n_episodes=N, device=device,
    )
    all_results['sl_rl_full'] = sl_rl_full

    # Save results
    result_path = os.path.join(RL_DIR, 'ablation_results.json')
    # Convert numpy arrays to lists for JSON
    json_safe = {}
    for k, v in all_results.items():
        json_safe[k] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else
                              [float(x) for x in vv] if isinstance(vv, list) and vv and isinstance(vv[0], (np.floating, np.integer)) else
                              vv)
                        for kk, vv in v.items()}
    with open(result_path, 'w') as f:
        json.dump(json_safe, f, indent=2)
    print(f'\nResults saved to {result_path}', flush=True)

    # Summary
    print(f'\n{"="*60}', flush=True)
    print('SUMMARY', flush=True)
    print(f'{"="*60}', flush=True)
    for name, r in json_safe.items():
        hu = r.get('final_hu_rate_50', 'N/A')
        fan = r.get('final_avg_max_fan_50', 'N/A')
        print(f'{r["name"]}: hu_rate={hu:.0f}% avg_fan={fan:.0f}',
              flush=True)


if __name__ == '__main__':
    main()
