"""Feature #1 ablation: Ting Indicator.

Compares RL-baseline (6ch) vs RL+ting (7ch + intermediate reward).
"""
import os, sys, time, json, numpy as np
import torch, torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SL'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RL'))

from model_var import CNNModelVar
from feature_agent_ext import make_agent_cls, TILE_LIST, OFFSET_TILE, OFFSET_ACT
from env import MahjongGBEnv

try:
    from MahjongGB import MahjongFanCalculator
except ImportError:
    print('MahjongGB required!')
    raise

device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EPISODES = 500
SEED = 42

# ── Ting Agent (7ch) ────────────────────────────────────────
OBS_SIZE_TING = 7   # 6 base + 1 ting channel

class TingAgent:
    """Wrapper that adds ting channel to observation and provides intermediate reward."""

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
            self.hand = t[1:]
            self._hand_update()
            return
        if t[0] == 'Huang':
            self.valid = []
            return self._obs()
        if t[0] == 'Draw':
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_update()
            self._compute_ting()
            if self._check_mahjong(tile, isSelfDrawn=True, isAboutKong=self.isAboutKong):
                self.valid.append(OFFSET_ACT['Hu'])
            for t_ in set(self.hand):
                self.valid.append(OFFSET_ACT['Play'] + OFFSET_TILE[t_])
                if self.hand.count(t_) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(OFFSET_ACT['AnGang'] + OFFSET_TILE[t_])
            if not self.wallLast and self.tileWall[0] > 0:
                for pt, t_, _ in self.packs[0]:
                    if pt == 'PENG' and t_ in self.hand:
                        self.valid.append(OFFSET_ACT['BuGang'] + OFFSET_TILE[t_])
            return self._obs()
        p = (int(t[1]) + 4 - self.seatWind) % 4
        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return
        if t[2] in ('Invalid', 'Hu'):
            self.valid = []
            return self._obs()
        if t[2] == 'Play':
            self.tileFrom = p
            self.curTile = t[3]
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)
            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_update()
                return
            else:
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(OFFSET_ACT['Hu'])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(OFFSET_ACT['Peng'] + OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(OFFSET_ACT['Gang'] + OFFSET_TILE[self.curTile])
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = [color + str(num + i) for i in range(-2, 3)]
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)
                self.valid.append(OFFSET_ACT['Pass'])
                return self._obs()
        if t[2] == 'Chi':
            tile = t[3]; color, num = tile[0], int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2): self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                self.valid = []; self.hand.append(self.curTile)
                for i in range(-1, 2): self.hand.remove(color + str(num + i))
                self._hand_update()
                for t_ in set(self.hand): self.valid.append(OFFSET_ACT['Play'] + OFFSET_TILE[t_])
                return self._obs()
            return
        if t[2] in ('UnChi', 'UnPeng'): return
        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                self.valid = []
                for _ in range(2): self.hand.remove(self.curTile)
                self._hand_update()
                for t_ in set(self.hand): self.valid.append(OFFSET_ACT['Play'] + OFFSET_TILE[t_])
                return self._obs()
            return
        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            if p == 0:
                for _ in range(3): self.hand.remove(self.curTile)
                self._hand_update(); self.isAboutKong = True
            return
        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            if p == 0: self.isAboutKong = True; [self.hand.remove(tile) for _ in range(4)]
            else: self.isAboutKong = False
            return
        if t[2] == 'BuGang':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2]); break
            self.shownTiles[tile] += 1
            if p == 0:
                self.hand.remove(tile); self._hand_update(); self.isAboutKong = True; return
            else:
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn=False, isAboutKong=True):
                    self.valid.append(OFFSET_ACT['Hu'])
                self.valid.append(OFFSET_ACT['Pass'])
                return self._obs()
        raise NotImplementedError(f'Unknown: {request}')

    def action2response(self, action):
        if action < OFFSET_ACT['Hu']: return 'Pass'
        if action < OFFSET_ACT['Play']: return 'Hu'
        if action < OFFSET_ACT['Chi']: return 'Play ' + TILE_LIST[action - OFFSET_ACT['Play']]
        if action < OFFSET_ACT['Peng']:
            t = (action - OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < OFFSET_ACT['Gang']: return 'Peng'
        if action < OFFSET_ACT['AnGang']: return 'Gang'
        if action < OFFSET_ACT['BuGang']: return 'Gang ' + TILE_LIST[action - OFFSET_ACT['AnGang']]
        return 'BuGang ' + TILE_LIST[action - OFFSET_ACT['BuGang']]

    def response2action(self, response):
        t = response.split()
        if t[0] == 'Pass': return OFFSET_ACT['Pass']
        if t[0] == 'Hu': return OFFSET_ACT['Hu']
        if t[0] == 'Play': return OFFSET_ACT['Play'] + OFFSET_TILE[t[1]]
        if t[0] == 'Chi': return OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
        if t[0] == 'Peng': return OFFSET_ACT['Peng'] + OFFSET_TILE[t[1]]
        if t[0] == 'Gang': return OFFSET_ACT['Gang'] + OFFSET_TILE[t[1]]
        if t[0] == 'AnGang': return OFFSET_ACT['AnGang'] + OFFSET_TILE[t[1]]
        if t[0] == 'BuGang': return OFFSET_ACT['BuGang'] + OFFSET_TILE[t[1]]
        return OFFSET_ACT['Pass']

    def _obs(self):
        mask = np.zeros(235)
        for a in self.valid: mask[a] = 1
        return {'observation': self.obs.reshape((7, 4, 9)).copy(), 'action_mask': mask}

    def _hand_update(self):
        self.obs[2:6] = 0
        d = defaultdict(int)
        for tile in self.hand: d[tile] += 1
        for tile, cnt in d.items():
            self.obs[2:2 + cnt, OFFSET_TILE[tile]] = 1

    def _compute_ting(self):
        """Fill channel 6: which discards lead to ting (1-shanten)."""
        self.obs[6] = 0
        if len(self.hand) not in (2, 5, 8, 11, 14):
            return  # Non-standard hand size
        # For each unique discard, check if any draw completes the hand
        for unique_tile in set(self.hand):
            test_hand = list(self.hand)
            test_hand.remove(unique_tile)
            # Try each possible "win tile" (sample common tiles only for speed)
            for win_tile in set(test_hand + list(self.shownTiles.keys())):
                if win_tile.startswith('F') or win_tile.startswith('J'):
                    continue
                try:
                    fans = MahjongFanCalculator(
                        pack=tuple(self.packs[0]),
                        hand=tuple(test_hand),
                        winTile=win_tile,
                        flowerCount=0, isSelfDrawn=True,
                        is4thTile=(self.shownTiles.get(win_tile, 0)) == 3,
                        isAboutKong=False, isWallLast=self.wallLast,
                        seatWind=self.seatWind, prevalentWind=self.prevalentWind,
                        verbose=True,
                    )
                    fan_sum = sum(fp * c for fp, c, _, _ in fans)
                    if fan_sum >= 8:
                        self.obs[6][OFFSET_TILE[unique_tile]] = 1
                        break
                except Exception:
                    pass

    def _check_mahjong(self, winTile, isSelfDrawn=False, isAboutKong=False):
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[0]), hand=tuple(self.hand), winTile=winTile,
                flowerCount=0, isSelfDrawn=isSelfDrawn,
                is4thTile=(self.shownTiles.get(winTile, 0) + isSelfDrawn) == 4,
                isAboutKong=isAboutKong, isWallLast=self.wallLast,
                seatWind=self.seatWind, prevalentWind=self.prevalentWind, verbose=True,
            )
            if sum(fp * c for fp, c, _, _ in fans) < 8: raise Exception
        except Exception:
            return False
        return True

    def is_ting(self):
        """Check if current hand is in ting state (any discard leads to win)."""
        return self.obs[6].sum() > 0


# Make env-compatible class
class TingAgentCls:
    def __init__(self, seatWind):
        return TingAgent(seatWind)


# ── Training Loop ──────────────────────────────────────────
def run_experiment(name, use_ting=False, n_episodes=N_EPISODES, seed=SEED):
    print(f'\n{"="*50}', flush=True)
    print(f'{name} | {n_episodes} episodes', flush=True)

    torch.manual_seed(seed); np.random.seed(seed)

    if use_ting:
        obs_size = 7
        from types import SimpleNamespace
        agent_wrapper = SimpleNamespace(__init__=lambda self, sw: TingAgent(sw),
                                         action2response=lambda self, a: TingAgent(0).action2response(a),
                                         response2action=lambda self, r: TingAgent(0).response2action(r),
                                         request2obs=lambda self, r: TingAgent(0).request2obs(r))
        # Use the real env with TingAgent
        model = CNNModelVar(in_channels=7).to(device)
    else:
        obs_size = 6
        AgentCls = make_agent_cls(6)
        model = CNNModelVar(in_channels=6).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # For ting experiment, we need the env to use TingAgent
    if use_ting:
        # We need a proper MahjongGBAgent subclass for the env
        from agent import MahjongGBAgent
        class TingMahjongAgent(MahjongGBAgent):
            def __init__(self, seatWind):
                self._agent = TingAgent(seatWind)
            def request2obs(self, r):
                result = self._agent.request2obs(r)
                # Return None/empty for notification requests
                return result
            def action2response(self, a):
                return self._agent.action2response(a)
            def response2action(self, r):
                return self._agent.response2action(r)
        env = MahjongGBEnv(config={'agent_clz': TingMahjongAgent})
    else:
        env = MahjongGBEnv(config={'agent_clz': make_agent_cls(6)})

    agent_names = env.agent_names
    gamma, gae_lambda = 0.98, 0.95
    clip_eps, value_coeff, entropy_coeff = 0.2, 1.0, 0.01
    ppo_epochs, batch_size = 4, 256

    hu_rates, max_fans, lengths = [], [], []
    prev_was_ting = defaultdict(bool)  # for intermediate ting reward

    t0 = time.time()

    for ep in range(n_episodes):
        obs_dict = env.reset()
        traj = {a: {'obs':[], 'mask':[], 'act':[], 'rew':[], 'val':[], 'agent': None}
                for a in agent_names}
        done, ep_len = False, 0
        terminal_rewards = None

        while not done:
            actions, values = {}, {}
            for a in obs_dict:
                traj[a]['obs'].append(obs_dict[a]['observation'])
                traj[a]['mask'].append(obs_dict[a]['action_mask'])
                # Store agent reference for ting reward
                if use_ting and traj[a]['agent'] is None:
                    for ag in env.agents:
                        if hasattr(ag, '_agent'):
                            traj[a]['agent'] = ag._agent
                            break
                ot = torch.tensor(obs_dict[a]['observation'], dtype=torch.float).unsqueeze(0).to(device)
                mt = torch.tensor(obs_dict[a]['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    logits, value = model({'observation': ot, 'action_mask': mt})
                    dist = Categorical(logits=logits)
                    action = dist.sample().item()
                actions[a] = action
                values[a] = value.item()
                traj[a]['act'].append(action)
                traj[a]['val'].append(value.item())

            next_obs, rewards, done = env.step(actions)

            # Add ting intermediate reward
            if use_ting:
                for a in rewards:
                    r = rewards[a]
                    agent = traj[a].get('agent')
                    if agent and hasattr(agent, 'is_ting'):
                        is_ting = agent.is_ting()
                        was_ting = prev_was_ting[a]
                        if is_ting and not was_ting:
                            r += 2.0  # just reached ting
                        prev_was_ting[a] = is_ting
                    traj[a]['rew'].append(r)
            else:
                for a in rewards:
                    traj[a]['rew'].append(rewards[a])

            if done: terminal_rewards = rewards
            obs_dict = next_obs
            ep_len += 1

        lengths.append(ep_len)
        if terminal_rewards:
            rv = list(terminal_rewards.values())
            hu = max(rv) > 0; max_r = max(rv)
        else:
            hu = False; max_r = 0
        hu_rates.append(1 if hu else 0)
        max_fans.append(max_r)

        # PPO update
        all_obs, all_mask, all_act, all_adv, all_target = [], [], [], [], []
        for a in agent_names:
            data = traj[a]
            if not data['act']: continue
            n = len(data['act'])
            rews = data['rew'][:n] if len(data['rew']) >= n else (data['rew'] + [0])[:n]
            vals = data['val'][:n]
            nvs = data['val'][1:] + [0]
            td_t = np.array(rews) + gamma * np.array(nvs)
            td_d = td_t - np.array(vals)
            advs = []; adv = 0.0
            for d in reversed(td_d):
                adv = gamma * gae_lambda * adv + d; advs.append(adv)
            advs = np.array(advs[::-1], dtype=np.float32)
            all_obs.append(np.stack(data['obs']))
            all_mask.append(np.stack(data['mask']))
            all_act.append(np.array(data['act'], dtype=np.int64))
            all_adv.append(advs)
            all_target.append(td_t.astype(np.float32))

        if not all_obs: continue
        obs_arr = np.concatenate(all_obs)
        mask_arr = np.concatenate(all_mask)
        act_arr = np.concatenate(all_act)
        adv_arr = np.concatenate(all_adv)
        tgt_arr = np.concatenate(all_target)
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        total_loss = 0; n_upd = 0
        indices = np.random.permutation(len(act_arr))
        for start in range(0, len(act_arr), batch_size):
            idx = indices[start:start+batch_size]
            ob = torch.tensor(obs_arr[idx], dtype=torch.float).to(device)
            mb = torch.tensor(mask_arr[idx], dtype=torch.float).to(device)
            ab = torch.tensor(act_arr[idx]).unsqueeze(-1).to(device)
            dvb = torch.tensor(adv_arr[idx], dtype=torch.float).to(device)
            tgb = torch.tensor(tgt_arr[idx], dtype=torch.float).to(device)
            model.train()
            with torch.no_grad():
                ol, _ = model({'observation': ob, 'action_mask': mb})
                olp = torch.log(F.softmax(ol, dim=1).gather(1, ab) + 1e-8)
            for _ in range(ppo_epochs):
                l, vs = model({'observation': ob, 'action_mask': mb})
                dist = Categorical(logits=l)
                lp = torch.log(F.softmax(l, dim=1).gather(1, ab) + 1e-8)
                ratio = torch.exp(lp - olp)
                s1 = ratio * dvb; s2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * dvb
                pl = -torch.mean(torch.min(s1, s2))
                vl = torch.mean(F.mse_loss(vs.squeeze(-1), tgb))
                el = -torch.mean(dist.entropy())
                loss = pl + value_coeff * vl + entropy_coeff * el
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item(); n_upd += 1
        avg_loss = total_loss / max(n_upd, 1)

        if (ep+1) % 50 == 0 or ep == 0:
            elapsed = time.time() - t0
            hr = np.mean(hu_rates[-50:]) * 100
            af = np.mean(max_fans[-50:])
            print(f'Ep {ep+1}/{n_episodes} | loss={avg_loss:.4f} | '
                  f'hu_rate={hr:.0f}% | avg_fan={af:.0f} | '
                  f'len={ep_len} | {elapsed:.0f}s', flush=True)

    elapsed = time.time() - t0
    result = {
        'name': name,
        'use_ting': use_ting,
        'episodes': n_episodes,
        'elapsed_sec': elapsed,
        'hu_rates': hu_rates,
        'max_fans': max_fans,
        'lengths': lengths,
        'final_hu_rate_50': float(np.mean(hu_rates[-50:]) * 100),
        'final_avg_fan_50': float(np.mean(max_fans[-50:])),
    }
    print(f'Done. hu_rate={result["final_hu_rate_50"]:.0f}%, '
          f'avg_fan={result["final_avg_fan_50"]:.0f}, '
          f'{elapsed:.0f}s', flush=True)
    return result


if __name__ == '__main__':
    results = {}

    # Experiment 1: RL + ting
    r = run_experiment('RL+ting (7ch)', use_ting=True, n_episodes=N_EPISODES)
    results['rl_ting'] = r

    # Load baseline result for comparison
    baseline_path = 'rl_baseline_result.json'
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            results['rl_baseline'] = json.load(f)

    # Print comparison
    print(f'\n{"="*50}', flush=True)
    print('COMPARISON', flush=True)
    print(f'{"="*50}', flush=True)
    for k, r in results.items():
        hu = r.get('final_hu_rate_50', r.get('final_avg_reward_10', 'N/A'))
        fan = r.get('final_avg_fan_50', 'N/A')
        print(f'{r["name"]}: hu_rate={hu:.0f}% avg_fan={fan:.0f}')
    print(f'\nTing improvement: {results["rl_ting"]["final_hu_rate_50"] - results.get("rl_baseline", {}).get("final_hu_rate_50", 0):+.0f}% hu_rate',
          flush=True)

    # Save
    with open('ting_experiment_result.json', 'w') as f:
        json.dump(results, f, indent=2)
