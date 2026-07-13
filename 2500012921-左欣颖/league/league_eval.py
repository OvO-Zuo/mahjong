"""
League Evaluation — runs periodic evaluation tournaments.

Evaluates all league models against each other to:
1. Measure true strength via head-to-head matches
2. Update Elo rankings from real match results
3. Select best model for promotion
4. Detect which models are obsolete (can be pruned)
"""

import os
import sys
import json
import time
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RL'))

from model import CNNModel
from feature import FeatureAgent
from env import MahjongGBEnv


class LeagueEvaluator:
    def __init__(self, league_manager, config):
        """
        Args:
            league_manager: LeagueManager instance
            config: eval-specific config
                - eval_games: total games per matchup (default 100)
                - eval_seats: list of seat positions for the evaluated model
                - device: 'cpu' or 'cuda'
        """
        self.league = league_manager
        self.config = config
        self.device = config.get('device', 'cpu')
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'

    def evaluate_all(self, current_model, current_ep):
        """Run full league evaluation.

        Returns:
            dict with evaluation results
        """
        results = {
            'ep': current_ep,
            'timestamp': time.time(),
            'matchups': {},
            'summary': {},
        }

        # Get all league models
        opponents = self.league.pool.get_all_opponents()
        if not opponents:
            return results

        # Evaluate current model vs each opponent category
        for entry in opponents:
            cat = entry['category']
            opp_id = entry['model_id']

            matchup_key = f"rl_active_vs_{cat}_{opp_id}"
            matchup_results = self._evaluate_matchup(
                current_model, entry['state_dict'], current_ep
            )
            results['matchups'][matchup_key] = matchup_results

            # Update Elo if current model won more than half
            if matchup_results['hu_rate'] > 0.5:
                self.league.elo.update(
                    'rl_active', opp_id,
                    score=matchup_results['hu_rate']
                )

        # Summary statistics
        all_hu_rates = [m['hu_rate'] for m in results['matchups'].values()]
        results['summary'] = {
            'avg_hu_rate': np.mean(all_hu_rates) if all_hu_rates else 0.0,
            'max_hu_rate': max(all_hu_rates) if all_hu_rates else 0.0,
            'min_hu_rate': min(all_hu_rates) if all_hu_rates else 0.0,
            'num_matchups': len(all_hu_rates),
            'elo_rank': self.league.elo.get_elo('rl_active'),
        }

        # Update best model if current evaluation beats best
        avg_hu = results['summary']['avg_hu_rate']
        self.league.update_best(
            self._get_model_state(current_model),
            avg_hu, current_ep
        )

        return results

    def _evaluate_matchup(self, current_model, opponent_sd, current_ep, n_games=None):
        """Evaluate current model against a specific opponent.

        Returns:
            dict with hu_rate, avg_fan, avg_turns
        """
        n_games = n_games or self.config.get('eval_games_per_matchup', 30)

        # Load opponent model - always on CPU for evaluation (avoids CUDA OOM)
        opp_model = CNNModel()
        opp_model.load_state_dict(opponent_sd, strict=False)
        opp_model.eval()
        # Keep on CPU for evaluation simplicity
        current_model_cpu = CNNModel()
        current_sd = {
            k: v.cpu().clone() if v.is_cuda else v.clone()
            for k, v in current_model.state_dict().items()
        }
        current_model_cpu.load_state_dict(current_sd, strict=False)
        current_model_cpu.eval()

        if self.device == 'cuda':
            opp_model = opp_model.cuda()
            current_model_cpu = current_model_cpu.cuda()

        hu_count = 0
        total_fan = 0
        total_turns = 0

        for g in range(n_games):
            result = self._play_eval_game(current_model_cpu, opp_model, g)
            if result['hu']:
                hu_count += 1
                total_fan += result['fan']
            total_turns += result['turns']

        return {
            'games': n_games,
            'hu_rate': hu_count / n_games,
            'avg_fan': total_fan / max(hu_count, 1),
            'avg_turns': total_turns / n_games,
        }

    def _play_eval_game(self, current_model, opp_model, seed):
        """Play a single evaluation game.

        The current model plays seat 0; opponents fill seats 1-3.
        Uses deterministic argmax for evaluation.
        """
        config = {
            'agent_clz': FeatureAgent,
            'duplicate': True,
            'variety': 10000,
        }
        env = MahjongGBEnv(config)
        obs_dict = env.reset()

        # Models for each seat: seat 0 = current, seats 1-3 = opponent
        models = [current_model, opp_model, opp_model, opp_model]
        agents = [FeatureAgent(i) for i in range(4)]

        # Process initial winds
        for i in range(4):
            agents[i].request2obs('Wind %d' % (i % 4))

        done = False
        turns = 0
        max_turns = 500

        while not done and turns < max_turns:
            actions = {}
            for name in env.agent_names:
                i = int(name.split('_')[1]) - 1
                obs = obs_dict.get(name)
                if obs is not None:
                    action = self._get_action(obs, models[i], agents[i])
                    actions[name] = action

            if not actions:
                break

            obs_dict, reward_dict, done_dict = env.step(actions)
            done = done_dict
            turns += 1

        # Determine if seat 0 (current model) won
        seat0_reward = reward_dict.get('player_1', 0) if 'reward_dict' in dir() else 0
        hu = seat0_reward > 0
        fan = max(0, seat0_reward // 3 - 8) if hu else 0

        return {'hu': hu, 'fan': fan, 'turns': turns}

    def _get_action(self, obs, model, agent):
        """Get argmax action from model."""
        obs_t = torch.from_numpy(np.expand_dims(obs['observation'], 0))
        mask_t = torch.from_numpy(np.expand_dims(obs['action_mask'], 0))
        if self.device == 'cuda':
            obs_t = obs_t.cuda()
            mask_t = mask_t.cuda()
        with torch.no_grad():
            # RL model uses input_dict["observation"] directly
            logits, _ = model({
                'observation': obs_t.float(),
                'action_mask': mask_t.float()
            })
        return int(logits.cpu().numpy().flatten().argmax())

    def _get_model_state(self, model):
        return {k: v.cpu().clone() for k, v in model.state_dict().items()}
