"""
League Manager — orchestrates the league-based self-play system.

Responsibilities:
1. Manage model pool (SL, RL, best, historical, exploiters, champion)
2. Schedule matches between different model tiers
3. Track Elo rankings
4. Trigger periodic evaluation and best-model updates
5. Coordinate drift detection and rollback
6. Manage exploit detection and exploiter training
"""

import os
import sys
import time
import json
import torch
import numpy as np

# Add parent RL directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RL'))

from model import CNNModel
from feature import FeatureAgent
from env import MahjongGBEnv

from elo import EloRanker
from league_model_pool import LeagueModelPool
from dynamic_sampler import DynamicSampler
from drift_detector import DriftDetector
from exploit_detector import ExploitDetector
from meta_policy import MetaPolicySelector


class LeagueManager:
    def __init__(self, config, league_pool, elo_ranker, sl_model_path=None):
        """
        Args:
            config: dict with league hyperparameters
            league_pool: LeagueModelPool instance
            elo_ranker: EloRanker instance
            sl_model_path: path to SL pretrained model for anchor
        """
        self.config = config
        self.pool = league_pool
        self.elo = elo_ranker

        # Components
        self.sampler = DynamicSampler(self.elo)
        self.drift = DriftDetector(
            kl_threshold=config.get('kl_threshold', 0.5),
            kl_window=config.get('kl_window', 10),
            rollback_cooldown=config.get('rollback_cooldown', 50),
        )
        self.exploit = ExploitDetector(
            window_size=config.get('exploit_window', 200),
        )
        self.meta = MetaPolicySelector()

        # Device (must be set before loading models)
        requested = config.get('device', 'cpu')
        if requested == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Load SL anchor
        self.sl_model = None
        if sl_model_path and os.path.exists(sl_model_path):
            self.sl_model = self._load_sl_model(sl_model_path)
            sl_sd = self.sl_model.state_dict()
            self.pool.add_sl(sl_sd, {'path': sl_model_path})
            self.drift.set_sl_anchor(sl_sd)
            self.elo.register('sl_anchor', 1400.0)  # Fixed base Elo

        # State
        self.current_ep = 0
        self.best_eval_hu = 0.0
        self.best_eval_ep = 0
        self.total_games = 0
        self.league_log = []

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def initialize(self, initial_rl_state_dict):
        """Initialize the league with the starting RL model."""
        model_id = self.pool.add_rl_active(
            initial_rl_state_dict, {'ep': 0}
        )
        self.elo.register(model_id, 1500.0)
        self.pool.set_champion(initial_rl_state_dict, {'ep': 0})
        self.drift.set_best(initial_rl_state_dict)
        return model_id

    def select_opponents(self, current_model_id, n_opponents=3):
        """Select opponents for a self-play match using dynamic sampling."""
        opponent_ids = self.pool.get_opponent_ids()
        if not opponent_ids:
            return []

        # Add the champion if not already in pool
        champion = self.pool.get_champion()
        if champion and champion['model_id'] not in opponent_ids:
            opponent_ids.append(champion['model_id'])

        # Sample opponents
        selected = self.sampler.sample_opponent_group(
            current_model_id, opponent_ids, n=n_opponents
        )

        # Ensure we have exactly n_opponents (fill with random if needed)
        while len(selected) < n_opponents and opponent_ids:
            remaining = [m for m in opponent_ids if m not in selected]
            if not remaining:
                break
            selected.append(remaining[0])

        return selected[:n_opponents]

    def record_match_result(self, winner_id, loser_ids, scores=None):
        """Update Elo rankings after a match.

        Args:
            winner_id: model_id of the winner
            loser_ids: list of model_ids that lost
            scores: optional list of scores (fan points)
        """
        for lid in loser_ids:
            self.elo.update(winner_id, lid, score=1.0)
        self.total_games += 1

    def record_episode(self, result):
        """Record one episode for exploit analysis."""
        self.exploit.record_episode(result)

    def check_drift(self, current_model, current_ep):
        """Check if policy has drifted too far from anchor.

        Returns:
            (should_rollback, info)
        """
        if self.sl_model is None:
            return False, {'kl': 0.0, 'mean_kl': 0.0}

        # Sample observations from recent play (use random for drift check)
        batch_obs = torch.randn(16, 6, 4, 9)
        batch_mask = torch.ones(16, 235)

        return self.drift.update(
            current_model, self.sl_model, batch_obs, batch_mask, current_ep
        )

    def maybe_rollback(self, current_model):
        """Perform rollback if drift detected."""
        best_sd = self.drift.get_rollback_weights()
        if best_sd is not None:
            current_model.load_state_dict(best_sd)
            return True
        return False

    def update_best(self, state_dict, hu_rate, ep):
        """Update best model if current eval beats previous best."""
        if hu_rate > self.best_eval_hu:
            self.best_eval_hu = hu_rate
            self.best_eval_ep = ep
            self.pool.add_best(state_dict, {
                'ep': ep, 'hu_rate': hu_rate
            })
            self.drift.set_best(state_dict)
            return True
        return False

    def update_champion(self):
        """Update champion: model with highest Elo."""
        rankings = self.elo.get_rankings()
        if not rankings:
            return None

        top_id, top_elo = rankings[0]
        # Find the state dict for the top model
        champion = self.pool.get_champion()
        if champion is None or champion['model_id'] != top_id:
            # The top Elo model becomes champion
            # We need to find its state dict from the pool
            for cat_entries in [
                self.pool._models['rl_active'],
                self.pool._models['best'],
                self.pool._models['historical'],
            ]:
                for entry in cat_entries:
                    if entry['model_id'] == top_id:
                        self.pool.set_champion(
                            entry['state_dict'],
                            {'elo': top_elo, **entry['metadata']}
                        )
                        return top_id
        return None

    def should_snapshot_historical(self, ep):
        """Determine if we should save a historical snapshot."""
        interval = self.config.get('historical_snapshot_interval', 100)
        return ep % interval == 0 and ep > 0

    def should_evaluate(self, ep):
        """Determine if we should run league evaluation."""
        interval = self.config.get('eval_interval', 200)
        return ep % interval == 0 and ep > 0

    def should_train_exploiter(self):
        """Determine if an exploiter should be trained."""
        return self.exploit.should_train_exploiter()

    def get_league_stats(self):
        """Return current league statistics."""
        rankings = self.elo.get_rankings()
        return {
            'total_games': self.total_games,
            'total_episodes': self.current_ep,
            'best_eval_hu': self.best_eval_hu,
            'best_eval_ep': self.best_eval_ep,
            'elo_rankings': rankings[:10] if rankings else [],
            'drift_stats': self.drift.get_stats(),
            'exploit_report': self.exploit.get_weakness_report(),
            'pool_counts': {
                cat: len(entries) for cat, entries in self.pool._models.items()
            },
        }

    def save_state(self, path):
        """Save league state to disk."""
        state = {
            'current_ep': self.current_ep,
            'best_eval_hu': self.best_eval_hu,
            'best_eval_ep': self.best_eval_ep,
            'total_games': self.total_games,
            'league_log': self.league_log[-1000:],  # Keep last 1000 entries
        }
        with open(path, 'w') as f:
            json.dump(state, f)
        self.pool.save_to_disk()
        elo_path = path.replace('.json', '_elo.json')
        self.elo.save(elo_path)

    def load_state(self, path):
        """Load league state from disk."""
        with open(path, 'r') as f:
            state = json.load(f)
        self.current_ep = state.get('current_ep', 0)
        self.best_eval_hu = state.get('best_eval_hu', 0.0)
        self.best_eval_ep = state.get('best_eval_ep', 0)
        self.total_games = state.get('total_games', 0)
        self.league_log = state.get('league_log', [])

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #
    def _load_sl_model(self, path):
        model = CNNModel()
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model', ckpt)
        sd = {k: v for k, v in sd.items() if not k.startswith('_value_branch')}
        model.load_state_dict(sd, strict=False)  # SL has no value branch
        model.eval()
        if self.device == 'cuda':
            model = model.cuda()
        return model
