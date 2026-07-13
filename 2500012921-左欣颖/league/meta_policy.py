"""
Meta Policy Selector.

Chooses which sub-policy to use based on game state features:
- Early game (high wall): conservative, build hand
- Mid game: aggressive, aim for ready hand
- Late game (low wall): defensive, avoid dealing into wins
- Seat-specific: adjust strategy based on seat wind
- Score-aware: lead vs catch-up behavior
"""

import numpy as np


class MetaPolicySelector:
    def __init__(self):
        # Phase thresholds (fraction of wall remaining)
        self.early_threshold = 0.5    # >50% wall = early game
        self.late_threshold = 0.15     # <15% wall = late game

        # Strategy modes
        self.STRATEGIES = ['conservative', 'aggressive', 'defensive', 'balanced']

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def select_strategy(self, state_info):
        """Select strategy based on current game state.

        Args:
            state_info: dict with keys:
                - wall_remaining: float (fraction, 0-1)
                - hand_ready: bool (is hand one tile from win?)
                - seat_wind: int (0-3)
                - prevalent_wind: int (0-3)
                - opponent_hu_risk: float (heuristic for opponent win risk, 0-1)
                - score_lead: float (positive = leading, negative = trailing)

        Returns:
            str: strategy name
        """
        wall = state_info.get('wall_remaining', 0.5)
        ready = state_info.get('hand_ready', False)
        risk = state_info.get('opponent_hu_risk', 0.3)
        lead = state_info.get('score_lead', 0.0)

        # Late game → defensive if not ready
        if wall < self.late_threshold:
            if ready:
                return 'aggressive'
            return 'defensive'

        # High opponent risk → defensive
        if risk > 0.7 and not ready:
            return 'defensive'

        # Early game with big lead → balanced
        if wall > self.early_threshold and lead > 10:
            return 'balanced'

        # Mid-game, hand ready → aggressive
        if ready and wall < self.early_threshold:
            return 'aggressive'

        # Early game → conservative (build hand)
        if wall > self.early_threshold:
            return 'conservative'

        # Default
        return 'balanced'

    def get_temperature(self, strategy):
        """Get action sampling temperature for each strategy.

        Returns:
            float: temperature (1.0 = standard, <1.0 = more greedy, >1.0 = more random)
        """
        temps = {
            'conservative': 1.2,   # More exploration to build hand
            'aggressive':    0.5,   # More greedy to win
            'defensive':     0.8,   # Moderate: avoid risky moves
            'balanced':      1.0,   # Standard
        }
        return temps.get(strategy, 1.0)

    def get_action_bias(self, strategy, valid_actions, feature_agent):
        """Adjust action logits based on strategy (optional biasing).

        Returns:
            np.ndarray of shape (235,) with bias values (0 = no bias)
        """
        bias = np.zeros(235)

        if strategy == 'aggressive':
            # Slightly prefer Hu actions
            bias[1] = 0.2  # Hu bias

        elif strategy == 'defensive':
            # Prefer safer discards (tiles not recently discarded by others)
            # This is a simple heuristic; full implementation would be more nuanced
            pass

        elif strategy == 'conservative':
            # Slight preference for drawing (Pass) over risky calls
            pass

        return bias

    def should_force_argmax(self, strategy):
        """Whether to use deterministic action selection for this strategy."""
        return strategy == 'aggressive'
