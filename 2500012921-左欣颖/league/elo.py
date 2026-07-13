"""
League Elo Rating System.

Standard Elo with K-factor scheduling, intended for tracking strengths of
different model versions across self-play matchups.
"""

import math


class EloRanker:
    def __init__(self, initial_elo=1500.0, k_factor=32.0, min_k=8.0,
                 decay_per_game=0.97):
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.min_k = min_k
        self.decay_per_game = decay_per_game
        self.ratings = {}       # model_id -> current Elo
        self.game_counts = {}   # model_id -> total games played

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def register(self, model_id, elo=None):
        if model_id not in self.ratings:
            self.ratings[model_id] = (
                elo if elo is not None else self.initial_elo
            )
            self.game_counts[model_id] = 0

    def remove(self, model_id):
        self.ratings.pop(model_id, None)
        self.game_counts.pop(model_id, None)

    def update(self, winner_id, loser_id, score=1.0):
        """Update Elo after a match.  `score=1.0` = winner won, 0.5 = draw."""
        self.register(winner_id)
        self.register(loser_id)

        rw = self.ratings[winner_id]
        rl = self.ratings[loser_id]

        ew = 1.0 / (1.0 + 10.0 ** ((rl - rw) / 400.0))
        el = 1.0 - ew

        kw = self._effective_k(winner_id)
        kl = self._effective_k(loser_id)

        self.ratings[winner_id] = rw + kw * (score - ew)
        self.ratings[loser_id]  = rl + kl * ((1.0 - score) - el)

        self.game_counts[winner_id] += 1
        self.game_counts[loser_id]  += 1

    def update_batch(self, results):
        """results: list of (winner_id, loser_id, score) tuples."""
        for w, l, s in results:
            self.update(w, l, s)

    def get_elo(self, model_id):
        self.register(model_id)
        return self.ratings.get(model_id, self.initial_elo)

    def get_rankings(self):
        """Return list of (model_id, elo) sorted descending by Elo."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def get_top_n(self, n):
        return self.get_rankings()[:n]

    def get_strength_distribution(self, model_ids):
        """Return probability distribution based on Elo for opponent sampling."""
        if not model_ids:
            return {}
        elos = [self.get_elo(m) for m in model_ids]
        # Softmax over Elo (temperature = 1/400 gives standard Elo win-prob scale)
        max_elo = max(elos)
        exps = [math.exp((e - max_elo) / 400.0) for e in elos]
        total = sum(exps)
        return {m: exps[i] / total for i, m in enumerate(model_ids)}

    def save(self, path):
        import json
        data = {
            'ratings': self.ratings,
            'game_counts': self.game_counts,
            'initial_elo': self.initial_elo,
            'k_factor': self.k_factor,
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.ratings = data['ratings']
        self.game_counts = data.get('game_counts', {})
        self.initial_elo = data.get('initial_elo', 1500.0)
        self.k_factor = data.get('k_factor', 32.0)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #
    def _effective_k(self, model_id):
        games = self.game_counts.get(model_id, 0)
        # K decays with number of games
        k = self.k_factor * (self.decay_per_game ** games)
        return max(k, self.min_k)


def compute_expected_score(elo_a, elo_b):
    """Probability that A beats B."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
