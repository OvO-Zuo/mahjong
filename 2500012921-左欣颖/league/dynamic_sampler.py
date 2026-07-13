"""
Dynamic opponent sampling based on strength distribution.

Uses Elo-based softmax to sample opponents, ensuring:
- Stronger opponents appear more often (quality training)
- Weaker opponents still appear occasionally (diversity)
- Temperature can be adjusted to control concentration
"""

import random
import math


class DynamicSampler:
    def __init__(self, elo_ranker, temperature=400.0, uniform_mix=0.1):
        """
        Args:
            elo_ranker: EloRanker instance
            temperature: Higher = more uniform; 400 = standard Elo scale
            uniform_mix: Fraction of time to sample uniformly (exploration)
        """
        self.elo = elo_ranker
        self.temperature = temperature
        self.uniform_mix = uniform_mix

    def sample_opponent(self, current_model_id, candidate_ids):
        """Sample one opponent ID from the candidate pool.

        Excludes current_model_id from candidates.
        """
        candidates = [m for m in candidate_ids if m != current_model_id]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        if random.random() < self.uniform_mix:
            return random.choice(candidates)

        # Softmax over Elo ratings
        elos = [self.elo.get_elo(m) for m in candidates]
        max_elo = max(elos)
        weights = [math.exp((e - max_elo) * 400.0 / self.temperature) for e in elos]
        total = sum(weights)

        r = random.random() * total
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return candidates[i]

        return candidates[-1]

    def sample_opponent_group(self, current_model_id, candidate_ids, n=3):
        """Sample `n` opponents without repetition."""
        selected = []
        pool = list(candidate_ids)
        for _ in range(n):
            opp = self.sample_opponent(current_model_id, pool)
            if opp is None:
                break
            selected.append(opp)
            pool.remove(opp)
        return selected

    def sample_by_tier(self, current_model_id, candidate_ids, tier_weights=None):
        """Sample with tier-based probability distribution.

        Tier weights: dict mapping tier_name -> probability mass.
        Candidates are sorted by Elo into tiers.
        """
        if tier_weights is None:
            tier_weights = {'top': 0.5, 'mid': 0.3, 'bottom': 0.2}

        if not candidate_ids:
            return None

        # Sort by Elo
        ranked = sorted(candidate_ids, key=lambda m: self.elo.get_elo(m), reverse=True)
        n = len(ranked)
        if n == 0:
            return None

        tier_size = max(1, n // 3)
        tiers = {
            'top':    ranked[:tier_size],
            'mid':    ranked[tier_size:2*tier_size],
            'bottom': ranked[2*tier_size:],
        }

        # Sample tier by weight, then uniformly within tier
        tier_r = random.random()
        cumulative = 0.0
        for tier_name, weight in tier_weights.items():
            cumulative += weight
            if tier_r <= cumulative and tiers[tier_name]:
                return random.choice(tiers[tier_name])

        return random.choice(ranked)
