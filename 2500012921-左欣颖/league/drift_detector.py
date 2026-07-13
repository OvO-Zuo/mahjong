"""
Drift Detection: monitors KL divergence between current policy and SL anchor,
triggering rollback if policy drifts too far (catastrophic forgetting guard).

Key metrics:
- KL(π_current || π_SL): measures how far policy has drifted from anchor
- KL(π_current || π_best):  measures deviation from best-known policy
- Action distribution entropy: monitors exploration collapse
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque


class DriftDetector:
    def __init__(self, kl_threshold=0.5, kl_window=10, rollback_cooldown=50):
        """
        Args:
            kl_threshold:    Rolling-mean KL above which rollback is triggered
            kl_window:       Number of recent KL measurements for rolling mean
            rollback_cooldown: Minimum episodes between rollbacks
        """
        self.kl_threshold = kl_threshold
        self.kl_window = kl_window
        self.rollback_cooldown = rollback_cooldown

        self.kl_history = deque(maxlen=kl_window)
        self.entropy_history = deque(maxlen=kl_window)
        self.rollback_count = 0
        self.last_rollback_ep = -rollback_cooldown
        self.best_state_dict = None
        self.sl_anchor = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def set_sl_anchor(self, state_dict):
        self.sl_anchor = {k: v.detach().cpu().clone() for k, v in state_dict.items()}

    def set_best(self, state_dict):
        self.best_state_dict = {k: v.detach().cpu().clone() for k, v in state_dict.items()}

    def compute_kl(self, current_model, sl_model, observations, action_masks):
        """Compute KL(π_current || π_sl) over a batch of observations.

        Args:
            current_model: Current policy network
            sl_model:      SL anchor network
            observations:  torch.Tensor (batch, 6, 4, 9)
            action_masks:  torch.Tensor (batch, 235)

        Returns:
            Mean KL divergence (scalar float)
        """
        with torch.no_grad():
            # Current policy log-probs
            cur_logits, _ = current_model({
                'observation': observations.float(),
                'action_mask': action_masks.float()
            })
            cur_log_probs = F.log_softmax(cur_logits, dim=-1)

            # SL anchor log-probs (SL model returns logits only, no value)
            if sl_model is not None:
                sl_output = sl_model({
                    'observation': observations.float(),
                    'action_mask': action_masks.float()
                })
                if isinstance(sl_output, tuple):
                    sl_logits = sl_output[0]
                else:
                    sl_logits = sl_output
            else:
                # Fallback: use stored anchor if provided
                return 0.0

            sl_log_probs = F.log_softmax(sl_logits, dim=-1)

            # KL(cur || sl) = sum(cur_prob * (log_cur - log_sl))
            cur_probs = torch.exp(cur_log_probs)
            kl = (cur_probs * (cur_log_probs - sl_log_probs)).sum(dim=-1).mean()

        return kl.item()

    def compute_entropy(self, model, observations, action_masks):
        """Compute mean policy entropy over a batch."""
        with torch.no_grad():
            output = model({
                'observation': observations.float(),
                'action_mask': action_masks.float()
            })
            logits = output[0] if isinstance(output, tuple) else output
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy.item()

    def update(self, current_model, sl_model, observations, action_masks, current_ep):
        """Record new KL measurement and check for drift.

        Returns:
            (should_rollback: bool, kl_value: float, info: dict)
        """
        kl = self.compute_kl(current_model, sl_model, observations, action_masks)
        entropy = self.compute_entropy(current_model, observations, action_masks)

        self.kl_history.append(kl)
        self.entropy_history.append(entropy)

        mean_kl = np.mean(self.kl_history) if self.kl_history else kl
        mean_entropy = np.mean(self.entropy_history) if self.entropy_history else entropy

        should_rollback = False
        info = {
            'kl': kl,
            'mean_kl': mean_kl,
            'entropy': entropy,
            'mean_entropy': mean_entropy,
        }

        if (len(self.kl_history) >= self.kl_window and
            mean_kl > self.kl_threshold and
            current_ep - self.last_rollback_ep >= self.rollback_cooldown):
            should_rollback = True
            self.last_rollback_ep = current_ep
            self.rollback_count += 1
            info['rollback_triggered'] = True
            info['rollback_count'] = self.rollback_count

        return should_rollback, info

    def get_rollback_weights(self):
        """Return best state dict for rollback, or None."""
        return self.best_state_dict

    def is_drift_anomaly(self, kl_value):
        """Quick check: is this KL abnormally high?"""
        if len(self.kl_history) < 3:
            return kl_value > self.kl_threshold * 2
        mean = np.mean(self.kl_history)
        std = np.std(self.kl_history) + 1e-8
        return (kl_value - mean) / std > 3.0

    def get_stats(self):
        return {
            'mean_kl': np.mean(self.kl_history) if self.kl_history else 0.0,
            'mean_entropy': np.mean(self.entropy_history) if self.entropy_history else 0.0,
            'rollback_count': self.rollback_count,
            'last_rollback_ep': self.last_rollback_ep,
        }
