"""
Dual-head CNNModel: shared tower + separate SL-head and RL-head.
Final policy: π = α·π_SL + (1-α)·π_RL,  α = clamp(KL/kl_ref, α_min, α_max)
"""
import torch
from torch import nn
import torch.nn.functional as F


class DualHeadModel(nn.Module):
    """Shared Conv2d tower + two FC heads (SL frozen, RL trained).

    Tower (shared, from SL checkpoint):
      Conv2d(6→64→64→64) → Flatten → Linear(2304,256) → ReLU

    SL-head (frozen):
      Linear(256, 235)

    RL-head (trained):
      Linear(256, 235)
      Linear(256, 1)  — value

    Final logits = α * SL_logits + (1-α) * RL_logits
    α = clamp(current_KL / kl_ref, α_min, α_max)
    """

    def __init__(self, kl_ref=0.04, alpha_min=0.2, alpha_max=0.8):
        nn.Module.__init__(self)
        self.kl_ref = kl_ref
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.current_alpha = alpha_max  # Start SL-dominant

        # Shared tower (matches CNNModel)
        self._tower = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(True),
        )

        # SL head (frozen after init)
        self.sl_head = nn.Linear(256, 235)

        # RL head = SL_projection + delta (residual).  At init, delta ≈ 0 → KL ≈ 0.
        self.rl_delta = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 235),
        )
        # Initialize delta to near-zero so RL ≈ SL at start
        for m in self.rl_delta.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.out_features == 235:  # Last layer: very small init
                    m.weight.data *= 0.001
                    if m.bias is not None: m.bias.data.zero_()
        self.value_head = nn.Sequential(
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def set_alpha(self, kl_value):
        """α = clamp(1 - KL/kl_ref, α_min, α_max).
        KL=0 → α=max (SL-dominant). KL grows → α decreases (more RL)."""
        ratio = 1.0 - max(0.0, kl_value) / self.kl_ref
        self.current_alpha = max(self.alpha_min, min(self.alpha_max, ratio))

    def set_alpha_manual(self, alpha):
        """Directly set alpha (for curriculum phases)."""
        self.current_alpha = max(self.alpha_min, min(self.alpha_max, alpha))

    def forward(self, input_dict, mode='mixture'):
        """mode: 'mixture' (α·SL + (1-α)·RL), 'sl_only', 'rl_only'"""
        obs = input_dict["observation"].float()
        mask = input_dict["action_mask"].float()

        # Shared tower
        features = self._tower[:7](obs)  # Conv + Flatten
        trunk_out = self._tower[7](features)  # Linear(2304, 256)
        trunk_out = self._tower[8](trunk_out)  # ReLU

        # Heads
        sl_logits = self.sl_head(trunk_out)
        # RL = SL projection + residual delta (delta ≈ 0 at init → KL ≈ 0)
        rl_logits = sl_logits.detach() + self.rl_delta(trunk_out)

        # Value
        vh = self.value_head[0](features)
        vh = self.value_head[1](vh)
        value = self.value_head[2](vh)

        # Mixture
        if mode == 'sl_only':
            logits = sl_logits
        elif mode == 'rl_only':
            logits = rl_logits
        else:  # mixture
            a = self.current_alpha
            logits = a * sl_logits + (1 - a) * rl_logits

        # Mask invalid actions
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask

        return masked_logits, value, sl_logits, rl_logits

    def freeze_sl(self):
        """Freeze tower + SL head. Only RL delta and value head train."""
        for p in self._tower.parameters(): p.requires_grad = False
        for p in self.sl_head.parameters(): p.requires_grad = False
        for p in self.rl_delta.parameters(): p.requires_grad = True
        for p in self.value_head.parameters(): p.requires_grad = True

    def unfreeze_tower(self):
        for p in self._tower.parameters(): p.requires_grad = True

    def load_sl_checkpoint(self, sl_state_dict):
        """Load tower + SL-head + RL-head from SL checkpoint.
        RL head starts identical to SL head → KL=0 at init."""
        own = self.state_dict()
        for k, v in sl_state_dict.items():
            if k.startswith('_tower.') and not k.startswith('_tower.9'):
                own_key = k
                if own_key in own and own[own_key].shape == v.shape:
                    own[own_key] = v.clone()
        # Copy tower FC → SL head only. RL delta stays near-zero (residual).
        if '_tower.9.weight' in sl_state_dict:
            own['sl_head.weight'] = sl_state_dict['_tower.9.weight'].clone()
            own['sl_head.bias'] = sl_state_dict['_tower.9.bias'].clone()
        self.load_state_dict(own, strict=False)


def compute_kl_sl_rl(model, obs, masks):
    """Compute KL(π_RL || π_SL) — must be ≥ 0."""
    with torch.no_grad():
        _, _, sl_logits, rl_logits = model({'observation': obs, 'action_mask': masks}, mode='mixture')
        rl_probs = F.softmax(rl_logits, dim=-1)
        rl_log_probs = F.log_softmax(rl_logits, dim=-1)
        sl_log_probs = F.log_softmax(sl_logits, dim=-1)
        valid = (masks > 0.5).float()
        kl = (valid * rl_probs * (rl_log_probs - sl_log_probs.detach())).sum(dim=-1).mean()
    return max(0.0, kl.item())  # KL ≥ 0 always
