"""CNNModel with variable input channels — SL-compatible architecture."""
import torch
from torch import nn


class CNNModelVar(nn.Module):
    """CNN matching SL architecture + value head for RL.

    Tower structure (matches SL model.py exactly):
      Conv2d(in, 64, 3) → ReLU
      Conv2d(64, 64, 3) → ReLU
      Conv2d(64, 64, 3) → ReLU → Flatten
      Linear(64*36, 256) → ReLU
      Linear(256, 235)           ← policy logits (loaded from SL)
    Value branch (added for RL):
      Linear(64*36, 256) → ReLU → Linear(256, 1)
    """

    def __init__(self, in_channels=6):
        nn.Module.__init__(self)
        self.in_channels = in_channels

        # Tower (matches SL model.py)
        self._tower = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 235),
        )

        # Value branch (new, learned during RL)
        self._value_branch = nn.Sequential(
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        # Tower forward: Conv→ReLU→Conv→ReLU→Conv→ReLU→Flatten→FC→ReLU→FC
        flat = self._tower[0:7](obs)  # Conv layers + Flatten → (N, 64*36)
        fc1 = self._tower[7](flat)      # Linear(64*36, 256)
        fc1_relu = self._tower[8](fc1)   # ReLU
        logits = self._tower[9](fc1_relu)  # Linear(256, 235)

        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask

        # Value branch (uses same flattened features)
        vh = self._value_branch[0](flat)
        vh = self._value_branch[1](vh)
        try:
            value = self._value_branch[2](vh)
        except RuntimeError:
            w = self._value_branch[2].weight
            b = self._value_branch[2].bias
            value = torch.sum(vh * w, dim=1, keepdim=True)
            if b is not None:
                value = value + b.view(1, 1)
        return masked_logits, value

    def load_sl_tower(self, sl_state_dict, target_channels):
        """Load tower weights from SL model, expanding first conv if needed."""
        own_state = self.state_dict()

        for k, v in sl_state_dict.items():
            if k.startswith('_tower.'):
                own_k = k
                if own_k not in own_state:
                    continue
                if k == '_tower.0.weight' and target_channels > v.shape[1]:
                    # Expand first conv: copy first 6 channels, small-noise init rest
                    expanded = torch.zeros(
                        v.shape[0], target_channels, *v.shape[2:])
                    expanded[:, :6] = v
                    for c in range(6, target_channels):
                        expanded[:, c] = v[:, 0] * \
                            0.5 + torch.randn_like(v[:, 0]) * 0.01
                    own_state[own_k] = expanded
                else:
                    own_state[own_k] = v.clone()

        self.load_state_dict(own_state)
