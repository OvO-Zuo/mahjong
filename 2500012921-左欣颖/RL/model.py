import torch
from torch import nn


class CNNModel(nn.Module):
    """CNN matching SL/botzone architecture + value head for RL.

    Tower (matches SL model exactly):
      Conv2d(6, 64, 3)  → ReLU
      Conv2d(64, 64, 3) → ReLU
      Conv2d(64, 64, 3) → ReLU  → Flatten
      Linear(64*36, 256) → ReLU
      Linear(256, 235)           ← policy logits (loaded from checkpoint)

    Value branch (added during RL):
      Linear(64*36, 256) → ReLU → Linear(256, 1)
    """

    def __init__(self):
        nn.Module.__init__(self)
        # Tower: conv stack + policy FC layers (matches checkpoint _tower.0..9)
        self._tower = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1, bias=False),   # 0
            nn.ReLU(True),                              # 1
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),   # 2
            nn.ReLU(True),                              # 3
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),   # 4
            nn.ReLU(True),                              # 5
            nn.Flatten(),                               # 6
            nn.Linear(64 * 4 * 9, 256),               # 7
            nn.ReLU(True),                              # 8
            nn.Linear(256, 235),                       # 9
        )
        # Value branch (separate, loaded from checkpoint _value_branch.*)
        self._value_branch = nn.Sequential(
            nn.Linear(64 * 4 * 9, 256),               # 0
            nn.ReLU(True),                              # 1
            nn.Linear(256, 1),                         # 2
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        logits = self._tower(obs)
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask

        # Extract features before logits FC for value branch
        features = self._tower[:7](obs)  # Conv stack + Flatten
        value_hidden = self._value_branch[0](features)
        value_hidden = self._value_branch[1](value_hidden)
        try:
            value = self._value_branch[2](value_hidden)
        except RuntimeError as e:
            if value_hidden.device.type == 'cpu' and 'primitive descriptor' in str(e):
                w = self._value_branch[2].weight
                b = self._value_branch[2].bias
                value = torch.sum(value_hidden * w, dim=1, keepdim=True)
                if b is not None:
                    value = value + b.view(1, 1)
            else:
                raise
        return masked_logits, value
