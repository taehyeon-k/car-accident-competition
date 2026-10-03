from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class OrdinalHead(nn.Module):
    """CORAL-style ordinal head: one latent score and ordered cut points.

    Returns K-1 cumulative logits ``P(y > k)`` for ordered classes 0..K-1.
    """

    def __init__(self, input_dim: int, classes: int = 3):
        super().__init__()
        self.score = nn.Linear(input_dim, 1)
        self.first = nn.Parameter(torch.tensor(-1.0))
        self.gaps = nn.Parameter(torch.zeros(classes - 2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        cuts = torch.cat((self.first[None], self.first + torch.cumsum(F.softplus(self.gaps) + 1e-3, 0)))
        return self.score(value) - cuts


class MotionHeads(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.05, yaw_aux: bool = True,
                 auxiliary: dict | None = None):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.acceleration = nn.Linear(hidden_dim, 2)
        self.speed = nn.Linear(hidden_dim, 1)
        self.stop = nn.Linear(hidden_dim, 1)
        self.steering = nn.Linear(hidden_dim, 1)
        self.yaw = nn.Linear(hidden_dim, 1) if yaw_aux else None
        # Optional Stage 3 v2 auxiliary heads; absent in v1 configs.
        aux = auxiliary or {}
        self.jerk = nn.Linear(hidden_dim, 1) if aux.get("jerk") else None
        self.steer_rate = nn.Linear(hidden_dim, 1) if aux.get("steer_rate") else None
        self.class_mode = aux.get("class_loss", "ce")
        make = (lambda: OrdinalHead(hidden_dim, 3)) if self.class_mode == "ordinal" else (lambda: nn.Linear(hidden_dim, 3))
        self.accel_class = make() if aux.get("ordinal_accel") else None
        self.steer_class = make() if aux.get("ordinal_steer") else None

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(value)
        output = {
            "acceleration": self.acceleration(shared),
            "speed": F.softplus(self.speed(shared).squeeze(-1)),
            "stop_logit": self.stop(shared).squeeze(-1),
            "steering_angle": self.steering(shared).squeeze(-1),
        }
        if self.yaw is not None:
            output["yaw_rate_aux"] = self.yaw(shared).squeeze(-1)
        if self.jerk is not None:
            output["jerk"] = self.jerk(shared).squeeze(-1)
        if self.steer_rate is not None:
            output["steer_rate"] = self.steer_rate(shared).squeeze(-1)
        if self.accel_class is not None:
            output["accel_class"] = self.accel_class(shared)
        if self.steer_class is not None:
            output["steer_class"] = self.steer_class(shared)
        return output
