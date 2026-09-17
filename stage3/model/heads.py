from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MotionHeads(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.05, yaw_aux: bool = True):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.acceleration = nn.Linear(hidden_dim, 2)
        self.speed = nn.Linear(hidden_dim, 1)
        self.stop = nn.Linear(hidden_dim, 1)
        self.steering = nn.Linear(hidden_dim, 1)
        self.yaw = nn.Linear(hidden_dim, 1) if yaw_aux else None

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
        return output
