from __future__ import annotations

import torch
from torch import nn

from .heads import MotionHeads
from .motion_cnn import MotionCNN
from .physics_mlp import PhysicsMLP
from .tcn import TemporalConvNet


class Stage3MotionModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        cnn = cfg["motion_cnn"]
        temporal = cfg["tcn"]
        self.motion_cnn = MotionCNN(
            10, tuple(cnn["widths"]), cnn.get("blocks_per_stage", 1), cnn.get("pooling", "attention")
        )
        self.physics_mlp = PhysicsMLP(20, 32, cfg.get("physics_dropout", 0.05))
        self.fusion = nn.Sequential(nn.Linear(self.motion_cnn.output_dim + 32, temporal["dim"]), nn.LayerNorm(temporal["dim"]))
        self.temporal = TemporalConvNet(
            temporal["dim"], tuple(temporal["dilations"]), temporal["dropout"], temporal.get("causal", False),
            temporal.get("kernel_size", 3), temporal.get("padding_mode", "replicate")
        )
        self.heads = MotionHeads(temporal["dim"], 64, cfg.get("head_dropout", 0.05), cfg.get("yaw_aux", True))

    def forward(self, motion: torch.Tensor, physics: torch.Tensor) -> dict[str, torch.Tensor]:
        spatial = self.motion_cnn(motion)
        physical = self.physics_mlp(physics)
        encoded = self.temporal(self.fusion(torch.cat((spatial, physical), dim=-1)))
        return self.heads(encoded)
