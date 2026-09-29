"""RGB-only TCP feature student with the unchanged Stage 3 temporal block and heads."""
from __future__ import annotations

import torch
from torch import nn

from stage3.model.heads import MotionHeads
from stage3.model.model import _temporal


class TCPFeatureStudent(nn.Module):
    def __init__(self, cfg: dict, fine_tune_projection: bool = False,
                 projection_state: dict | None = None):
        super().__init__()
        dim = cfg['tcn']['dim']
        self.fine_tune_projection = fine_tune_projection
        if fine_tune_projection:
            self.tcp_projection = nn.Linear(512, 1000)
            if projection_state is None:
                raise ValueError('Pretrained TCP projection tensors required')
            self.tcp_projection.load_state_dict(projection_state, strict=True)
        self.visual_adapter = nn.Sequential(nn.LayerNorm(1000), nn.Linear(1000, dim), nn.SiLU())
        self.temporal = _temporal(cfg, dim)
        self.heads = MotionHeads(dim, 64, cfg.get('head_dropout', 0.05), cfg.get('yaw_aux', True))

    def forward(self, motion: torch.Tensor, physics: torch.Tensor,
                lengths: torch.Tensor | None = None, chunk_frames: int | None = None,
                visual: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if visual is None or visual.shape[-2:] != (1, 1512):
            raise ValueError('Expected cached TCP RGB-only features [B,T,1,1512]')
        source = visual.squeeze(-2).float()
        driving = self.tcp_projection(source[..., :512]) if self.fine_tune_projection else source[..., 512:]
        encoded = self.temporal(self.visual_adapter(driving), lengths)
        return self.heads(encoded)
