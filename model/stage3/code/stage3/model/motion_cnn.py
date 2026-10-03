from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.body(value))


class MaskedAttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv2d(channels, 1, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(features).flatten(1)
        if mask is not None:
            mask = torch.nn.functional.interpolate(mask.float(), features.shape[-2:], mode="nearest").flatten(1).bool()
            logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=-1)
        return torch.sum(features.flatten(2) * weights[:, None], dim=-1)


class MotionCNN(nn.Module):
    def __init__(self, input_channels: int = 10, widths: tuple[int, ...] = (32, 64, 96, 128), blocks_per_stage: int = 1, pooling: str = "attention"):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, widths[0], 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, widths[0]),
            nn.SiLU(),
        ]
        for stage, width in enumerate(widths):
            if stage:
                layers.extend([
                    nn.Conv2d(widths[stage - 1], width, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(8, width), nn.SiLU(),
                ])
            layers.extend(ResidualBlock(width) for _ in range(blocks_per_stage))
        self.encoder = nn.Sequential(*layers)
        self.pool = MaskedAttentionPool(widths[-1]) if pooling == "attention" else nn.AdaptiveAvgPool2d(1)
        self.pooling = pooling
        self.output_dim = widths[-1]

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = motion.shape
        flat = motion.reshape(batch * time, channels, height, width)
        features = self.encoder(flat)
        if self.pooling == "attention":
            pooled = self.pool(features, flat[:, 7:8] > 0.5)
        else:
            pooled = self.pool(features).flatten(1)
        return pooled.reshape(batch, time, -1)
