from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TCNBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, dropout: float, causal: bool = False, kernel_size: int = 3, padding_mode: str = "replicate"):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size, dilation=dilation)
        self.pointwise = nn.Conv1d(dim, dim, 1)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.dilation = dilation
        self.causal = causal
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        encoded = self.norm(value).transpose(1, 2)
        span = (self.kernel_size - 1) * self.dilation
        padding = (span, 0) if self.causal else (span // 2, span - span // 2)
        encoded = F.pad(encoded, padding, mode=self.padding_mode)
        encoded = self.dropout(self.activation(self.conv(encoded)))
        encoded = self.dropout(self.pointwise(encoded)).transpose(1, 2)
        return value + encoded


class TemporalConvNet(nn.Module):
    def __init__(self, dim: int = 128, dilations: tuple[int, ...] = (1, 2, 4, 8, 16), dropout: float = 0.1, causal: bool = False, kernel_size: int = 3, padding_mode: str = "replicate"):
        super().__init__()
        self.blocks = nn.Sequential(*(TCNBlock(dim, d, dropout, causal, kernel_size, padding_mode) for d in dilations))
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.blocks(value)
