"""Small reusable neural-network blocks with explicit mask handling."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """Pre-LN 384D Transformer encoder accepting a validity mask."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=384,
            nhead=6,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, nn.LayerNorm(384))

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Mask invalid positions from keys and values before attention."""
        return self.encoder(tokens, src_key_padding_mask=~valid)


def sinusoidal(valid: torch.Tensor) -> torch.Tensor:
    """Fixed 384D encoding of normalized local sequence position.

    The position is index-based only; it never encodes FPS or physical time.
    """
    batch, length = valid.shape
    position = torch.arange(length, device=valid.device, dtype=torch.float32)
    position = position.unsqueeze(0).expand(batch, -1)
    denominator = (valid.sum(dim=1, keepdim=True) - 1).clamp_min(1)
    position = position / denominator

    frequencies = torch.exp(
        torch.arange(0, 384, 2, device=valid.device) * (-math.log(10000.0) / 384)
    )
    encoding = torch.zeros(batch, length, 384, device=valid.device)
    encoding[..., 0::2] = torch.sin(position[..., None] * frequencies)
    encoding[..., 1::2] = torch.cos(position[..., None] * frequencies)
    return encoding


class VideoPool(nn.Module):
    """Learned-query attention pooling over valid coarse temporal states."""

    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(384) / math.sqrt(384))

    def forward(self, states: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        attention_logits = states @ self.query
        attention_logits = attention_logits.masked_fill(~valid, -torch.inf)
        attention = torch.softmax(attention_logits, dim=-1)
        return (attention[..., None] * states).sum(dim=1)


class ResidualTemporalConv(nn.Module):
    """Masked depthwise-separable temporal convolution with residual connection."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            384, 384, kernel_size, padding=kernel_size // 2, groups=384
        )
        self.pointwise = nn.Conv1d(384, 384, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, states: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # Re-mask before and after convolution so temporal padding cannot leak.
        states = states * valid[..., None]
        convolved = self.depthwise(states.transpose(1, 2))
        convolved = self.pointwise(convolved).transpose(1, 2)
        convolved = self.dropout(F.gelu(convolved))
        return (states + convolved) * valid[..., None]
