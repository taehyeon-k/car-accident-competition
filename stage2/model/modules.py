"""Small reusable neural-network blocks with explicit mask handling."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """Pre-LN Transformer encoder accepting a validity mask."""

    def __init__(
        self,
        num_layers: int,
        d_model: int = 256,
        heads: int = 4,
        ffn_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("Transformer d_model must be divisible by the head count")

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers,
            nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        # TransformerEncoder clones the supplied layer, including its initial
        # values. Initialize each clone independently to break that symmetry.
        for encoder_layer in self.encoder.layers:
            nn.init.xavier_uniform_(encoder_layer.self_attn.in_proj_weight)
            nn.init.zeros_(encoder_layer.self_attn.in_proj_bias)
            encoder_layer.self_attn.out_proj.reset_parameters()
            encoder_layer.linear1.reset_parameters()
            encoder_layer.linear2.reset_parameters()

    def forward(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Mask invalid positions from keys and values before attention."""
        # Skip entirely padded sequences: attention with no valid key can produce
        # NaNs, and executing those rows serves no purpose. Clear invalid inputs
        # before attention as NaN * 0 is still NaN.
        active_rows = valid.any(dim=1)
        clean_tokens = tokens.masked_fill(
            ~valid[..., None],
            0,
        )
        output = torch.zeros_like(clean_tokens)
        if active_rows.any():
            output[active_rows] = self.encoder(
                clean_tokens[active_rows],
                src_key_padding_mask=~valid[active_rows],
            )
        return output.masked_fill(
            ~valid[..., None],
            0,
        )
