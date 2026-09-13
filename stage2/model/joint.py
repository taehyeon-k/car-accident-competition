"""Single-stage local/global Stage 2 event spotter.

DINO features are cached; frozen V-JEPA runs online in JointSystem.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Transformer


class MaskedDilatedConv(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(384)
        self.depthwise = nn.Conv1d(
            384, 384, 5, padding=2 * dilation, dilation=dilation, groups=384
        )
        self.pointwise = nn.Conv1d(384, 384, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        clean = self.norm(x).masked_fill(~valid[..., None], 0)
        update = self.depthwise(clean.transpose(1, 2))
        update = self.pointwise(F.gelu(update)).transpose(1, 2)
        return (x + self.dropout(update)).masked_fill(~valid[..., None], 0)


class HybridTemporalBlock(nn.Module):
    """Windowed MHSA in parallel with three depthwise temporal scales."""

    def __init__(self, radius=16):
        super().__init__()
        if not isinstance(radius, int) or radius < 0:
            raise ValueError("Attention radius must be a nonnegative integer")
        self.radius = radius
        self.norm = nn.LayerNorm(384)
        self.qkv = nn.Linear(384, 1152)
        self.attention_out = nn.Linear(384, 384)
        self.relative_bias = nn.Parameter(torch.zeros(6, 2 * radius + 1))
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(384, 384, 5, padding=2 * d, dilation=d, groups=384)
                for d in (1, 2, 4)
            ]
        )
        self.conv_projection = nn.Linear(1152, 384)
        self.dropout = nn.Dropout(0.1)
        self.ffn_norm = nn.LayerNorm(384)
        self.ffn = nn.Sequential(
            nn.Linear(384, 1536), nn.GELU(), nn.Dropout(0.1), nn.Linear(1536, 384)
        )

    def forward(self, x, valid):
        clean = self.norm(x).masked_fill(~valid[..., None], 0)
        batch, length, _ = x.shape
        q, k, v = self.qkv(clean).reshape(batch, length, 3, 6, 64).unbind(2)
        window = 2 * self.radius + 1

        def neighbors(value):
            value = F.pad(value, (0, 0, 0, 0, self.radius, self.radius))
            return value.unfold(1, window, 1).permute(0, 2, 1, 4, 3)

        keys, values = neighbors(k), neighbors(v)
        scores = torch.einsum("bthd,bhtwd->bhtw", q, keys).float() / 8
        scores = scores + self.relative_bias[None, :, None, :]
        allowed = F.pad(valid, (self.radius, self.radius)).unfold(1, window, 1)
        # Padded queries get one harmless key, then their outputs are cleared.
        allowed = allowed.clone()
        allowed[:, :, self.radius] |= ~valid
        attention = torch.softmax(scores.masked_fill(~allowed[:, None], -torch.inf), -1)
        attention = self.dropout(attention).to(values.dtype)
        context = torch.einsum("bhtw,bhtwd->bthd", attention, values).reshape(
            batch, length, 384
        )
        conv = torch.cat(
            [
                F.gelu(layer(clean.transpose(1, 2))).transpose(1, 2)
                for layer in self.convs
            ],
            -1,
        )
        update = self.attention_out(context) + self.conv_projection(conv)
        x = (x + self.dropout(update)).masked_fill(~valid[..., None], 0)
        return (x + self.dropout(self.ffn(self.ffn_norm(x)))).masked_fill(
            ~valid[..., None], 0
        )


class EventSpatialAttention(nn.Module):
    """An event query attends to 16 scene cells and 12 persistent objects."""

    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(384)
        self.token_norm = nn.LayerNorm(384)
        self.attention = nn.MultiheadAttention(384, 6, dropout=0.1, batch_first=True)
        self.out_norm = nn.LayerNorm(384)

    def forward(self, probability, hidden, tokens, valid):
        weights = probability.detach().float()
        query = torch.einsum("bt,btd->bd", weights, hidden.float())
        tokens = tokens.masked_fill(~valid[..., None], 0)
        pooled = torch.einsum("bt,btsd->bsd", weights, tokens.float())
        token_valid = valid.any(1)
        normalized = self.token_norm(pooled)
        context, _ = self.attention(
            self.query_norm(query)[:, None],
            normalized,
            normalized,
            key_padding_mask=~token_valid,
            need_weights=False,
        )
        return self.out_norm(query + context[:, 0])


class TemporalCrossAttention(nn.Module):
    """Local queries retrieve sparse global context with signed time bias."""

    def __init__(self, heads: int = 6) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = 384 // heads
        self.local_norm = nn.LayerNorm(384)
        self.global_norm = nn.LayerNorm(384)
        self.q = nn.Linear(384, 384)
        self.k = nn.Linear(384, 384)
        self.v = nn.Linear(384, 384)
        self.out = nn.Linear(384, 384)
        self.time_mlp = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, heads))
        self.gate = nn.Linear(768, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(
        self,
        local: torch.Tensor,
        global_tokens: torch.Tensor,
        local_time: torch.Tensor,
        global_time: torch.Tensor,
        local_valid: torch.Tensor,
        global_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = local.shape
        global_length = global_tokens.shape[1]
        q = self.q(self.local_norm(local)).view(
            batch, length, self.heads, self.head_dim
        )
        k = self.k(self.global_norm(global_tokens)).view(
            batch, global_length, self.heads, self.head_dim
        )
        v = self.v(self.global_norm(global_tokens)).view(
            batch, global_length, self.heads, self.head_dim
        )
        scores = torch.einsum("bthd,bghd->bhtg", q, k) / math.sqrt(self.head_dim)
        delta = local_time[:, :, None] - global_time[:, None, :]
        time_features = torch.stack((delta, delta.abs(), delta.square()), dim=-1)
        bias = self.time_mlp(time_features.float()).permute(0, 3, 1, 2)
        scores = scores.float() + bias
        scores = scores.masked_fill(~global_valid[:, None, None, :], -torch.inf)
        attention = torch.softmax(scores, dim=-1).to(v.dtype)
        context = torch.einsum("bhtg,bghd->bthd", attention, v).reshape(
            batch, length, 384
        )
        context = self.out(context)
        gate = torch.sigmoid(self.gate(torch.cat((local, context), dim=-1)))
        return (local + gate * context).masked_fill(~local_valid[..., None], 0)


class JointStage2Model(nn.Module):
    """Consume compact frozen features and predict every Stage 2 target."""

    def __init__(self, geometry_dim: int = 13, temporal_radius: int = 16) -> None:
        super().__init__()
        self.scene_projection = nn.Sequential(nn.Linear(768, 384), nn.LayerNorm(384))
        self.roi_projection = nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256))
        self.geometry_projection = nn.Sequential(
            nn.Linear(geometry_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
        )
        self.scene_positions = nn.Parameter(torch.randn(17, 384) / math.sqrt(384))
        self.spatial = Transformer(num_layers=1)
        self.global_projection = nn.Sequential(nn.Linear(1024, 384), nn.LayerNorm(384))
        self.fusion = TemporalCrossAttention()
        self.temporal = nn.ModuleList(
            [HybridTemporalBlock(temporal_radius) for _ in range(2)]
            + [MaskedDilatedConv(1)]
        )
        self.entry_head = self._event_head()
        self.collision_head = self._event_head()
        self.entry_spatial = EventSpatialAttention()
        self.collision_spatial = EventSpatialAttention()
        self.side_head = self._attribute_head(384, 2)
        self.evasion_head = self._attribute_head(384, 1)

    @staticmethod
    def _event_head() -> nn.Module:
        return nn.Sequential(nn.Linear(384, 128), nn.GELU(), nn.Linear(128, 1))

    @staticmethod
    def _attribute_head(input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.to(x.dtype)
        return (x * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(
            1
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        valid = batch["time_valid"].bool()
        if not valid.any(1).all():
            raise ValueError("Each joint sample needs a valid frame")
        object_valid = batch["object_valid"].bool() & valid[..., None]
        parameter_dtype = self.scene_projection[0].weight.dtype
        scene = self.scene_projection(batch["scene_features"].to(dtype=parameter_dtype))
        scene = scene + self.scene_positions[None, None]
        roi = self.roi_projection(batch["roi_features"].to(dtype=parameter_dtype))
        geometry = self.geometry_projection(batch["geometry"].float()).to(roi.dtype)
        objects = torch.cat((roi, geometry), dim=-1)
        objects = objects.masked_fill(~object_valid[..., None], 0)
        tokens = torch.cat((scene, objects), dim=2)
        spatial_valid = torch.cat(
            (valid[..., None].expand(-1, -1, 17), object_valid), dim=2
        )
        batch_size, length = valid.shape
        spatial = self.spatial(
            tokens.reshape(batch_size * length, 29, 384),
            spatial_valid.reshape(batch_size * length, 29),
        ).reshape(batch_size, length, 29, 384)
        local = spatial[:, :, 0]

        if not batch["global_valid"].any(1).all():
            raise ValueError("Each joint sample needs a valid global token")
        global_tokens = self.global_projection(
            batch["global_features"].to(dtype=parameter_dtype)
        )
        hidden = self.fusion(
            local,
            global_tokens,
            batch["local_time"].float(),
            batch["global_time"].float(),
            valid,
            batch["global_valid"].bool(),
        )
        for block in self.temporal:
            hidden = block(hidden, valid)

        entry_logits = (
            self.entry_head(hidden).squeeze(-1).masked_fill(~valid, -torch.inf)
        )
        collision_logits = (
            self.collision_head(hidden).squeeze(-1).masked_fill(~valid, -torch.inf)
        )
        entry_prob = torch.softmax(entry_logits.float(), dim=-1)
        collision_prob = torch.softmax(collision_logits.float(), dim=-1)
        spatial_tokens, attribute_valid = spatial[:, :, 1:], spatial_valid[:, :, 1:]
        entry_embedding = self.entry_spatial(
            entry_prob, hidden, spatial_tokens, attribute_valid
        )
        collision_embedding = self.collision_spatial(
            collision_prob, hidden, spatial_tokens, attribute_valid
        )
        return {
            "entry_logits": entry_logits,
            "collision_logits": collision_logits,
            "side_logits": self.side_head(entry_embedding),
            "evasion_logits": self.evasion_head(collision_embedding).squeeze(-1),
            "hidden": hidden,
        }
