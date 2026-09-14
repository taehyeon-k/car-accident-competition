"""Single-stage local/global Stage 2 event spotter.

DINOv3 and V-JEPA run online with LoRA in JointSystem; every module here is
randomly initialised, so its width and depth are the part of the model that can
overfit 251 labelled videos. All dimensions are configurable and default to the
narrow setting: 256-D hidden, four heads, one spatial layer, one hybrid temporal
block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .joint_tracking import GEOMETRY_DIM
from .modules import Transformer

SCENE_TOKENS = 17

HEAD_DEFAULTS = {
    "hidden_dim": 256,
    "roi_dim": 192,
    "geometry_embedding_dim": 64,
    "spatial": {"layers": 1, "heads": 4, "ffn_dim": 768, "dropout": 0.10},
    "temporal": {"hybrid_blocks": 1, "heads": 4, "ffn_dim": 768, "dropout": 0.10},
    "attribute_hidden_dim": 64,
    "attribute_dropout": 0.15,
}


def head_config(config: dict | None = None) -> dict:
    """Resolve and validate the joint head's dimensions."""
    config = config or {}
    resolved = {}
    for key, default in HEAD_DEFAULTS.items():
        value = config.get(key, default)
        resolved[key] = (
            {**default, **(value or {})} if isinstance(default, dict) else value
        )
    hidden = int(resolved["hidden_dim"])
    roi, geometry = int(resolved["roi_dim"]), int(resolved["geometry_embedding_dim"])
    if hidden < 1:
        raise ValueError("hidden_dim must be positive")
    # The object token is the concatenation of appearance and geometry, so the two
    # must add up exactly rather than being reconciled by a projection.
    if roi + geometry != hidden:
        raise ValueError(
            f"roi_dim + geometry_embedding_dim must equal hidden_dim "
            f"({roi} + {geometry} != {hidden})"
        )
    for section in ("spatial", "temporal"):
        heads = int(resolved[section]["heads"])
        if heads < 1 or hidden % heads:
            raise ValueError(
                f"hidden_dim ({hidden}) must be divisible by {section}.heads ({heads})"
            )
    if int(resolved["spatial"]["layers"]) != 1:
        raise ValueError("The spatial stack is exactly one layer")
    if int(resolved["temporal"]["hybrid_blocks"]) < 1:
        raise ValueError("temporal.hybrid_blocks must be at least one")
    resolved["hidden_dim"], resolved["roi_dim"] = hidden, roi
    resolved["geometry_embedding_dim"] = geometry
    return resolved


class MaskedDilatedConv(nn.Module):
    def __init__(self, dilation: int, dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim, dim, 5, padding=2 * dilation, dilation=dilation, groups=dim
        )
        self.pointwise = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        clean = self.norm(x).masked_fill(~valid[..., None], 0)
        update = self.depthwise(clean.transpose(1, 2))
        update = self.pointwise(F.gelu(update)).transpose(1, 2)
        return (x + self.dropout(update)).masked_fill(~valid[..., None], 0)


class HybridTemporalBlock(nn.Module):
    """Windowed MHSA in parallel with three depthwise temporal scales."""

    def __init__(self, radius=16, dim=256, heads=4, ffn_dim=768, dropout=0.1):
        super().__init__()
        if not isinstance(radius, int) or radius < 0:
            raise ValueError("Attention radius must be a nonnegative integer")
        if dim % heads:
            raise ValueError("Temporal dim must be divisible by the head count")
        self.radius = radius
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attention_out = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(torch.zeros(heads, 2 * radius + 1))
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(dim, dim, 5, padding=2 * d, dilation=d, groups=dim)
                for d in (1, 2, 4)
            ]
        )
        self.conv_projection = nn.Linear(3 * dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x, valid):
        clean = self.norm(x).masked_fill(~valid[..., None], 0)
        batch, length, _ = x.shape
        q, k, v = (
            self.qkv(clean)
            .reshape(batch, length, 3, self.heads, self.head_dim)
            .unbind(2)
        )
        window = 2 * self.radius + 1

        def neighbors(value):
            value = F.pad(value, (0, 0, 0, 0, self.radius, self.radius))
            return value.unfold(1, window, 1).permute(0, 2, 1, 4, 3)

        keys, values = neighbors(k), neighbors(v)
        scores = torch.einsum("bthd,bhtwd->bhtw", q, keys).float() / math.sqrt(
            self.head_dim
        )
        scores = scores + self.relative_bias[None, :, None, :]
        allowed = F.pad(valid, (self.radius, self.radius)).unfold(1, window, 1)
        # Padded queries get one harmless key, then their outputs are cleared.
        allowed = allowed.clone()
        allowed[:, :, self.radius] |= ~valid
        attention = torch.softmax(scores.masked_fill(~allowed[:, None], -torch.inf), -1)
        attention = self.dropout(attention).to(values.dtype)
        context = torch.einsum("bhtw,bhtwd->bthd", attention, values).reshape(
            batch, length, self.dim
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

    def __init__(self, dim: int = 256, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("Event attention dim must be divisible by the head count")
        self.query_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.out_norm = nn.LayerNorm(dim)

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

    def __init__(self, dim: int = 256, heads: int = 4) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("Fusion dim must be divisible by the head count")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.local_norm = nn.LayerNorm(dim)
        self.global_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.time_mlp = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, heads))
        self.gate = nn.Linear(2 * dim, 1)
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
            batch, length, self.dim
        )
        context = self.out(context)
        gate = torch.sigmoid(self.gate(torch.cat((local, context), dim=-1)))
        return (local + gate * context).masked_fill(~local_valid[..., None], 0)


class JointStage2Model(nn.Module):
    """Consume compact frozen features and predict every Stage 2 target."""

    def __init__(
        self,
        geometry_dim: int = GEOMETRY_DIM,
        temporal_radius: int = 16,
        config: dict | None = None,
    ) -> None:
        super().__init__()
        settings = head_config(config)
        hidden = settings["hidden_dim"]
        roi_dim = settings["roi_dim"]
        geometry_embedding = settings["geometry_embedding_dim"]
        spatial, temporal = settings["spatial"], settings["temporal"]
        self.settings = settings
        self.hidden_dim = hidden

        self.scene_projection = nn.Sequential(
            nn.Linear(768, hidden), nn.LayerNorm(hidden)
        )
        self.roi_projection = nn.Sequential(
            nn.Linear(768, roi_dim), nn.LayerNorm(roi_dim)
        )
        # A deliberately small geometry embedding: nine scalars do not need width.
        self.geometry_projection = nn.Sequential(
            nn.Linear(geometry_dim, geometry_embedding // 2),
            nn.LayerNorm(geometry_embedding // 2),
            nn.GELU(),
            nn.Linear(geometry_embedding // 2, geometry_embedding),
            nn.LayerNorm(geometry_embedding),
        )
        self.scene_positions = nn.Parameter(
            torch.randn(SCENE_TOKENS, hidden) / math.sqrt(hidden)
        )
        self.spatial = Transformer(
            num_layers=spatial["layers"],
            d_model=hidden,
            heads=spatial["heads"],
            ffn_dim=spatial["ffn_dim"],
            dropout=spatial["dropout"],
        )
        self.global_projection = nn.Sequential(
            nn.Linear(1024, hidden), nn.LayerNorm(hidden)
        )
        self.fusion = TemporalCrossAttention(hidden, temporal["heads"])
        self.temporal = nn.ModuleList(
            [
                HybridTemporalBlock(
                    temporal_radius,
                    dim=hidden,
                    heads=temporal["heads"],
                    ffn_dim=temporal["ffn_dim"],
                    dropout=temporal["dropout"],
                )
                for _ in range(temporal["hybrid_blocks"])
            ]
            + [MaskedDilatedConv(1, dim=hidden, dropout=temporal["dropout"])]
        )
        self.entry_head = self._event_head(hidden, settings["attribute_hidden_dim"])
        self.collision_head = self._event_head(hidden, settings["attribute_hidden_dim"])
        self.entry_spatial = EventSpatialAttention(hidden, temporal["heads"])
        self.collision_spatial = EventSpatialAttention(hidden, temporal["heads"])
        self.side_head = self._attribute_head(hidden, 2, settings)
        self.evasion_head = self._attribute_head(hidden, 1, settings)

    @staticmethod
    def _event_head(input_dim: int, hidden_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    @staticmethod
    def _attribute_head(input_dim: int, output_dim: int, settings: dict) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, settings["attribute_hidden_dim"]),
            nn.GELU(),
            nn.Dropout(settings["attribute_dropout"]),
            nn.Linear(settings["attribute_hidden_dim"], output_dim),
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
            (valid[..., None].expand(-1, -1, SCENE_TOKENS), object_valid), dim=2
        )
        batch_size, length = valid.shape
        tokens_per_frame = tokens.shape[2]
        spatial = self.spatial(
            tokens.reshape(batch_size * length, tokens_per_frame, self.hidden_dim),
            spatial_valid.reshape(batch_size * length, tokens_per_frame),
        ).reshape(batch_size, length, tokens_per_frame, self.hidden_dim)
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
