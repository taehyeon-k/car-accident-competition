"""Single-stage local/global Stage 2 event spotter.

The visual encoders are deliberately outside this module.  Frozen DINOv3 and
V-JEPA features are cached once, while this compact head is trained repeatedly.
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

    def __init__(self, geometry_dim: int = 13) -> None:
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
        self.scene_positions = nn.Parameter(torch.randn(7, 384) / math.sqrt(384))
        self.spatial = Transformer(num_layers=1)
        self.global_projection = nn.Sequential(nn.Linear(1024, 384), nn.LayerNorm(384))
        self.fusion = TemporalCrossAttention()
        self.temporal = nn.ModuleList(
            [MaskedDilatedConv(dilation) for dilation in (1, 2, 4, 8)]
        )
        self.entry_head = self._event_head()
        self.collision_head = self._event_head()
        self.side_head = self._attribute_head(768, 2)
        self.evasion_head = self._attribute_head(1152, 1)

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
            (valid[..., None].expand(-1, -1, 7), object_valid), dim=2
        )
        batch_size, length = valid.shape
        local = self.spatial(
            tokens.reshape(batch_size * length, 19, 384),
            spatial_valid.reshape(batch_size * length, 19),
        )[:, 0].reshape(batch_size, length, 384)

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
        entry_embedding = torch.einsum(
            "bt,btd->bd", entry_prob.detach(), hidden.float()
        )
        collision_embedding = torch.einsum(
            "bt,btd->bd", collision_prob.detach(), hidden.float()
        )
        global_embedding = self._masked_mean(
            global_tokens.float(), batch["global_valid"].bool()
        )
        # Six coarse spatial cells, pooled at the predicted collision instant.
        collision_scene = torch.einsum(
            "bt,btsd->bsd", collision_prob.detach(), scene[:, :, 1:].float()
        ).mean(dim=1)
        return {
            "entry_logits": entry_logits,
            "collision_logits": collision_logits,
            "side_logits": self.side_head(
                torch.cat((entry_embedding, global_embedding), dim=-1)
            ),
            "evasion_logits": self.evasion_head(
                torch.cat(
                    (collision_embedding, collision_scene, global_embedding), dim=-1
                )
            ).squeeze(-1),
            "hidden": hidden,
        }
