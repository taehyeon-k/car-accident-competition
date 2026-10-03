"""Compact temporal feature pyramid for dense event localization."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from stage2.spotting_experiments.models import EventConditionedHeads


class ResidualDW(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.GroupNorm(8, hidden)
        self.depthwise = nn.Conv1d(hidden, hidden, 5, padding=2, groups=hidden)
        self.pointwise = nn.Conv1d(hidden, hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.pointwise(F.gelu(self.depthwise(self.norm(x)))))


class TemporalPyramid(nn.Module):
    def __init__(self, hidden=128, token_dim=16, dropout=0.35, levels=4,
                 global_coarse=False, category_head="global"):
        super().__init__()
        self.norm = nn.LayerNorm(384)
        self.token = nn.Linear(384, token_dim)
        self.frame = nn.Sequential(nn.Dropout(dropout), nn.Linear(70 * token_dim, hidden), nn.GELU())
        self.blocks = nn.ModuleList(ResidualDW(hidden, dropout) for _ in range(levels))
        self.global_coarse = global_coarse
        if global_coarse:
            self.coarse_norm = nn.LayerNorm(hidden)
            self.coarse_attention = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
            self.coarse_ff_norm = nn.LayerNorm(hidden)
            self.coarse_ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                           nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
        self.fusions = nn.ModuleList(ResidualDW(hidden, dropout) for _ in range(levels - 1))
        self.event = nn.Linear(hidden, 2)
        self.category_head = category_head
        if category_head == "event":
            self.categories = EventConditionedHeads(hidden, token_dim, 70, dropout)
        else:
            self.attn = nn.Linear(hidden, 1)
            self.side = nn.Linear(hidden, 2)
            self.evasion = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, valid):
        tokens = self.token(self.norm(x.float()))
        h = self.frame(tokens.flatten(2)).transpose(1, 2)
        levels = []
        level_valid = valid[:, None].float()
        for i, block in enumerate(self.blocks):
            if i:
                h = F.avg_pool1d(h, 2, stride=2, ceil_mode=True)
                level_valid = F.max_pool1d(level_valid, 2, stride=2, ceil_mode=True)
            h = block(h) * level_valid
            levels.append(h)
        if self.global_coarse:
            coarse = h.transpose(1, 2)
            valid_coarse = level_valid[:, 0].bool()
            query = self.coarse_norm(coarse)
            attended = self.coarse_attention(query, query, query,
                                             key_padding_mask=~valid_coarse,
                                             need_weights=False)[0]
            coarse = coarse + attended
            coarse = coarse + self.coarse_ff(self.coarse_ff_norm(coarse))
            h = coarse.transpose(1, 2) * level_valid
            levels[-1] = h
        h = levels[-1]
        for i in range(len(levels) - 2, -1, -1):
            h = self.fusions[i](levels[i] + F.interpolate(h, size=levels[i].shape[-1], mode="linear", align_corners=False))
        h = h.transpose(1, 2)
        neg = torch.finfo(h.dtype).min / 4
        event = self.event(self.drop(h))
        entry = event[..., 0].masked_fill(~valid, neg)
        collision = event[..., 1].masked_fill(~valid, neg)
        if self.category_head == "event":
            side, evasion = self.categories(h, valid, tokens, entry, collision)
        else:
            weights = self.attn(h).squeeze(-1).masked_fill(~valid, neg).softmax(-1)
            pooled = torch.einsum("bt,bth->bh", weights, h)
            side, evasion = self.side(self.drop(pooled)), self.evasion(self.drop(pooled)).squeeze(-1)
        return {"entry_logits": entry, "collision_logits": collision,
                "side_logits": side, "evasion_logits": evasion}
