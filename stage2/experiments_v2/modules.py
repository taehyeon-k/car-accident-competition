"""Frame encoding, multi-rate temporal differences, and readout heads."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class FrameEncoder(nn.Module):
    """Compress the dense DINO patch grid per frame while keeping spatial layout."""

    def __init__(self, n_tokens=70, dim=384, token_dim=32, hidden=192, dropout=0.3):
        super().__init__()
        self.n_tokens = n_tokens
        self.norm = nn.LayerNorm(dim)
        self.token = nn.Linear(dim, token_dim)
        self.frame = nn.Sequential(nn.Dropout(dropout),
                                   nn.Linear(n_tokens * token_dim, hidden), nn.GELU())

    def forward(self, x):
        tokens = self.token(self.norm(x.float()))
        return self.frame(tokens.flatten(2)), tokens


class MultiRateDifference(nn.Module):
    """Multi-rate temporal difference features (prompt v2 section 6).

    For per-frame ``F[t]`` build ``D_s[t] = F[t] - F[t-s]`` for each configured
    stride ``s``, clamping at the sequence start so early frames use the nearest
    available past frame. Strides are frame-space only and carry no time units.
    The fused result is gated so the model can learn how much each rate matters.
    """

    def __init__(self, hidden, strides=(1, 2, 4, 8), dropout=0.3, gated=True):
        super().__init__()
        self.strides = tuple(int(s) for s in strides)
        if any(s < 1 for s in self.strides):
            raise ValueError("strides must be >= 1")
        self.gated = gated
        width = hidden * (len(self.strides) + 1)
        self.project = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gate = nn.Linear(width, hidden) if gated else None
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h):
        """``h`` is B,T,H in sampled-frame order."""
        t = h.shape[1]
        index = torch.arange(t, device=h.device)
        parts = [h]
        for s in self.strides:
            past = h.index_select(1, (index - s).clamp_min(0))
            parts.append(h - past)
        stacked = torch.cat(parts, dim=-1)
        fused = self.project(stacked)
        if self.gate is not None:
            fused = fused * torch.sigmoid(self.gate(stacked))
        return self.norm(h + fused)


class GlobalCategoryHeads(nn.Module):
    """Attention-pooled global readout for entry_side and evasion_space."""

    def __init__(self, hidden, dropout):
        super().__init__()
        self.attn = nn.Linear(hidden, 1)
        self.side = nn.Linear(hidden, 2)
        self.evasion = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, valid, tokens, entry_logits, collision_logits):
        neg = torch.finfo(h.dtype).min / 4
        weights = self.attn(h).squeeze(-1).masked_fill(~valid, neg).softmax(-1)
        pooled = torch.einsum("bt,bth->bh", weights, h)
        return self.side(self.drop(pooled)), self.evasion(self.drop(pooled)).squeeze(-1)


class EventConditionedHeads(nn.Module):
    """Event-conditioned readout (prompt v2 section 13).

    ``entry_side`` reads a probability-weighted feature around the predicted
    ENTRY plus left/centre/right spatial zones, so horizontal direction survives
    pooling. ``evasion_space`` reads around the predicted COLLISION. Event
    probabilities are detached so the attribute losses cannot move the temporal
    distributions.
    """

    def __init__(self, hidden, token_dim, n_tokens, dropout, grid_width=10, detach=True):
        super().__init__()
        self.width = grid_width if n_tokens % grid_width == 0 else n_tokens
        self.detach = detach
        self.side = nn.Sequential(nn.Linear(hidden * 3 + token_dim * 3, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 2))
        self.evasion = nn.Sequential(nn.Linear(hidden * 2 + token_dim, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, 1))

    @staticmethod
    def pool(weights, values):
        return torch.einsum("bt,bth->bh", weights, values)

    def forward(self, h, valid, tokens, entry_logits, collision_logits):
        if self.detach:
            entry_logits = entry_logits.detach()
            collision_logits = collision_logits.detach()
        pe = torch.softmax(entry_logits.float(), -1)
        pc = torch.softmax(collision_logits.float(), -1)
        before_e = torch.cat([torch.zeros_like(pe[:, :1]), pe.cumsum(-1)[:, :-1]], -1)
        after_e = torch.flip(torch.cat([torch.zeros_like(pe[:, :1]),
                                        torch.flip(pe, [-1]).cumsum(-1)[:, :-1]], -1), [-1])
        before_c = torch.cat([torch.zeros_like(pc[:, :1]), pc.cumsum(-1)[:, :-1]], -1)

        def norm(w):
            w = w * valid
            return w / w.sum(-1, keepdim=True).clamp_min(1e-8)

        b, t, n, d = tokens.shape
        grid = tokens.reshape(b, t, -1, self.width, d)
        cuts = [0, self.width // 3, 2 * self.width // 3, self.width]
        zones = [grid[:, :, :, cuts[i]:cuts[i + 1]].mean((2, 3)) for i in range(3)]
        directional = torch.cat([self.pool(pe, z) for z in zones], -1)
        side_x = torch.cat([self.pool(norm(before_e), h), self.pool(pe, h),
                            self.pool(norm(after_e), h), directional], -1)
        evasion_x = torch.cat([self.pool(norm(before_c), h), self.pool(pc, h),
                               self.pool(pc, tokens.mean(2))], -1)
        return self.side(side_x), self.evasion(evasion_x).squeeze(-1)


def build_category_head(cfg):
    if cfg.get("category_head", "global") == "event_conditioned":
        return EventConditionedHeads(cfg["hidden"], cfg["token_dim"], cfg["n_tokens"],
                                     cfg["dropout"], grid_width=cfg.get("grid_width", 10),
                                     detach=cfg.get("detach_event_for_categories", True))
    return GlobalCategoryHeads(cfg["hidden"], cfg["dropout"])
