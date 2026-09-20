"""Composable spotter: frame encoder -> optional multi-rate diffs -> temporal head."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .modules import FrameEncoder, MultiRateDifference, build_category_head


class IdentityTemporal(nn.Module):
    """No temporal mixing: per-frame readout only (prompt v2 baseline / E1)."""

    def forward(self, h, valid):
        return h


class DilatedConvTemporal(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.t1 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.t2 = nn.Conv1d(hidden, hidden, 5, dilation=2, padding=4)

    def forward(self, h, valid):
        z = h.transpose(1, 2)
        z = z + F.gelu(self.t1(z))
        z = z + F.gelu(self.t2(self.drop(z)))
        return z.transpose(1, 2)


class TemporalMaxerLite(nn.Module):
    """TemporalMaxer-style parameter-free multi-scale pooling (candidate B)."""

    def __init__(self, hidden, dropout, kernels=(3, 5, 9, 17)):
        super().__init__()
        self.kernels = tuple(kernels)
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden * (len(self.kernels) + 1), hidden, 1), nn.GELU(),
            nn.Dropout(dropout), nn.Conv1d(hidden, hidden, 1))

    def forward(self, h, valid):
        base = h.transpose(1, 2)
        pooled = [F.max_pool1d(base, k, stride=1, padding=k // 2) for k in self.kernels]
        return (base + self.fuse(torch.cat([base] + pooled, 1))).transpose(1, 2)


class LocalAttentionLayer(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))

    def forward(self, x, valid, mask):
        q = self.norm1(x)
        a = self.attn(q, q, q, attn_mask=mask, key_padding_mask=~valid, need_weights=False)[0]
        x = x + a
        return x + self.ff(self.norm2(x))


class LocalAttentionTemporal(nn.Module):
    """Small local-window temporal transformer (candidate C).

    Attention is restricted to a relative window, and no absolute temporal
    embedding is added, so the model cannot memorise where events usually sit.
    """

    def __init__(self, hidden, dropout, layers=2, heads=4, window=17):
        super().__init__()
        self.window = int(window)
        self.layers = nn.ModuleList([LocalAttentionLayer(hidden, heads, dropout)
                                     for _ in range(layers)])

    def forward(self, h, valid):
        t = h.shape[1]
        idx = torch.arange(t, device=h.device)
        mask = (idx[:, None] - idx[None]).abs() > self.window
        for layer in self.layers:
            h = layer(h, valid, mask)
        return h


def build_temporal(cfg):
    kind = cfg.get("temporal", "none")
    hidden, dropout = cfg["hidden"], cfg["dropout"]
    if kind == "none":
        return IdentityTemporal()
    if kind == "dilated_conv":
        return DilatedConvTemporal(hidden, dropout)
    if kind == "temporal_maxer":
        return TemporalMaxerLite(hidden, dropout, cfg.get("pool_kernels", (3, 5, 9, 17)))
    if kind == "local_attention":
        return LocalAttentionTemporal(hidden, dropout, cfg.get("attention_layers", 2),
                                      cfg.get("attention_heads", 4),
                                      cfg.get("attention_window", 17))
    raise ValueError(f"unknown temporal head: {kind}")


class Spotter(nn.Module):
    """Dense per-frame ENTRY/COLLISION spotter with optional auxiliary state heads."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        hidden = cfg["hidden"]
        self.encoder = FrameEncoder(cfg["n_tokens"], cfg["feature_dim"], cfg["token_dim"],
                                    hidden, cfg["dropout"])
        self.diff = (MultiRateDifference(hidden, cfg.get("diff_strides", (1, 2, 4, 8)),
                                         cfg["dropout"], cfg.get("diff_gated", True))
                     if cfg.get("multi_rate_diff", False) else None)
        self.temporal = build_temporal(cfg)
        self.categories = build_category_head(cfg)
        self.event = nn.Linear(hidden, 2)
        self.state_aux = bool(cfg.get("state_aux", False))
        if self.state_aux:
            # BEFORE/BOUNDARY/AFTER for entry, PRE/CONTACT/POST for collision.
            self.entry_state = nn.Linear(hidden, 3)
            self.collision_state = nn.Linear(hidden, 3)

    def forward(self, x, valid):
        h, tokens = self.encoder(x)
        if self.diff is not None:
            h = self.diff(h)
        h = self.temporal(h, valid)
        logits = self.event(h)
        neg = torch.finfo(logits.dtype).min / 4
        entry = logits[..., 0].masked_fill(~valid, neg)
        collision = logits[..., 1].masked_fill(~valid, neg)
        side, evasion = self.categories(h, valid, tokens, entry, collision)
        out = {"entry_logits": entry, "collision_logits": collision,
               "side_logits": side, "evasion_logits": evasion}
        if self.state_aux:
            out["entry_state_logits"] = self.entry_state(h)
            out["collision_state_logits"] = self.collision_state(h)
        return out


class PooledRegressor(nn.Module):
    """Prompt v2 'current simple head' baseline.

    Mean-pools over time and regresses one normalized scalar position per event.
    Deliberately not dense - this is the thing E1 is meant to beat.
    """

    def __init__(self, cfg):
        super().__init__()
        hidden = cfg["hidden"]
        self.encoder = FrameEncoder(cfg["n_tokens"], cfg["feature_dim"], cfg["token_dim"],
                                    hidden, cfg["dropout"])
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(cfg["dropout"]))
        self.event = nn.Linear(hidden, 2)
        self.side = nn.Linear(hidden, 2)
        self.evasion = nn.Linear(hidden, 1)

    def forward(self, x, valid):
        h, _ = self.encoder(x)
        mask = valid.float()[..., None]
        pooled = self.trunk((h * mask).sum(1) / mask.sum(1).clamp_min(1))
        positions = torch.sigmoid(self.event(pooled))
        return {"entry_position": positions[..., 0], "collision_position": positions[..., 1],
                "side_logits": self.side(pooled), "evasion_logits": self.evasion(pooled).squeeze(-1)}


def build_model(cfg):
    if cfg.get("model", "spotter") == "pooled_regressor":
        return PooledRegressor(cfg)
    return Spotter(cfg)
