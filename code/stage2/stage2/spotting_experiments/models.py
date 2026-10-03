from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn


class FrameEncoder(nn.Module):
    def __init__(self, n_tokens=70, dim=384, token_dim=32, hidden=192, dropout=0.3):
        super().__init__()
        self.n_tokens = n_tokens
        self.norm = nn.LayerNorm(dim)
        self.token = nn.Linear(dim, token_dim)
        self.frame = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_tokens * token_dim, hidden), nn.GELU())

    def forward(self, x):
        tokens = self.token(self.norm(x.float()))
        return self.frame(tokens.flatten(2)), tokens


class CategoryHeads(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.attn = nn.Linear(hidden, 1); self.side = nn.Linear(hidden, 2); self.evasion = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, valid, *_):
        neg = torch.finfo(h.dtype).min / 4
        weights = self.attn(h).squeeze(-1).masked_fill(~valid, neg).softmax(-1)
        pooled = torch.einsum("bt,bth->bh", weights, h)
        return self.side(self.drop(pooled)), self.evasion(self.drop(pooled)).squeeze(-1)


class EventConditionedHeads(nn.Module):
    """Soft event pooling with before/at/after context and left/center/right geometry."""
    def __init__(self, hidden, token_dim, n_tokens, dropout):
        super().__init__()
        width = round(math.sqrt(n_tokens * 10 / 7))
        self.width = width if n_tokens % width == 0 else n_tokens
        self.side = nn.Sequential(nn.Linear(hidden * 3 + token_dim * 3, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 2))
        self.evasion = nn.Sequential(nn.Linear(hidden * 2 + token_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    @staticmethod
    def pool(weights, values): return torch.einsum("bt,bth->bh", weights, values)

    def forward(self, h, valid, tokens, entry_logits, collision_logits):
        pe = torch.softmax(entry_logits.float(), -1); pc = torch.softmax(collision_logits.float(), -1)
        before_e = torch.cat([torch.zeros_like(pe[:, :1]), pe.cumsum(-1)[:, :-1]], -1)
        after_e = torch.flip(torch.cat([torch.zeros_like(pe[:, :1]), torch.flip(pe, [-1]).cumsum(-1)[:, :-1]], -1), [-1])
        before_c = torch.cat([torch.zeros_like(pc[:, :1]), pc.cumsum(-1)[:, :-1]], -1)
        def norm(w):
            w = w * valid; return w / w.sum(-1, keepdim=True).clamp_min(1e-8)
        spatial = tokens.reshape(*tokens.shape[:3], -1)
        # tokens is B,T,N,D; average three horizontal zones while preserving direction.
        b, t, n, d = tokens.shape; grid = tokens.reshape(b, t, -1, self.width, d)
        cuts = [0, self.width // 3, 2 * self.width // 3, self.width]
        zones = [grid[:, :, :, cuts[i]:cuts[i+1]].mean((2, 3)) for i in range(3)]
        directional = torch.cat([self.pool(pe, z) for z in zones], -1)
        side_x = torch.cat([self.pool(norm(before_e), h), self.pool(pe, h), self.pool(norm(after_e), h), directional], -1)
        collision_spatial = tokens.mean(2)
        evasion_x = torch.cat([self.pool(norm(before_c), h), self.pool(pc, h), self.pool(pc, collision_spatial)], -1)
        return self.side(side_x), self.evasion(evasion_x).squeeze(-1)


class BaseSpotter(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.encoder = FrameEncoder(cfg["n_tokens"], cfg["feature_dim"], cfg["token_dim"], cfg["hidden"], cfg["dropout"])
        cat = cfg.get("category_head", "global")
        self.categories = (EventConditionedHeads(cfg["hidden"], cfg["token_dim"], cfg["n_tokens"], cfg["dropout"])
                           if cat == "event_conditioned" else CategoryHeads(cfg["hidden"], cfg["dropout"]))
        self.offsets_enabled = bool(cfg.get("offsets", False))
        self.offset_limit = float(cfg.get("offset_limit", 0.08))

    def finish(self, h, tokens, valid):
        logits = self.event(h)
        neg = torch.finfo(logits.dtype).min / 4
        entry = logits[..., 0].masked_fill(~valid, neg); collision = logits[..., 1].masked_fill(~valid, neg)
        side, evasion = self.categories(h, valid, tokens, entry, collision)
        out = {"entry_logits": entry, "collision_logits": collision, "side_logits": side, "evasion_logits": evasion}
        if self.offsets_enabled:
            offsets = torch.tanh(self.offset(h)) * self.offset_limit
            out.update(entry_offsets=offsets[..., 0], collision_offsets=offsets[..., 1])
        return out


class BaselineSpotter(BaseSpotter):
    def __init__(self, cfg):
        super().__init__(cfg); h = cfg["hidden"]; self.drop = nn.Dropout(cfg["dropout"])
        self.t1 = nn.Conv1d(h, h, 5, padding=2); self.t2 = nn.Conv1d(h, h, 5, dilation=2, padding=4)
        self.event = nn.Linear(h, 2); self.offset = nn.Linear(h, 2)
    def forward(self, x, valid):
        h, tokens = self.encoder(x); z = h.transpose(1, 2)
        z = z + F.gelu(self.t1(z)); z = z + F.gelu(self.t2(self.drop(z)))
        return self.finish(z.transpose(1, 2), tokens, valid)


class ConvBlock(nn.Module):
    def __init__(self, h, dropout):
        super().__init__(); self.net = nn.Sequential(nn.Conv1d(h, h, 3, padding=1), nn.GroupNorm(8, h), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(h, h, 3, padding=1))
    def forward(self, x): return x + self.net(x)


class TDeedLite(BaseSpotter):
    def __init__(self, cfg):
        super().__init__(cfg); h = cfg["hidden"]; d = cfg["dropout"]
        self.local = ConvBlock(h, d); self.deep = nn.Sequential(ConvBlock(h, d), ConvBlock(h, d)); self.fuse = nn.Conv1d(h * 2, h, 1)
        self.event = nn.Linear(h, 2); self.offset = nn.Linear(h, 2)
    def forward(self, x, valid):
        h, tokens = self.encoder(x); full = self.local(h.transpose(1, 2))
        down = F.max_pool1d(full, 2, ceil_mode=True); deep = self.deep(down)
        up = F.interpolate(deep, size=full.shape[-1], mode="linear", align_corners=False)
        h = F.gelu(self.fuse(torch.cat([full, up], 1))).transpose(1, 2)
        return self.finish(h, tokens, valid)


class TemporalMaxerLite(BaseSpotter):
    def __init__(self, cfg):
        super().__init__(cfg); h = cfg["hidden"]; kernels = cfg.get("pool_kernels", [3, 5, 9, 17])
        self.kernels = kernels; self.fuse = nn.Sequential(nn.Conv1d(h * (len(kernels) + 1), h, 1), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Conv1d(h, h, 1))
        self.event = nn.Linear(h, 2); self.offset = nn.Linear(h, 2)
    def forward(self, x, valid):
        h, tokens = self.encoder(x); base = h.transpose(1, 2)
        pooled = [F.max_pool1d(base, k, stride=1, padding=k // 2) for k in self.kernels]
        h = (base + self.fuse(torch.cat([base] + pooled, 1))).transpose(1, 2)
        return self.finish(h, tokens, valid)


class LocalAttentionLayer(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__(); self.norm1 = nn.LayerNorm(hidden); self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden); self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
    def forward(self, x, valid, mask):
        q = self.norm1(x); a = self.attn(q, q, q, attn_mask=mask, key_padding_mask=~valid, need_weights=False)[0]
        x = x + a; return x + self.ff(self.norm2(x))


class ASFormerLite(BaseSpotter):
    def __init__(self, cfg):
        super().__init__(cfg); self.window = int(cfg.get("attention_window", 17))
        self.layers = nn.ModuleList([LocalAttentionLayer(cfg["hidden"], cfg.get("attention_heads", 4), cfg["dropout"]) for _ in range(cfg.get("attention_layers", 2))])
        self.event = nn.Linear(cfg["hidden"], 2); self.offset = nn.Linear(cfg["hidden"], 2)
    def forward(self, x, valid):
        h, tokens = self.encoder(x); t = h.shape[1]; idx = torch.arange(t, device=h.device)
        mask = (idx[:, None] - idx[None]).abs() > self.window
        for layer in self.layers: h = layer(h, valid, mask)
        return self.finish(h, tokens, valid)


def build_model(cfg):
    return {"baseline": BaselineSpotter, "tdeed": TDeedLite, "temporal_maxer": TemporalMaxerLite,
            "asformer": ASFormerLite}[cfg["head"]](cfg)
