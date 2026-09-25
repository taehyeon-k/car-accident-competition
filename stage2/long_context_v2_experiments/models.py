"""P2 temporal pyramid with controlled variants (parameter names match TemporalPyramid for avg pooling).

pool:   avg (P2 / C0) | max (C1) | sgp (C2: learnable strided depthwise conv downsampling, +~1k params/level)
phase:  0 = none; 3 = PRE / BETWEEN / POST dense phase head (D)
coarse: none | ssm (F1: small bidirectional diagonal SSM on the two coarsest levels)
        | local_attn (F2: windowed self-attention on the two coarsest levels)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from stage2.temporal_pyramid_experiments.models import ResidualDW


class SGPDown(nn.Module):
    """Strided depthwise conv (k=3, s=2) + residual avg-pool path: learnable, <1% extra params."""
    def __init__(self, hidden):
        super().__init__()
        self.dw = nn.Conv1d(hidden, hidden, 3, stride=2, padding=1, groups=hidden)
        nn.init.zeros_(self.dw.weight); nn.init.zeros_(self.dw.bias)

    def forward(self, x):
        avg = F.avg_pool1d(x, 2, stride=2, ceil_mode=True)
        return avg + self.dw(x)[..., :avg.shape[-1]]


class BiDiagSSM(nn.Module):
    """Tiny bidirectional diagonal linear recurrence (S4D/Mamba-flavoured, no selectivity), residual."""
    def __init__(self, hidden, state=16, dropout=0.35):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.inp = nn.Linear(hidden, 2 * state)
        self.decay_logit = nn.Parameter(torch.linspace(-2.0, 3.0, state).repeat(2))  # short..long memory, both directions
        self.out = nn.Linear(2 * state, hidden)
        self.drop = nn.Dropout(dropout)
        self.state = state

    def scan(self, u, decay):
        # u [B,T,S]; h_t = a*h_{t-1} + (1-a)*u_t (T is at most ~40 at coarse levels, loop is cheap)
        h = torch.zeros_like(u[:, 0]); outs = []
        for t in range(u.shape[1]):
            h = decay * h + (1 - decay) * u[:, t]; outs.append(h)
        return torch.stack(outs, 1)

    def forward(self, x, valid):  # x [B,H,T]
        z = self.inp(self.norm(x.transpose(1, 2))) * valid[..., None]
        a = 0.3 + 0.695 * torch.sigmoid(self.decay_logit)  # per-channel decay in (0.3, 0.995)
        fwd = self.scan(z[..., :self.state], a[:self.state])
        bwd = self.scan(z[..., self.state:].flip(1), a[self.state:]).flip(1)
        return x + self.drop(self.out(torch.cat([fwd, bwd], -1))).transpose(1, 2) * valid[:, None]


class LocalAttn(nn.Module):
    def __init__(self, hidden, window=9, dropout=0.35):
        super().__init__()
        self.norm = nn.LayerNorm(hidden); self.window = window
        self.attn = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)

    def forward(self, x, valid):
        t = x.shape[-1]; idx = torch.arange(t, device=x.device)
        mask = (idx[None] - idx[:, None]).abs() > self.window // 2
        q = self.norm(x.transpose(1, 2))
        out = self.attn(q, q, q, attn_mask=mask, key_padding_mask=~valid, need_weights=False)[0]
        return x + (out * valid[..., None]).transpose(1, 2)


class PhaseRefine(nn.Module):
    """MS-TCN-style refinement stage: phase probabilities -> dilated residual convs -> refined phase logits."""
    def __init__(self, channels=32, dilations=(1, 2, 4, 8), dropout=0.35):
        super().__init__()
        self.inp = nn.Conv1d(3, channels, 1)
        self.layers = nn.ModuleList(nn.ModuleDict({"d": nn.Conv1d(channels, channels, 3, padding=d, dilation=d),
                                                   "p": nn.Conv1d(channels, channels, 1)}) for d in dilations)
        self.drop = nn.Dropout(dropout); self.out = nn.Conv1d(channels, 3, 1)

    def forward(self, logits, valid):  # logits [B,T,3]
        m = valid[:, None].float()
        h = self.inp(logits.float().softmax(-1).transpose(1, 2)) * m
        for l in self.layers: h = (h + self.drop(l["p"](F.relu(l["d"](h))))) * m
        return self.out(h).transpose(1, 2)


class LCPyramid(nn.Module):
    def __init__(self, hidden=128, token_dim=16, dropout=0.35, levels=4, pool="avg", phase=0, coarse="none", motion=False,
                 phase_refine=0, decoder="direct"):
        super().__init__()
        self.decoder = decoder
        self.refine = nn.ModuleList(PhaseRefine(dropout=dropout) for _ in range(phase_refine))
        self.uses_motion = motion
        if motion:
            self.motion_proj = nn.Sequential(nn.LayerNorm(28), nn.Linear(28, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(384)
        self.token = nn.Linear(384, token_dim)
        self.frame = nn.Sequential(nn.Dropout(dropout), nn.Linear(70 * token_dim, hidden), nn.GELU())
        self.blocks = nn.ModuleList(ResidualDW(hidden, dropout) for _ in range(levels))
        self.pool = pool
        if pool == "sgp": self.downs = nn.ModuleList(SGPDown(hidden) for _ in range(levels - 1))
        self.coarse = coarse
        if coarse == "ssm": self.coarse_mods = nn.ModuleList(BiDiagSSM(hidden, dropout=dropout) for _ in range(2))
        elif coarse == "local_attn": self.coarse_mods = nn.ModuleList(LocalAttn(hidden, dropout=dropout) for _ in range(2))
        self.fusions = nn.ModuleList(ResidualDW(hidden, dropout) for _ in range(levels - 1))
        self.event = nn.Linear(hidden, 2)
        self.phase = nn.Linear(hidden, phase) if phase else None
        self.attn = nn.Linear(hidden, 1)
        self.side = nn.Linear(hidden, 2)
        self.evasion = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def down(self, i, h):
        if self.pool == "avg": return F.avg_pool1d(h, 2, stride=2, ceil_mode=True)
        if self.pool == "max": return F.max_pool1d(h, 2, stride=2, ceil_mode=True)
        return self.downs[i - 1](h)

    def forward(self, x, valid, return_hidden=False, motion=None):
        tokens = self.token(self.norm(x.float()))
        h = self.frame(tokens.flatten(2))
        if self.uses_motion: h = h + self.motion_proj(motion.float())
        h = h.transpose(1, 2)
        levels, masks = [], []
        level_valid = valid[:, None].float()
        for i, block in enumerate(self.blocks):
            if i:
                h = self.down(i, h * level_valid)
                level_valid = F.max_pool1d(level_valid, 2, stride=2, ceil_mode=True)
            h = block(h) * level_valid
            if self.coarse != "none" and i >= len(self.blocks) - 2:
                h = self.coarse_mods[i - (len(self.blocks) - 2)](h, level_valid[:, 0].bool()) * level_valid
            levels.append(h); masks.append(level_valid)
        h = levels[-1]
        for i in range(len(levels) - 2, -1, -1):
            h = self.fusions[i](levels[i] + F.interpolate(h, size=levels[i].shape[-1], mode="linear", align_corners=False))
        h = h.transpose(1, 2)
        neg = torch.finfo(h.dtype).min / 4
        event = self.event(self.drop(h))
        out = {"entry_logits": event[..., 0].masked_fill(~valid, neg),
               "collision_logits": event[..., 1].masked_fill(~valid, neg)}
        weights = self.attn(h).squeeze(-1).masked_fill(~valid, neg).softmax(-1)
        pooled = torch.einsum("bt,bth->bh", weights, h)
        out["side_logits"] = self.side(self.drop(pooled)); out["evasion_logits"] = self.evasion(self.drop(pooled)).squeeze(-1)
        if self.phase is not None:
            stages = [self.phase(self.drop(h))]
            for r in self.refine: stages.append(r(stages[-1], valid))
            out["phase_logits"] = stages[-1]; out["phase_stages"] = stages[:-1]
        if return_hidden: out["hidden"] = h
        return out
