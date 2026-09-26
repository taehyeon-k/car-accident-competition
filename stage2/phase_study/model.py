"""LCPyramid (unchanged backbone, same module names/sizes as long_context_v2_experiments.models) + phase-head variants.

phase_rep:  none | cat (3 logits PRE/BETWEEN/POST) | ord (2 logits, nested cumulative model:
            q_E = P(phase >= BETWEEN) = sigmoid(z1), q_C = P(phase >= POST) = q_E * sigmoid(z2)  => q_E >= q_C always).
attach:     final       phase head on the final full-resolution fused sequence (S1, current approach)
            detach      same head on h.detach() (diagnostic: phase is learned but sends no gradient into the backbone)
            multiscale  final head + one aux head per coarser pyramid level (S2; aux heads are training-only)
            cond_detach / cond_e2e   final head, and the event head also receives local phase evidence (S3)
Output 'phase_logp' [B,T,3] = log P(PRE), log P(BETWEEN), log P(POST) for every representation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from stage2.long_context_v2_experiments.models import LCPyramid

ATTACH = ("final", "detach", "multiscale", "cond_detach", "cond_e2e")


def implied_logp(raw, rep):
    if rep == "cat": return F.log_softmax(raw.float(), -1)
    z1, z2 = raw.float().unbind(-1)
    return torch.stack([F.logsigmoid(-z1), F.logsigmoid(z1) + F.logsigmoid(-z2), F.logsigmoid(z1) + F.logsigmoid(z2)], -1)


class PhasePyramid(LCPyramid):
    def __init__(self, phase_rep="none", attach="final", motion=False, **kw):
        super().__init__(phase=0, motion=motion, **kw)
        assert phase_rep in ("none", "cat", "ord") and attach in ATTACH
        self.phase_rep, self.attach, self.decoder = phase_rep, attach, "direct"
        hidden = self.event.in_features
        k = {"cat": 3, "ord": 2}.get(phase_rep, 0)
        self.phase = nn.Linear(hidden, k) if k else None
        if k and attach == "multiscale":
            self.aux_phase = nn.ModuleList(nn.Linear(hidden, k) for _ in range(len(self.blocks) - 1))
        if k and attach.startswith("cond"):
            # local phase evidence [logp(t), logp(t) - logp(t-1)] -> event-logit offsets; zero-init = starts as plain model
            self.cond = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 2))
            nn.init.zeros_(self.cond[-1].weight); nn.init.zeros_(self.cond[-1].bias)

    def forward(self, x, valid, motion=None, return_hidden=False):
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
            levels.append(h); masks.append(level_valid)
        h = levels[-1]
        for i in range(len(levels) - 2, -1, -1):
            h = self.fusions[i](levels[i] + F.interpolate(h, size=levels[i].shape[-1], mode="linear", align_corners=False))
        h = h.transpose(1, 2)
        neg = torch.finfo(h.dtype).min / 4
        event = self.event(self.drop(h))
        out = {}
        if self.phase is not None:
            src = h.detach() if self.attach == "detach" else h
            raw = self.phase(self.drop(src))
            logp = implied_logp(raw, self.phase_rep)
            out["phase_raw"], out["phase_logp"] = raw, logp
            if self.attach.startswith("cond"):
                lp = logp.detach() if self.attach == "cond_detach" else logp
                prev = torch.cat([lp[:, :1], lp[:, :-1]], 1)
                event = event + self.cond(torch.cat([lp, lp - prev], -1).to(event.dtype))
            if self.attach == "multiscale" and self.training:
                out["aux_phase_raw"] = [head(self.drop(lv.transpose(1, 2))) for head, lv in zip(self.aux_phase, levels[1:])]
                out["aux_valid"] = [m[:, 0] for m in masks[1:]]
        out["entry_logits"] = event[..., 0].masked_fill(~valid, neg)
        out["collision_logits"] = event[..., 1].masked_fill(~valid, neg)
        weights = self.attn(h).squeeze(-1).masked_fill(~valid, neg).softmax(-1)
        pooled = torch.einsum("bt,bth->bh", weights, h)
        out["side_logits"] = self.side(self.drop(pooled)); out["evasion_logits"] = self.evasion(self.drop(pooled)).squeeze(-1)
        if return_hidden: out["hidden"] = h
        return out


def build(cfg):
    return PhasePyramid(phase_rep=cfg["phase_rep"], attach=cfg["attach"], motion=cfg.get("motion", False))


def load(path, device):
    st = torch.load(path, map_location="cpu", weights_only=False)
    m = build(st["config"]); m.load_state_dict(st["model"]); return m.to(device).eval()
