"""Geometry losses, representation anchor and the multi-task loss manager.

Every dense loss is ``sum(weight * loss) / sum(weight)`` where the weight folds
together the valid mask (IGNORE / missing label -> 0) and pseudo-label
confidence, i.e. ``mean(confidence * valid_mask * task_loss)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from stage2.geometry_pretrain.common import IGNORE


def _wmean(loss, weight):
    denom = weight.sum()
    return (loss * weight).sum() / denom.clamp_min(1.0), denom


def boundary_map(label: torch.Tensor) -> torch.Tensor:
    """1 where a 3x3 neighborhood contains a different valid class."""
    x = label.float()[:, None]
    mx = F.max_pool2d(x, 3, 1, 1)
    mn = -F.max_pool2d(-x, 3, 1, 1)
    return ((mx != mn) & (label[:, None] != IGNORE)).float()[:, 0]


def seg_ce(logits, target, weight, boundary_boost: float = 0.0, class_weight=None):
    valid = (target != IGNORE).float() * weight
    if boundary_boost > 0:
        valid = valid * (1 + boundary_boost * boundary_map(target))
    loss = F.cross_entropy(logits.float(), target.clamp_max(logits.shape[1] - 1).long(), weight=class_weight, reduction="none")
    return _wmean(loss, valid)


def binary_thin(logits, target, weight, pos_weight: float = 4.0, dice: float = 1.0):
    """Weighted BCE + soft Dice for thin structures (lanes, curbs, contact bands)."""
    valid = (target != IGNORE).float() * weight
    t = (target == 1).float()
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, t, pos_weight=logits.new_tensor(pos_weight), reduction="none")
    l_bce, denom = _wmean(bce, valid)
    if dice <= 0 or denom == 0:
        return l_bce, denom
    p = torch.sigmoid(logits) * valid
    tt = t * valid
    dims = (1, 2)
    inter = (p * tt).sum(dims)
    union = p.sum(dims) + tt.sum(dims)
    has = (valid.sum(dims) > 0).float()
    d = 1 - (2 * inter + 1) / (union + 1)
    l_dice = (d * has).sum() / has.sum().clamp_min(1)
    return l_bce + dice * l_dice, denom


def ssi_depth(pred, target, weight):
    """Scale-and-shift-invariant loss on relative inverse depth (MiDaS style)."""
    B = pred.shape[0]
    pred = pred.float().reshape(B, -1)
    target = target.float().reshape(B, -1)
    w = weight.float().reshape(B, -1)
    losses, n = [], 0
    for b in range(B):
        m = w[b] > 0
        if m.sum() < 64:
            continue
        p, t, ww = pred[b][m], target[b][m], w[b][m]

        def norm(x):
            med = x.median()
            s = (x - med).abs().mean().clamp_min(1e-6)
            return (x - med) / s

        l = (ww * (norm(p) - norm(t)).abs()).sum() / ww.sum()
        losses.append(l)
        n += 1
    if not losses:
        return pred.sum() * 0, torch.tensor(0.0)
    return torch.stack(losses).mean(), torch.tensor(float(n))


def depth_gradient(pred, target, weight):
    """Gradient-matching term on normalized disparity (sharpens depth edges)."""
    def norm(x, m):
        B = x.shape[0]
        out = []
        for b in range(B):
            v = x[b][m[b] > 0]
            if v.numel() < 64:
                out.append(torch.zeros_like(x[b]))
                continue
            med = v.median()
            s = (v - med).abs().mean().clamp_min(1e-6)
            out.append((x[b] - med) / s)
        return torch.stack(out)

    p, t = norm(pred.float(), weight), norm(target.float(), weight)
    r = p - t
    gx = (r[..., :, 1:] - r[..., :, :-1]).abs()
    gy = (r[..., 1:, :] - r[..., :-1, :]).abs()
    wx = weight[..., :, 1:] * weight[..., :, :-1]
    wy = weight[..., 1:, :] * weight[..., :-1, :]
    return 0.5 * (_wmean(gx, wx)[0] + _wmean(gy, wy)[0])


def flow_loss(pred, target, weight, eps: float = 0.01, q: float = 0.4):
    """Robust (generalized Charbonnier) endpoint loss in stride-4 pixels."""
    diff = (pred.float() - target.float()).abs().sum(1)
    loss = (diff + eps) ** q
    return _wmean(loss, weight.float())


def anchor_losses(new, old):
    """Feature cosine + Gram (patch-relation) preservation vs frozen original DINO."""
    B, C, h, w = new.shape
    a = F.normalize(new.float().flatten(2), dim=1)  # B,C,N
    b = F.normalize(old.float().flatten(2), dim=1)
    feat = (1 - (a * b).sum(1)).mean()
    ga = torch.einsum("bcn,bcm->bnm", a, a)
    gb = torch.einsum("bcn,bcm->bnm", b, b)
    gram = (ga - gb).pow(2).mean()
    return feat, gram


class LossManager(nn.Module):
    """Manual weights, optionally times Kendall-style learned uncertainty weights.

    total = sum_i  w_i * exp(-s_i) * L_i + s_i   (uncertainty)
          = sum_i  w_i * L_i                     (manual)
    Anchor terms are never uncertainty-weighted (their weight is a constraint).
    """

    def __init__(self, weights: dict, mode: str = "manual", fixed: tuple[str, ...] = ("anchor_feature", "anchor_gram")):
        super().__init__()
        self.weights = dict(weights)
        self.mode = mode
        self.fixed = set(fixed)
        learn = [k for k in self.weights if k not in self.fixed]
        self.log_vars = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in learn}) if mode == "uncertainty" else None

    def forward(self, losses: dict):
        total = 0.0
        logs = {}
        for name, value in losses.items():
            if value is None or name not in self.weights:
                continue
            w = self.weights[name]
            if w == 0:
                continue
            logs[f"loss/{name}"] = float(value.detach())
            if self.log_vars is not None and name in self.log_vars:
                s = self.log_vars[name].clamp(-2.0, 3.0)
                total = total + w * torch.exp(-s) * value + s
                logs[f"weight/{name}"] = float(w * torch.exp(-s).detach())
            else:
                total = total + w * value
        return total, logs
