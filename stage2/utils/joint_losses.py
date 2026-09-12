"""Metric-aligned losses and decoding for the single-stage model."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _event_loss(logits, target, valid, seconds, sigma_seconds=0.1):
    distance = seconds - seconds.gather(1, target[:, None])
    soft = torch.exp(-0.5 * (distance / sigma_seconds).square()) * valid
    soft = soft / soft.sum(-1, keepdim=True).clamp_min(1e-8)
    logp = F.log_softmax(logits.float(), dim=-1).masked_fill(~valid, 0)
    distribution = -(soft * logp).sum(-1)
    predicted_cdf = torch.softmax(logits.float(), -1).cumsum(-1)
    target_cdf = soft.cumsum(-1)
    gaps = torch.diff(seconds, dim=-1)
    gap_valid = valid[:, 1:] & valid[:, :-1]
    step = (gaps * gap_valid).sum(-1) / gap_valid.sum(-1).clamp_min(1)
    cdf = ((predicted_cdf - target_cdf).abs() * valid).sum(-1) * step / 0.3
    return distribution + 0.05 * cdf, distribution, cdf


def joint_loss(outputs, batch, reduction="mean"):
    valid = batch["time_valid"].bool()
    entry, entry_dist, entry_cdf = _event_loss(
        outputs["entry_logits"],
        batch["entry_index"],
        valid,
        batch["frame_seconds"].float(),
    )
    supervised = batch.get(
        "entry_supervised", torch.ones_like(entry, dtype=torch.bool)
    ).to(entry.dtype)
    entry = entry * supervised
    entry_dist = entry_dist * supervised
    entry_cdf = entry_cdf * supervised
    collision, collision_dist, collision_cdf = _event_loss(
        outputs["collision_logits"],
        batch["collision_index"],
        valid,
        batch["frame_seconds"].float(),
    )
    side = F.cross_entropy(
        outputs["side_logits"].float(), batch["entry_side"], reduction="none"
    )
    evasion = F.binary_cross_entropy_with_logits(
        outputs["evasion_logits"].float(), batch["evasion"].float(), reduction="none"
    )
    p_entry = torch.softmax(outputs["entry_logits"].float(), -1)
    p_collision = torch.softmax(outputs["collision_logits"].float(), -1)
    collision_before = p_collision.cumsum(-1).roll(1, dims=-1)
    collision_before[:, 0] = 0
    invalid = (p_entry * collision_before).sum(-1)
    total = 0.35 * entry + 0.35 * collision + 0.15 * side + 0.15 * evasion
    total = total + 0.05 * invalid
    parts = {
        "entry": entry_dist.detach(),
        "collision": collision_dist.detach(),
        "entry_cdf": entry_cdf.detach(),
        "collision_cdf": collision_cdf.detach(),
        "side": side.detach(),
        "evasion": evasion.detach(),
        "invalid_order": invalid.detach(),
    }
    if reduction == "mean":
        return total.mean(), {k: v.mean() for k, v in parts.items()}
    if reduction != "none":
        raise ValueError("Supported reductions: mean, none")
    return total, parts


def constrained_decode(entry_logits: torch.Tensor, collision_logits: torch.Tensor):
    """Return the maximum-score pair satisfying entry <= collision in O(T)."""
    prefix_score, prefix_index = torch.cummax(entry_logits, dim=-1)
    joint = prefix_score + collision_logits
    collision = joint.argmax(-1)
    entry = prefix_index.gather(1, collision[:, None]).squeeze(1)
    return entry, collision
