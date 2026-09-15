"""Metric-aligned losses and decoding for the single-stage model."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _event_loss(logits, target, valid, seconds, sigma_seconds=0.1):
    if sigma_seconds <= 0:
        raise ValueError("Gaussian sigma must be positive")
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


def joint_loss(
    outputs,
    batch,
    reduction="mean",
    entry_sigma_seconds=0.15,
    collision_sigma_seconds=0.10,
):
    valid = batch["time_valid"].bool()
    entry, entry_dist, entry_cdf = _event_loss(
        outputs["entry_logits"],
        batch["entry_index"],
        valid,
        batch["frame_seconds"].float(),
        sigma_seconds=entry_sigma_seconds,
    )
    collision, collision_dist, collision_cdf = _event_loss(
        outputs["collision_logits"],
        batch["collision_index"],
        valid,
        batch["frame_seconds"].float(),
        sigma_seconds=collision_sigma_seconds,
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
        "loss_entry": entry.detach(),
        "loss_collision": collision.detach(),
        "loss_entry_side": side.detach(),
        "loss_evasion_space": evasion.detach(),
        "loss_invalid_order": invalid.detach(),
    }
    if reduction == "mean":
        return total.mean(), {k: v.mean() for k, v in parts.items()}
    if reduction != "none":
        raise ValueError("Supported reductions: mean, none")
    return total, parts


def constrained_decode(
    entry_logits: torch.Tensor,
    collision_logits: torch.Tensor,
    max_span_frames: int | None = None,
):
    """Return the maximum-score pair satisfying entry <= collision.

    With ``max_span_frames`` the pair must also satisfy
    ``collision - entry <= max_span_frames``, which stops ENTRY landing seconds
    before an otherwise correct COLLISION. Ties resolve exactly as in the
    unconstrained path - the latest ENTRY among equal scores (``torch.cummax``)
    and the earliest COLLISION (``argmax``) - so a window covering the whole clip
    reproduces it exactly. bf16 logits produce exact ties, so this matters.
    """
    if max_span_frames is None:
        prefix_score, prefix_index = torch.cummax(entry_logits, dim=-1)
        joint = prefix_score + collision_logits
        collision = joint.argmax(-1)
        entry = prefix_index.gather(1, collision[:, None]).squeeze(1)
        return entry, collision
    length = entry_logits.shape[-1]
    span = min(int(max_span_frames), length - 1)
    padded = torch.nn.functional.pad(entry_logits, (span, 0), value=-float("inf"))
    # windows[:, c, k] holds the ENTRY score for frame c - span + k.
    windows = padded.unfold(-1, span + 1, 1)
    back = windows.flip(-1).argmax(-1)  # frames before c; latest wins ties
    joint = windows.max(-1).values + collision_logits
    collision = joint.argmax(-1)
    positions = torch.arange(length, device=entry_logits.device)
    entry = (positions[None, :] - back).gather(1, collision[:, None]).squeeze(1)
    return entry, collision
