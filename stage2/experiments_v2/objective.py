"""Losses and decoding. Every target is built in frame/order space only."""
from __future__ import annotations

import torch
import torch.nn.functional as F

TASK_WEIGHTS = (0.35, 0.35, 0.15, 0.15)


def event_distribution_loss(logits, target_index, valid, positions, mode, width):
    """Dense event supervision (prompt v2 sections 8 and 10).

    ``exact`` is one-hot over sampled positions. ``soft_index`` decays with
    distance in sampled-frame count. ``soft_normalized`` decays with distance in
    normalized clip position, which keeps the effective width comparable across
    different sampling cadences. ``width`` is never expressed in seconds.
    """
    if mode == "exact":
        soft = F.one_hot(target_index, logits.shape[-1]).float()
    elif mode == "soft_index":
        index = torch.arange(logits.shape[-1], device=logits.device)[None]
        soft = torch.exp(-0.5 * ((index - target_index[:, None]) / width).square())
    elif mode == "soft_normalized":
        target_pos = positions.gather(1, target_index[:, None])
        soft = torch.exp(-0.5 * ((positions - target_pos) / width).square())
    elif mode == "soft_laplace":
        target_pos = positions.gather(1, target_index[:, None])
        soft = torch.exp(-(positions - target_pos).abs() / width)
    else:
        raise ValueError(mode)
    soft = soft * valid
    soft = soft / soft.sum(-1, keepdim=True).clamp_min(1e-8)
    return -(soft * F.log_softmax(logits.float(), -1).masked_fill(~valid, 0)).sum(-1)


def state_labels(target_index, valid, radius):
    """0 = before, 1 = boundary, 2 = after, in sampled-position space."""
    t = valid.shape[1]
    index = torch.arange(t, device=valid.device)[None]
    delta = index - target_index[:, None]
    labels = torch.full_like(delta, 1)
    labels = torch.where(delta < -radius, torch.zeros_like(labels), labels)
    labels = torch.where(delta > radius, torch.full_like(labels, 2), labels)
    return labels


def state_aux_loss(logits, target_index, valid, radius):
    """Event-state auxiliary supervision (prompt v2 section 9)."""
    labels = state_labels(target_index, valid, radius)
    loss = F.cross_entropy(logits.float().transpose(1, 2), labels, reduction="none")
    return (loss * valid).sum(-1) / valid.sum(-1).clamp_min(1)


def binned_distribution(logits, valid, positions, bins):
    """Map a per-frame distribution back to shared original-frame coordinates.

    Probability mass is scatter-added into ``bins`` uniform bins of normalized
    clip position, which is a monotone function of the original frame number.
    Two views of the same clip therefore land on the same axis even when their
    sampling densities differ.
    """
    probs = torch.softmax(logits.float(), -1) * valid
    index = (positions.clamp(0, 1) * (bins - 1)).round().long()
    out = torch.zeros(probs.shape[0], bins, device=probs.device, dtype=probs.dtype)
    out.scatter_add_(1, index, probs)
    return out / out.sum(-1, keepdim=True).clamp_min(1e-8)


def consistency_loss(out_a, batch_a, out_b, batch_b, bins):
    """Jensen-Shannon agreement between two temporal views (prompt v2 section 12)."""
    total = 0.0
    for event in ("entry", "collision"):
        pa = binned_distribution(out_a[f"{event}_logits"], batch_a["time_valid"],
                                 batch_a["normalized_positions"].float(), bins)
        pb = binned_distribution(out_b[f"{event}_logits"], batch_b["time_valid"],
                                 batch_b["normalized_positions"].float(), bins)
        m = (0.5 * (pa + pb)).clamp_min(1e-8)
        js = 0.5 * (pa * (pa.clamp_min(1e-8) / m).log()).sum(-1) \
            + 0.5 * (pb * (pb.clamp_min(1e-8) / m).log()).sum(-1)
        total = total + js
    return total / 2


def order_penalty(entry_logits, collision_logits):
    """Discourage probability mass where COLLISION precedes ENTRY."""
    pe = torch.softmax(entry_logits.float(), -1)
    pc = torch.softmax(collision_logits.float(), -1)
    before = pc.cumsum(-1).roll(1, -1)
    before[:, 0] = 0
    return (pe * before).sum(-1)


def spotting_loss(outputs, batch, view, cfg):
    valid = view["time_valid"]
    positions = view["normalized_positions"].float()
    parts = {}

    if "entry_position" in outputs:  # pooled-regressor baseline
        losses = {}
        for event in ("entry", "collision"):
            target = positions.gather(1, view[f"{event}_index"][:, None]).squeeze(1)
            losses[event] = F.smooth_l1_loss(outputs[f"{event}_position"].float(), target,
                                             reduction="none")
        invalid = torch.zeros_like(losses["entry"])
    else:
        mode, width = cfg["event_target"], cfg["target_width"]
        losses = {e: event_distribution_loss(outputs[f"{e}_logits"], view[f"{e}_index"],
                                             valid, positions, mode, width)
                  for e in ("entry", "collision")}
        invalid = order_penalty(outputs["entry_logits"], outputs["collision_logits"])

    side = F.cross_entropy(outputs["side_logits"].float(), batch["entry_side"], reduction="none")
    evasion = F.binary_cross_entropy_with_logits(outputs["evasion_logits"].float(),
                                                 batch["evasion"].float(), reduction="none")
    we, wc, ws, wv = TASK_WEIGHTS
    total = we * losses["entry"] + wc * losses["collision"] + ws * side + wv * evasion \
        + cfg.get("order_weight", 0.05) * invalid
    parts.update(loss_entry=losses["entry"].mean(), loss_collision=losses["collision"].mean(),
                 loss_side=side.mean(), loss_evasion=evasion.mean(), loss_order=invalid.mean())

    if cfg.get("state_aux", False) and "entry_state_logits" in outputs:
        radius = int(cfg.get("state_radius", 3))
        aux = state_aux_loss(outputs["entry_state_logits"], view["entry_index"], valid, radius) \
            + state_aux_loss(outputs["collision_state_logits"], view["collision_index"], valid, radius)
        total = total + cfg.get("state_aux_weight", 0.1) * aux
        parts["loss_state_aux"] = aux.mean()

    return total.mean(), parts


def constrained_anchors(entry_logits, collision_logits):
    """Pick the best ENTRY <= COLLISION pair in one pass."""
    prefix, indices = torch.cummax(entry_logits, -1)
    collision = (prefix + collision_logits).argmax(-1)
    entry = indices.gather(1, collision[:, None]).squeeze(1)
    return entry, collision


def decode(outputs, view):
    """Return original-frame indices. The frame mapping is never lost."""
    positions = view["normalized_positions"].float()
    if "entry_position" in outputs:
        ep = outputs["entry_position"].float().clamp(0, 1)
        cp = outputs["collision_position"].float().clamp(0, 1)
    else:
        entry_i, collision_i = constrained_anchors(outputs["entry_logits"],
                                                   outputs["collision_logits"])
        ep = positions.gather(1, entry_i[:, None]).squeeze(1)
        cp = positions.gather(1, collision_i[:, None]).squeeze(1)
    ep = torch.minimum(ep.clamp(0, 1), cp.clamp(0, 1))
    frames = view["frame_numbers"].float()
    start = frames[:, 0]
    end = frames.gather(1, view["time_valid"].sum(-1)[:, None] - 1).squeeze(1)
    return (start + ep * (end - start)).round().long(), (start + cp * (end - start)).round().long()
