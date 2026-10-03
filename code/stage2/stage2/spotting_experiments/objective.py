from __future__ import annotations

import torch
import torch.nn.functional as F


def distribution_loss(logits, target_index, valid, positions, mode, width):
    target_pos = positions.gather(1, target_index[:, None])
    if mode == "exact": soft = F.one_hot(target_index, logits.shape[-1]).float()
    elif mode == "soft_index":
        index = torch.arange(logits.shape[-1], device=logits.device)[None]
        soft = torch.exp(-0.5 * ((index - target_index[:, None]) / width).square())
    elif mode == "soft_normalized": soft = torch.exp(-0.5 * ((positions - target_pos) / width).square())
    else: raise ValueError(mode)
    soft = soft * valid; soft = soft / soft.sum(-1, keepdim=True).clamp_min(1e-8)
    return -(soft * F.log_softmax(logits.float(), -1).masked_fill(~valid, 0)).sum(-1)


def radius_offset_loss(logits, offsets, target_index, valid, positions, radius):
    target_pos = positions.gather(1, target_index[:, None])
    delta = target_pos - positions; positive = (delta.abs() <= radius) & valid
    # Balanced dense classification without any seconds-based radius.
    pos_weight = (valid.sum(-1) - positive.sum(-1)).float() / positive.sum(-1).clamp_min(1)
    bce = F.binary_cross_entropy_with_logits(logits.float(), positive.float(), reduction="none")
    weights = torch.where(positive, pos_weight[:, None], torch.ones_like(bce)) * valid
    classification = (bce * weights).sum(-1) / weights.sum(-1).clamp_min(1)
    regression = (F.smooth_l1_loss(offsets.float(), delta, reduction="none") * positive).sum(-1) / positive.sum(-1).clamp_min(1)
    return classification + regression


def spotting_loss(outputs, batch, cfg):
    valid = batch["time_valid"]; positions = batch["normalized_positions"].float(); mode = cfg["event_target"]
    losses = {}
    for event in ("entry", "collision"):
        if mode == "radius_displacement":
            losses[event] = radius_offset_loss(outputs[f"{event}_logits"], outputs[f"{event}_offsets"], batch[f"{event}_index"], valid, positions, cfg["target_width"])
        else:
            losses[event] = distribution_loss(outputs[f"{event}_logits"], batch[f"{event}_index"], valid, positions, mode, cfg["target_width"])
    side = F.cross_entropy(outputs["side_logits"].float(), batch["entry_side"], reduction="none")
    evasion = F.binary_cross_entropy_with_logits(outputs["evasion_logits"].float(), batch["evasion"].float(), reduction="none")
    pe = torch.softmax(outputs["entry_logits"].float(), -1); pc = torch.softmax(outputs["collision_logits"].float(), -1)
    before = pc.cumsum(-1).roll(1, -1); before[:, 0] = 0; invalid = (pe * before).sum(-1)
    total = .35 * losses["entry"] + .35 * losses["collision"] + .15 * side + .15 * evasion + .05 * invalid
    parts = {"loss_entry": losses["entry"].mean(), "loss_collision": losses["collision"].mean(),
             "loss_side": side.mean(), "loss_evasion": evasion.mean(), "loss_order": invalid.mean()}
    return total.mean(), parts


def constrained_anchors(entry_logits, collision_logits):
    prefix, indices = torch.cummax(entry_logits, -1); collision = (prefix + collision_logits).argmax(-1)
    entry = indices.gather(1, collision[:, None]).squeeze(1); return entry, collision


def decode(outputs, batch):
    entry_i, collision_i = constrained_anchors(outputs["entry_logits"], outputs["collision_logits"])
    positions = batch["normalized_positions"].float()
    ep = positions.gather(1, entry_i[:, None]).squeeze(1); cp = positions.gather(1, collision_i[:, None]).squeeze(1)
    if "entry_offsets" in outputs:
        ep = ep + outputs["entry_offsets"].gather(1, entry_i[:, None]).squeeze(1)
        cp = cp + outputs["collision_offsets"].gather(1, collision_i[:, None]).squeeze(1)
    ep = ep.clamp(0, 1); cp = cp.clamp(0, 1); ep = torch.minimum(ep, cp)
    frames = batch["frame_numbers"].float(); start = frames[:, 0]; end = frames.gather(1, batch["time_valid"].sum(-1)[:, None] - 1).squeeze(1)
    entry_frame = (start + ep * (end - start)).round().long(); collision_frame = (start + cp * (end - start)).round().long()
    return entry_frame, collision_frame
