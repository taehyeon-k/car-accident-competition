"""Losses that exactly mirror the fixed Stage 2 objectives."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def coarse_loss(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute 0.35/0.35/0.15/0.15 coarse multi-task loss."""
    entry = F.cross_entropy(outputs["entry_logits"], batch["entry_bin"])
    collision = F.cross_entropy(outputs["collision_logits"], batch["collision_bin"])
    direction = F.cross_entropy(outputs["direction_logits"], batch["entry_side"])
    evasion = F.binary_cross_entropy_with_logits(
        outputs["evasion_logits"].squeeze(-1), batch["evasion"].float()
    )

    total = 0.35 * entry + 0.35 * collision + 0.15 * direction + 0.15 * evasion
    metrics = {
        "entry": entry.detach(),
        "collision": collision.detach(),
        "direction": direction.detach(),
        "evasion": evasion.detach(),
    }
    return total, metrics


def fine_loss(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute soft-frame CE + 0.3 state BCE + 0.2 expected-position loss."""
    is_collision = batch["event_type"][:, None].bool()
    frame_logits = torch.where(is_collision, outputs["collision_logits"], outputs["entry_logits"])
    state_logits = torch.where(
        is_collision, outputs["collision_state_logits"], outputs["entry_state_logits"]
    )

    valid = batch["time_valid"].bool()
    target_index = batch["event_local_index"]
    positions = torch.arange(frame_logits.shape[1], device=frame_logits.device)[None]

    # sigma = 1 frame. The target distribution has support only on valid positions.
    soft_target = torch.exp(-0.5 * (positions - target_index[:, None]).float().square())
    soft_target = soft_target * valid
    soft_target = soft_target / soft_target.sum(dim=-1, keepdim=True)
    frame = -(soft_target * F.log_softmax(frame_logits, dim=-1)).sum(dim=-1).mean()

    state_target = (positions >= target_index[:, None]).float()
    state = F.binary_cross_entropy_with_logits(state_logits[valid], state_target[valid])

    predicted_position = (torch.softmax(frame_logits, dim=-1) * positions).sum(dim=-1)
    position = F.smooth_l1_loss(predicted_position, target_index.float())

    total = frame + 0.3 * state + 0.2 * position
    metrics = {
        "frame": frame.detach(),
        "state": state.detach(),
        "position": position.detach(),
    }
    return total, metrics
