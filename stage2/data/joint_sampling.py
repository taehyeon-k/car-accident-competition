"""Shared index-only V-JEPA sampling for joint training and inference."""

import torch


def clip_positions(length: int) -> list[torch.Tensor]:
    """16 frames, stride 4, clip-start stride 48, and an end-aligned final clip.

    Short sequences retain the previous rounded-linspace policy (including repeats).
    No source FPS is read or inferred.
    """
    if length < 1:
        raise ValueError("A video must contain at least one frame")
    if length < 61:
        return [torch.linspace(0, length - 1, 16).round().long()]
    starts = sorted(set(list(range(0, length - 60, 48)) + [length - 61]))
    return [start + torch.arange(16) * 4 for start in starts]
