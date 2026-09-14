"""Shared index-only V-JEPA sampling and the temporal training policy."""

import numpy as np
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


TEMPORAL_MODES = ("full_video", "ordinary_crop", "pre_video_entry")
TEMPORAL_DEFAULTS = {
    "full_video": 0.70,
    "ordinary_crop": 0.15,
    "pre_video_entry": 0.15,
}
# A crop shorter than this cannot support the temporal head's receptive field.
MIN_CROP_FRAMES = 16
# The synthetic pre-video ENTRY crop must start within this many seconds of the
# real ENTRY so that the entering vehicle, and therefore LEFT/RIGHT, stays visible.
PRE_VIDEO_ENTRY_MAX_SECONDS = 0.3


def temporal_probabilities(config: dict | None) -> dict:
    """Resolve the three temporal-mode probabilities, defaulting to 70/15/15."""
    merged = dict(TEMPORAL_DEFAULTS)
    for key, value in (config or {}).items():
        if key not in merged:
            raise ValueError(f"Unknown temporal augmentation mode {key!r}")
        merged[key] = float(value)
    return merged


def choose_crop(length, entry, collision, fps, probabilities, rng):
    """Pick one training window and its crop-relative ENTRY/COLLISION indices.

    Returns ``(start, stop, entry_index, collision_index, mode)``. ENTRY and
    COLLISION are always inside the window. ``pre_video_entry`` deliberately
    starts just after the real ENTRY and relabels the first visible frame, which
    is the competition's convention for an entry that predates the clip.
    A mode whose preconditions cannot be met falls back to the full video rather
    than silently producing a degenerate window.
    """
    if not 0 <= entry <= collision < length:
        raise ValueError("Invalid event indices for temporal sampling")
    weights = [probabilities[name] for name in TEMPORAL_MODES]
    mode = TEMPORAL_MODES[int(rng.choice(len(TEMPORAL_MODES), p=weights))]

    if mode == "ordinary_crop":
        # Context on both sides varies independently, so neither event sits at a
        # fixed distance from a boundary and no positional shortcut exists.
        start = int(rng.integers(0, entry + 1))
        stop = int(rng.integers(collision + 1, length + 1))
        if stop - start >= MIN_CROP_FRAMES:
            return start, stop, entry - start, collision - start, mode

    elif mode == "pre_video_entry":
        limit = int(np.floor(PRE_VIDEO_ENTRY_MAX_SECONDS * float(fps))) if fps else 0
        latest = min(entry + max(limit, 0), collision - 1)
        if limit >= 1 and latest > entry:
            start = int(rng.integers(entry + 1, latest + 1))
            stop = int(rng.integers(collision + 1, length + 1))
            if stop - start >= MIN_CROP_FRAMES and collision > start:
                # The first visible frame becomes ENTRY by definition.
                return start, stop, 0, collision - start, mode

    return 0, length, entry, collision, "full_video"
