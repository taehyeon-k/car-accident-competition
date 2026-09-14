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


# ---------------------------------------------------------------------------
# Training memory cap
#
# This is deliberately NOT a fourth augmentation mode. The semantic policy above
# decides what the model should learn from; this step only bounds how large an
# autograd graph one sample may build, and it runs afterwards on the window that
# policy produced. With max_frames = 512 it is comfortably above the dataset's
# longest ENTRY->COLLISION interval (147 frames), so it never has to invent,
# move or drop a label to satisfy the limit.
# ---------------------------------------------------------------------------

DEFAULT_MAX_TRAIN_FRAMES = 512


def max_train_frames(config: dict | None) -> int | None:
    """Resolve ``training_memory.max_frames``: a positive integer, or None."""
    value = (config or {}).get("max_frames", DEFAULT_MAX_TRAIN_FRAMES)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "training_memory.max_frames must be null or a positive integer"
        )
    return value


def enforce_max_train_frames(
    start, stop, entry_index, collision_index, mode, max_frames, rng
):
    """Bound a semantic window to ``max_frames`` without changing its labels.

    Returns ``(start, stop, entry_index, collision_index, applied)`` in the same
    coordinates ``choose_crop`` uses. Windows already within the limit are
    returned untouched, so short clips keep their natural length rather than
    being padded or trimmed to a fixed size.
    """
    length = stop - start
    if max_frames is None or length <= max_frames:
        return start, stop, entry_index, collision_index, False

    if mode == "pre_video_entry":
        # The synthetic convention is "ENTRY is the first visible frame". Moving
        # the window start would silently destroy it, so the start is pinned and
        # only the tail is trimmed. For this mode entry_index is 0, so an
        # over-long span and an out-of-window COLLISION are the same condition;
        # it is reported with the message specific to the convention at risk.
        if collision_index >= max_frames:
            raise ValueError(
                "max_train_frames is too short to preserve the pre-video "
                "ENTRY -> COLLISION interval."
            )
        return start, start + max_frames, entry_index, collision_index, True

    span = collision_index - entry_index + 1
    if span > max_frames:
        raise ValueError(
            f"max_train_frames ({max_frames}) is shorter than this sample's "
            f"ENTRY->COLLISION span ({span} frames); it cannot be capped without "
            "corrupting the labels"
        )

    # Sample uniformly over every window that still contains both events, so no
    # fixed offset, centring or edge alignment is learnable from position alone.
    minimum_start = max(0, collision_index - max_frames + 1)
    maximum_start = min(entry_index, length - max_frames)
    if minimum_start > maximum_start:
        raise ValueError(
            f"No {max_frames}-frame window contains both events for this sample"
        )
    memory_start = int(rng.integers(minimum_start, maximum_start + 1))
    return (
        start + memory_start,
        start + memory_start + max_frames,
        entry_index - memory_start,
        collision_index - memory_start,
        True,
    )
