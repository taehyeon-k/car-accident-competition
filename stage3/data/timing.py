from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import warnings

import numpy as np


@dataclass
class DecodedVideo:
    frames: list[np.ndarray]
    actual_times: np.ndarray
    target_times: np.ndarray
    valid: np.ndarray
    source_indices: np.ndarray


def nearest_grid_indices(
    pts_times: np.ndarray,
    hz: float = 10.0,
    max_error: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map an exact target grid to nearest monotonic decoded PTS."""
    times = np.asarray(pts_times, dtype=np.float64)
    if len(times) == 0 or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("PTS timestamps must be finite and strictly increasing")
    target = np.arange(0.0, times[-1] - times[0] + 1e-9, 1.0 / hz) + times[0]
    right = np.searchsorted(times, target, side="left").clip(0, len(times) - 1)
    left = (right - 1).clip(0, len(times) - 1)
    choose_left = np.abs(times[left] - target) <= np.abs(times[right] - target)
    index = np.where(choose_left, left, right)
    valid = np.abs(times[index] - target) <= max_error
    return index.astype(np.int64), target, valid


def _decoded_frames(path: str | Path, start_time: float | None = None, end_time: float | None = None) -> Iterator[tuple[np.ndarray, float]]:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is required for Stage 3 video decoding") from error
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if start_time is not None and start_time > 0:
            container.seek(int(start_time / float(stream.time_base)), stream=stream, backward=True)
        last = -np.inf
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"Decoded frame has no PTS: {path}")
            timestamp = float(frame.pts * stream.time_base)
            if timestamp <= last:
                if timestamp == last:
                    continue
                raise ValueError(f"Non-monotonic PTS in {path}")
            last = timestamp
            if start_time is not None and timestamp < start_time - 0.1:
                continue
            if end_time is not None and timestamp > end_time + 0.1:
                break
            yield frame.to_ndarray(format="rgb24"), timestamp


def decode_external_training_video(
    path: str | Path,
    hz: float = 10.0,
    max_selection_error: float = 0.04,
    large_gap_seconds: float = 0.2,
    max_frames: int | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> DecodedVideo:
    requested_end = end_time
    if max_frames is not None:
        limited_end = (start_time or 0.0) + max(max_frames - 1, 0) / hz + max_selection_error
        requested_end = limited_end if requested_end is None else min(requested_end, limited_end)
    decoded = list(_decoded_frames(path, start_time, requested_end))
    if not decoded:
        raise ValueError(f"No frames decoded from {path}")
    frames, times = zip(*decoded)
    times_array = np.asarray(times)
    gaps = np.diff(times_array)
    if np.any(gaps > large_gap_seconds):
        warnings.warn(f"{path} contains {int((gaps > large_gap_seconds).sum())} large PTS gaps")
    grid_start = float(start_time) if start_time is not None else float(times_array[0])
    grid_end = float(end_time) if end_time is not None else float(times_array[-1])
    if max_frames is not None:
        grid_end = min(grid_end, grid_start + (max_frames - 1) / hz)
    target = np.arange(np.ceil(grid_start * hz) / hz, grid_end + 1e-9, 1.0 / hz)
    right = np.searchsorted(times_array, target, side="left").clip(0, len(times_array) - 1)
    left = (right - 1).clip(0, len(times_array) - 1)
    index = np.where(np.abs(times_array[left] - target) <= np.abs(times_array[right] - target), left, right)
    valid = np.abs(times_array[index] - target) <= max_selection_error
    if max_frames is not None:
        index, target, valid = index[:max_frames], target[:max_frames], valid[:max_frames]
    return DecodedVideo(
        [frames[int(i)] for i in index], times_array[index], target, valid, index
    )


def decode_dacon_stage3_video(path: str | Path, max_frames: int | None = None) -> DecodedVideo:
    """Decode every frame 1:1; private DACON sample indices ignore PTS."""
    frames, times = [], []
    for frame, timestamp in _decoded_frames(path):
        frames.append(frame)
        times.append(timestamp)
        if max_frames is not None and len(frames) >= max_frames:
            break
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    n = len(frames)
    target = np.arange(n, dtype=np.float64) * 0.1
    return DecodedVideo(
        frames, np.asarray(times), target, np.ones(n, bool), np.arange(n, dtype=np.int64)
    )
