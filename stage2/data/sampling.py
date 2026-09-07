"""Frame-order, coarse-bin, and exact-frame-window invariants."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np

COARSE_T = 32
FINE_MAX_K = 64
FINE_SLIDE_STRIDE = 32

def frame_id(path: str | Path) -> int:
    numbers = re.findall(r"\d+", Path(path).stem)
    if not numbers:
        raise ValueError(f"No integer frame id in {path!s}")
    return int(numbers[-1])

def sort_frame_paths(paths: Iterable[str | Path]) -> tuple[list[Path], list[int]]:
    ordered = sorted((Path(p) for p in paths), key=frame_id)
    ids = [frame_id(p) for p in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("Frame IDs must be unique within a sample")
    return ordered, ids

@dataclass(frozen=True)
class CoarseBins:
    candidate_start: np.ndarray; candidate_end: np.ndarray
    native_start: np.ndarray; native_end: np.ndarray
    representative_native_pos: np.ndarray; representative_frame_id: np.ndarray
    valid: np.ndarray; candidate_positions: np.ndarray

def build_coarse_bins(candidate_native_positions: list[int], frame_ids: list[int], training: bool, rng: np.random.Generator | None = None) -> CoarseBins:
    """Build exactly 32 half-open bins. Empty slots are invalid, never targets."""
    candidate = np.asarray(candidate_native_positions, dtype=np.int64)
    if candidate.size == 0: raise ValueError("A sample needs at least one frame")
    if np.any(np.diff(candidate) <= 0): raise ValueError("Candidate positions must be strictly ordered")
    rng = rng or np.random.default_rng()
    edges = np.floor(np.arange(COARSE_T + 1) * len(candidate) / COARSE_T).astype(int)
    starts, ends = edges[:-1], edges[1:]
    valid = ends > starts
    reps = np.full(COARSE_T, candidate[-1], dtype=np.int64)
    for j in np.flatnonzero(valid):
        reps[j] = candidate[rng.integers(starts[j], ends[j])] if training else candidate[(starts[j] + ends[j] - 1) // 2]
    native_start = np.full(COARSE_T, -1, dtype=np.int64); native_end = native_start.copy()
    native_start[valid] = candidate[starts[valid]]
    native_end[valid] = candidate[ends[valid] - 1] + 1
    rep_ids = np.asarray([frame_ids[p] for p in reps], dtype=np.int64)
    return CoarseBins(starts, ends, native_start, native_end, reps, rep_ids, valid, candidate)

def event_bin(event_native_pos: int, bins: CoarseBins) -> int:
    for j in np.flatnonzero(bins.valid):
        selected = bins.candidate_positions[bins.candidate_start[j]:bins.candidate_end[j]]
        if event_native_pos in selected: return int(j)
    raise ValueError("Event was not forced into the candidate sequence")

def event_preserving_crop(length: int, entry: int, collision: int, rng: np.random.Generator) -> np.ndarray:
    if not (0 <= entry <= collision < length): raise ValueError("Invalid event positions")
    return np.arange(rng.integers(0, entry + 1), rng.integers(collision, length) + 1, dtype=np.int64)

def temporal_rate_augment(native_positions: np.ndarray, entry: int, collision: int, native_fps: float | None, rng: np.random.Generator) -> np.ndarray:
    keep = np.ones(len(native_positions), dtype=bool)
    if native_fps:
        target = min(float(native_fps), rng.uniform(10, 30)); stride = max(1, int(round(native_fps / target)))
        keep[:] = False; keep[::stride] = True
    keep &= rng.random(len(native_positions)) >= 0.05
    keep[(native_positions == entry) | (native_positions == collision)] = True
    return native_positions[keep]

def recover_region(bins: CoarseBins, predicted_bin: int, radius: int = 2) -> np.ndarray:
    selected = [j for j in range(max(0, predicted_bin-radius), min(COARSE_T, predicted_bin+radius+1)) if bins.valid[j]]
    if not selected: raise ValueError("Predicted only invalid coarse bins")
    return np.arange(bins.native_start[selected[0]], bins.native_end[selected[-1]], dtype=np.int64)

def sliding_windows(region: np.ndarray) -> list[np.ndarray]:
    if len(region) <= FINE_MAX_K: return [region]
    starts = list(range(0, len(region)-FINE_MAX_K+1, FINE_SLIDE_STRIDE))
    if starts[-1] != len(region)-FINE_MAX_K: starts.append(len(region)-FINE_MAX_K)
    return [region[s:s+FINE_MAX_K] for s in starts]
