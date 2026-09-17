from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stage3.flow import build_flow_estimator
from stage3.flow.sea_raft import resize_frames
from stage3.geometry import build_motion_features
from stage3.utils.checkpoint import atomic_gzip_save

from .adapters.base import ClipRecord
from .timing import decode_external_training_video


FEATURE_CODE_VERSION = "stage3-v1.2-geometry-2-uint8-gzip"


def quantize_motion(motion: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Robust per-channel uint8 cache encoding with explicit inverse transform."""
    low = np.quantile(motion, 0.001, axis=(0, 2, 3)).astype(np.float32)
    high = np.quantile(motion, 0.999, axis=(0, 2, 3)).astype(np.float32)
    scale = np.maximum((high - low) / 255.0, 1e-7)
    encoded = np.clip(np.rint((motion - low[None, :, None, None]) / scale[None, :, None, None]), 0, 255).astype(np.uint8)
    return torch.from_numpy(encoded), torch.from_numpy(low), torch.from_numpy(scale)


def dequantize_motion(value: dict) -> torch.Tensor:
    if "motion" in value:  # schema-1 development caches
        return value["motion"].float()
    return value["motion_q"].float() * value["motion_scale"][None, :, None, None] + value["motion_offset"][None, :, None, None]


def cache_key(cfg: dict[str, Any]) -> str:
    identity = {
        "feature_code": FEATURE_CODE_VERSION,
        "flow": {k: cfg["flow"].get(k) for k in ("backend", "version", "working_size", "checkpoint_sha256")},
        "calibration": cfg["calibration"],
        "canonical_grid": cfg["geometry"]["canonical_size"],
        "foe_version": 1,
        "rho_version": 1,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]


def cache_record(record: ClipRecord, cfg: dict[str, Any], output: str | Path, max_frames: int | None = None, device: str | None = None, estimator=None) -> dict:
    decoded = decode_external_training_video(
        record.video_path,
        hz=cfg["timing"]["target_hz"],
        max_selection_error=cfg["timing"]["max_selection_error"],
        large_gap_seconds=cfg["timing"]["large_gap_seconds"],
        max_frames=max_frames,
        start_time=record.metadata.get("segment_start_time"),
        end_time=record.metadata.get("segment_end_time"),
    )
    height, width = cfg["flow"]["working_size"]
    frames = resize_frames(decoded.frames, (height, width))
    estimator = estimator or build_flow_estimator(cfg["flow"], device)
    flows, confidence = estimator.estimate_sequence(frames)
    motion, physics, geometry_meta = build_motion_features(
        flows, confidence, decoded.actual_times, cfg["calibration"], tuple(cfg["geometry"]["canonical_size"]), frames[0]
    )
    raw_signals = None
    if record.signals is not None:
        keep = (record.signals.t >= decoded.actual_times[0] - 2.0) & (record.signals.t <= decoded.actual_times[-1] + 2.0)
        raw_signals = {
            key: None if value is None else torch.from_numpy(value[keep])
            for key, value in record.signals.__dict__.items()
        }
    motion_q, motion_offset, motion_scale = quantize_motion(motion)
    value = {
        "schema": 2,
        "cache_key": cache_key(cfg),
        "clip_id": record.clip_id,
        "motion_q": motion_q,
        "motion_offset": motion_offset,
        "motion_scale": motion_scale,
        "physics": torch.from_numpy(physics),
        "actual_times": torch.from_numpy(decoded.actual_times),
        "target_times": torch.from_numpy(decoded.target_times),
        "time_valid": torch.from_numpy(decoded.valid),
        "signals": raw_signals,
        "metadata": {**record.metadata, **{key: value.tolist() for key, value in geometry_meta.items()}},
    }
    atomic_gzip_save(value, output)
    return {"frames": len(frames), "cache_key": value["cache_key"], "path": str(output)}
