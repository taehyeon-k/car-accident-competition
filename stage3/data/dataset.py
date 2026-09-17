from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from stage3.utils.config import read_jsonl

from .augment import horizontal_flip
from .adapters.base import Signals
from .targets import make_targets
from .cache import dequantize_motion
from stage3.utils.checkpoint import load_artifact


TARGET_NAMES = (
    "a_long_s1", "a_long_s2", "a_dvdt_s1", "a_dvdt_s2", "speed",
    "steering_angle", "yaw_rate_aux", "stopped", "valid_accel",
    "valid_accel_speed", "valid_speed", "valid_steer", "valid_yaw",
)


class CachedMotionDataset(Dataset):
    def __init__(self, manifest: str, crop_frames: int = 96, training: bool = False, seed: int = 42, flip_probability: float = 0.5, target_cfg: dict | None = None, event_fraction: float = 0.5, event_position_margin: int = 8):
        self.rows = read_jsonl(manifest)
        self.crop_frames = crop_frames
        self.training = training
        self.seed = seed
        self.flip_probability = flip_probability
        self.target_cfg = target_cfg or {}
        self.event_fraction = float(event_fraction)
        self.event_position_margin = int(event_position_margin)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        cache = load_artifact(row["cache_path"])
        if cache.get("schema") not in {1, 2} or cache.get("signals") is None:
            raise ValueError(f"Invalid training cache: {row['cache_path']}")
        raw = cache["signals"]
        signals = Signals(**{
            key: None if value is None else value.numpy()
            for key, value in raw.items()
        })
        generated = make_targets(signals, cache["actual_times"].numpy(), self.target_cfg)
        motion_all = dequantize_motion(cache)
        length = len(motion_all)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        if self.training and length > self.crop_frames:
            event = np.flatnonzero(
                (np.abs(generated["a_long_s1"]) > 0.25)
                | (np.abs(generated["steering_angle"]) > 5.0)
                | (np.r_[False, np.diff(generated["stopped"]) != 0])
            )
            if len(event) and rng.random() < self.event_fraction:
                chosen = int(rng.choice(event))
                low = min(self.event_position_margin, self.crop_frames - 1)
                high = max(low + 1, self.crop_frames - self.event_position_margin)
                position = int(rng.integers(low, high))
                start = int(np.clip(chosen - position, 0, length - self.crop_frames))
            else:
                start = int(rng.integers(0, length - self.crop_frames + 1))
            stop = start + self.crop_frames
        else:
            start, stop = 0, length
        targets = {name: torch.from_numpy(generated[name][start:stop]) for name in TARGET_NAMES}
        motion, physics = motion_all[start:stop], cache["physics"][start:stop]
        if self.training and rng.random() < self.flip_probability:
            motion, physics, targets = horizontal_flip(motion, physics, targets)
        return {
            "motion": motion.float(), "physics": physics.float(), **targets,
            "time_valid": cache["time_valid"][start:stop].bool(), "clip_id": cache["clip_id"],
        }


def motion_collate(items: list[dict]) -> dict:
    max_time = max(len(item["time_valid"]) for item in items)
    output: dict = {"clip_id": [item["clip_id"] for item in items]}
    for name, first in items[0].items():
        if name == "clip_id":
            continue
        padded = first.new_zeros((len(items), max_time, *first.shape[1:]))
        for i, item in enumerate(items):
            padded[i, : len(item[name])] = item[name]
            if name in {"motion", "physics"} and len(item[name]) < max_time:
                padded[i, len(item[name]) :] = item[name][-1]
        output[name] = padded
    return output
