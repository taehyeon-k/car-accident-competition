"""Cached-feature dataset for the frozen single-stage visual encoders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from stage2.utils.utils import read_manifest


class JointFeatureDataset(Dataset):
    def __init__(
        self, manifest, feature_dir, training=False, seed=42, censor_probability=0.15
    ):
        self.rows = read_manifest(manifest)
        self.feature_dir = Path(feature_dir)
        self.training = training
        self.seed = seed
        self.censor_probability = censor_probability
        self.epoch = torch.zeros((), dtype=torch.long).share_memory_()

    def set_epoch(self, epoch: int) -> None:
        self.epoch.fill_(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        cache = torch.load(
            self.feature_dir / f"{row['sample_id']}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if cache.get("schema") != 1 or cache["sample_id"] != row["sample_id"]:
            raise ValueError("Incompatible joint feature cache")
        frame_ids = cache["frame_ids"].long()
        entry = int((frame_ids == int(row["entry_frame"])).nonzero()[0])
        collision = int((frame_ids == int(row["collision_frame"])).nonzero()[0])
        start = 0
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(self.epoch), index])
        )
        if (
            self.training
            and collision > entry
            and rng.random() < self.censor_probability
        ):
            proposed = int(rng.integers(entry + 1, collision + 1))
            support = cache["global_support"]
            usable = (support[:, 0] >= proposed) & (support[:, 1] < len(frame_ids))
            if usable.any():
                start = proposed
        stop = len(frame_ids)
        support = cache["global_support"]
        keep = (support[:, 0] >= start) & (support[:, 1] < stop)
        if not keep.any():
            keep = torch.ones(len(support), dtype=torch.bool)
        length = stop - start
        fps = float(row.get("native_fps") or 1.0)
        side = row["entry_side"]
        return {
            "scene_features": cache["scene_features"][start:stop],
            "roi_features": cache["roi_features"][start:stop],
            "geometry": cache["geometry"][start:stop].float(),
            "object_valid": cache["object_valid"][start:stop].bool(),
            "time_valid": torch.ones(length, dtype=torch.bool),
            "local_time": torch.linspace(0, 1, length),
            "frame_seconds": torch.arange(length, dtype=torch.float32) / fps,
            "global_features": cache["global_features"][keep],
            "global_time": (
                (cache["global_anchor"][keep] - start) / max(1, length - 1)
            ).float(),
            "global_valid": torch.ones(int(keep.sum()), dtype=torch.bool),
            "entry_index": max(0, entry - start),
            "entry_supervised": start == 0,
            "collision_index": collision - start,
            "entry_side": int(side == "RIGHT") if isinstance(side, str) else int(side),
            "evasion": float(row["evasion_space"]),
            "frame_ids": frame_ids[start:stop],
            "sample_id": row["sample_id"],
            "source_id": row["source_id"],
        }


def joint_collate(items):
    """Pad native and global time axes independently."""
    batch = len(items)
    max_time = max(len(x["time_valid"]) for x in items)
    max_global = max(len(x["global_valid"]) for x in items)
    output = {
        "scene_features": torch.zeros(
            batch, max_time, 7, 768, dtype=items[0]["scene_features"].dtype
        ),
        "roi_features": torch.zeros(
            batch, max_time, 12, 768, dtype=items[0]["roi_features"].dtype
        ),
        "geometry": torch.zeros(batch, max_time, 12, 13),
        "object_valid": torch.zeros(batch, max_time, 12, dtype=torch.bool),
        "time_valid": torch.zeros(batch, max_time, dtype=torch.bool),
        "local_time": torch.zeros(batch, max_time),
        "frame_seconds": torch.zeros(batch, max_time),
        "global_features": torch.zeros(
            batch, max_global, 1024, dtype=items[0]["global_features"].dtype
        ),
        "global_time": torch.zeros(batch, max_global),
        "global_valid": torch.zeros(batch, max_global, dtype=torch.bool),
    }
    for i, item in enumerate(items):
        t, g = len(item["time_valid"]), len(item["global_valid"])
        for name in (
            "scene_features",
            "roi_features",
            "geometry",
            "object_valid",
            "time_valid",
            "local_time",
            "frame_seconds",
        ):
            output[name][i, :t] = item[name]
        for name in ("global_features", "global_time", "global_valid"):
            output[name][i, :g] = item[name]
    for name in (
        "entry_index",
        "entry_supervised",
        "collision_index",
        "entry_side",
        "evasion",
    ):
        output[name] = torch.tensor([x[name] for x in items])
    output["frame_ids"] = [x["frame_ids"] for x in items]
    output["sample_id"] = [x["sample_id"] for x in items]
    output["source_id"] = [x["source_id"] for x in items]
    return output
