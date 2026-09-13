"""Cached local features and crop-local RGB paths for online frozen V-JEPA."""

from pathlib import Path
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from stage2.data.cache_geometry import frame_paths
from stage2.utils.utils import read_manifest


def load_joint_cache(row, feature_dir):
    cache = torch.load(
        Path(feature_dir) / f"{row['sample_id']}.pt",
        map_location="cpu",
        weights_only=True,
    )
    if cache.get("schema") != 2 or cache.get("sample_id") != row["sample_id"]:
        raise ValueError(
            "Joint model needs schema-2 local caches; regenerate the cache"
        )
    length = len(cache["frame_ids"])
    expected = {
        "scene_features": (length, 17, 768),
        "roi_features": (length, 12, 768),
        "geometry": (length, 12, 13),
        "object_valid": (length, 12),
        "track_ids": (12,),
    }
    for name, shape in expected.items():
        if tuple(cache[name].shape) != shape:
            raise ValueError(f"Invalid {name} shape in {row['sample_id']}")
    paths, ids = frame_paths(row["frames_dir"])
    if ids != cache["frame_ids"].tolist():
        raise ValueError("RGB frame IDs and local cache do not match")
    if not length:
        raise ValueError("Empty joint sequence")
    return cache, paths


def joint_item(row, cache, paths, start=0, stop=None, supervised=False):
    stop = len(paths) if stop is None else stop
    length = stop - start
    if not 0 <= start < stop <= len(paths):
        raise ValueError("Invalid joint crop")
    ids = cache["frame_ids"][start:stop].long()
    item = {
        name: cache[name][start:stop]
        for name in ("scene_features", "roi_features", "geometry", "object_valid")
    }
    item.update(
        time_valid=torch.ones(length, dtype=torch.bool),
        local_time=torch.linspace(0, 1, length),
        frame_ids=ids,
        frame_paths=[str(p) for p in paths[start:stop]],
        sample_id=row["sample_id"],
        source_id=row.get("source_id", row["sample_id"]),
    )
    if supervised:
        fps = float(row.get("native_fps") or 0)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(
                "Training/evaluation seconds-based losses require native_fps; inference does not"
            )
        for event in ("entry", "collision"):
            matches = (ids == int(row[f"{event}_frame"])).nonzero().flatten()
            if len(matches) != 1:
                raise ValueError(f"Crop must contain exactly one {event} label")
            item[f"{event}_index"] = int(matches[0])
        if item["entry_index"] > item["collision_index"]:
            raise ValueError("ENTRY must not follow COLLISION")
        side = row["entry_side"]
        if side not in ("LEFT", "RIGHT", 0, 1) or row["evasion_space"] not in (0, 1):
            raise ValueError("Invalid joint attribute labels")
        item.update(
            frame_seconds=torch.arange(length, dtype=torch.float32) / fps,
            entry_side=int(side == "RIGHT") if isinstance(side, str) else int(side),
            evasion=float(row["evasion_space"]),
        )
    return item


class JointFeatureDataset(Dataset):
    def __init__(self, manifest, feature_dir, training=False, seed=42):
        self.rows = read_manifest(manifest)
        self.feature_dir, self.training, self.seed = Path(feature_dir), training, seed
        self.epoch = torch.zeros((), dtype=torch.long).share_memory_()

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        cache, paths = load_joint_cache(row, self.feature_dir)
        ids = cache["frame_ids"].long()
        positions = []
        for event in ("entry", "collision"):
            matches = (ids == int(row[f"{event}_frame"])).nonzero().flatten()
            if len(matches) != 1:
                raise ValueError(f"Missing or duplicated {event} frame")
            positions.append(int(matches[0]))
        entry, collision = positions
        if entry > collision:
            raise ValueError("ENTRY must not follow COLLISION")
        start, stop = 0, len(ids)
        if self.training:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, int(self.epoch), index])
            )
            start = int(rng.integers(0, entry + 1))
            stop = int(rng.integers(collision + 1, len(ids) + 1))
        return joint_item(row, cache, paths, start, stop, supervised=True)


def joint_collate(items):
    """Pad local features; keep RGB paths on the host and decode on demand."""
    max_time = max(len(item["time_valid"]) for item in items)
    output = {}
    for name in (
        "scene_features",
        "roi_features",
        "geometry",
        "object_valid",
        "time_valid",
        "local_time",
        "frame_seconds",
    ):
        if name not in items[0]:
            continue
        first = items[0][name]
        padded = first.new_zeros((len(items), max_time, *first.shape[1:]))
        for i, item in enumerate(items):
            padded[i, : len(item[name])] = item[name]
        output[name] = padded
    for name in ("entry_index", "collision_index", "entry_side", "evasion"):
        if name in items[0]:
            output[name] = torch.tensor([item[name] for item in items])
    for name in ("frame_ids", "frame_paths", "sample_id", "source_id"):
        output[name] = [item[name] for item in items]
    return output
