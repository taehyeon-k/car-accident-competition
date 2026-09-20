"""Dataset over the frozen FPS-blind geometry-DINOv3 feature cache.

Nothing here reads ``native_fps`` or any duration. Only ordered original frame
numbers and ordinal cache position are used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .sampling import build_view


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def view_tensors(cache_features, frame_numbers, slots, row):
    """Build one temporal view. ``slots`` indexes the cached base grid."""
    frames = frame_numbers[slots]
    span = max(int(frames[-1] - frames[0]), 1)
    normalized = (frames - frames[0]).astype(np.float32) / span
    return {
        "x": torch.from_numpy(cache_features[slots].copy()),
        "frame_numbers": torch.from_numpy(frames.copy()),
        "normalized_positions": torch.from_numpy(normalized),
        # Nearest sampled position to each labelled original frame.
        "entry_index": int(np.abs(frames - int(row["entry_frame"])).argmin()),
        "collision_index": int(np.abs(frames - int(row["collision_frame"])).argmin()),
    }


class SpottingViews(Dataset):
    """Yields one temporal view, or two augmented views for consistency training."""

    def __init__(self, rows, cache_dir, sample_counts=(128,), train=False, jitter=0.0,
                 drop_probability=0.0, warp=0.0, duplicate_probability=0.0,
                 two_views=False, seed=0):
        self.rows = rows
        self.cache_dir = Path(cache_dir)
        self.sample_counts = tuple(sample_counts)
        self.train = train
        self.jitter = jitter
        self.drop_probability = drop_probability
        self.warp = warp
        self.duplicate_probability = duplicate_probability
        self.two_views = two_views
        self.seed = seed
        meta = json.loads((self.cache_dir / "meta.json").read_text())
        if meta.get("mode") != "fps_blind" or meta.get("fps_accessed"):
            raise ValueError("experiments_v2 requires an FPS-blind feature cache")
        self.meta = meta

    def __len__(self):
        return len(self.rows)

    def _augment_args(self):
        if not self.train:
            return dict(jitter=0.0, drop_probability=0.0, warp=0.0, duplicate_probability=0.0)
        return dict(jitter=self.jitter, drop_probability=self.drop_probability,
                    warp=self.warp, duplicate_probability=self.duplicate_probability)

    def __getitem__(self, i):
        row = self.rows[i]
        cache = np.load(self.cache_dir / f"{row['sample_id']}.npz")
        features, frame_numbers = cache["features"], cache["frame_numbers"]
        # Fresh entropy per epoch during training; deterministic at evaluation.
        salt = int(np.random.randint(1 << 20)) if self.train else 0
        rng = np.random.default_rng(self.seed + i * 1000003 + salt)
        kwargs = self._augment_args()

        def one_view():
            count = int(rng.choice(self.sample_counts)) if self.train else int(self.sample_counts[0])
            slots = build_view(len(frame_numbers), count, rng=rng, **kwargs)
            return view_tensors(features, frame_numbers, slots, row)

        item = {
            "a": one_view(),
            "entry_frame": int(row["entry_frame"]),
            "collision_frame": int(row["collision_frame"]),
            "entry_side": int(row["entry_side"] == "RIGHT"),
            "evasion": int(row["evasion_space"]),
            "sample_id": row["sample_id"],
            "source_id": row.get("source_id", "unknown"),
            "num_available_frames": int(row["num_frames"]),
        }
        if self.two_views and self.train:
            item["b"] = one_view()
        return item


def _stack_views(views):
    length = max(len(v["x"]) for v in views)
    b = len(views)
    n, c = views[0]["x"].shape[1:]
    features = torch.zeros(b, length, n, c, dtype=torch.float16)
    valid = torch.zeros(b, length, dtype=torch.bool)
    frames = torch.zeros(b, length, dtype=torch.long)
    positions = torch.zeros(b, length)
    for j, v in enumerate(views):
        t = len(v["x"])
        features[j, :t] = v["x"]
        valid[j, :t] = True
        frames[j, :t] = v["frame_numbers"]
        positions[j, :t] = v["normalized_positions"]
        if t < length:
            # Pad by repeating the last real frame so decoding stays in range.
            frames[j, t:] = v["frame_numbers"][-1]
            positions[j, t:] = 1.0
    out = {"x": features, "time_valid": valid, "frame_numbers": frames,
           "normalized_positions": positions}
    for key in ("entry_index", "collision_index"):
        out[key] = torch.tensor([v[key] for v in views])
    return out


def collate(items):
    batch = {"a": _stack_views([x["a"] for x in items])}
    if "b" in items[0]:
        batch["b"] = _stack_views([x["b"] for x in items])
    for key in ("entry_frame", "collision_frame", "entry_side", "evasion", "num_available_frames"):
        batch[key] = torch.tensor([x[key] for x in items])
    for key in ("sample_id", "source_id"):
        batch[key] = [x[key] for x in items]
    return batch
