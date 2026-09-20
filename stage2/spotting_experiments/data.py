from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .sampling import sample_from_cache


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


class SpottingFeatures(Dataset):
    """Strict samples never read native_fps or duration."""
    def __init__(self, rows, cache_dir, sample_counts=(128,), train=False, jitter=0.0,
                 drop_probability=0.0, seed=0, strict_fps_blind=True):
        self.rows, self.cache_dir, self.sample_counts = rows, Path(cache_dir), tuple(sample_counts)
        self.train, self.jitter, self.drop_probability, self.seed = train, jitter, drop_probability, seed
        self.strict = strict_fps_blind
        meta = json.loads((self.cache_dir / "meta.json").read_text())
        if self.strict and (meta.get("mode") != "fps_blind" or meta.get("fps_accessed")):
            raise ValueError("Strict FPS-blind training requires an FPS-blind feature cache")

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        cache = np.load(self.cache_dir / f"{row['sample_id']}.npz")
        all_frames = cache["frame_numbers"]
        rng = np.random.default_rng(self.seed + i * 1000003 + (np.random.randint(1 << 20) if self.train else 0))
        count = int(rng.choice(self.sample_counts)) if self.train else int(self.sample_counts[0])
        positions = sample_from_cache(all_frames, count, jitter=self.jitter if self.train else 0.0,
                                      drop_probability=self.drop_probability if self.train else 0.0, rng=rng)
        frames = all_frames[positions]
        span = max(int(frames[-1] - frames[0]), 1)
        normalized = (frames - frames[0]).astype(np.float32) / span
        entry_index = int(np.abs(frames - int(row["entry_frame"])).argmin())
        collision_index = int(np.abs(frames - int(row["collision_frame"])).argmin())
        return {"x": torch.from_numpy(cache["features"][positions].copy()),
                "frame_numbers": torch.from_numpy(frames.copy()),
                "normalized_positions": torch.from_numpy(normalized),
                "entry_index": entry_index, "collision_index": collision_index,
                "entry_frame": int(row["entry_frame"]), "collision_frame": int(row["collision_frame"]),
                "entry_side": int(row["entry_side"] == "RIGHT"), "evasion": int(row["evasion_space"]),
                "sample_id": row["sample_id"], "source_id": row.get("source_id", "unknown"),
                "num_available_frames": int(row["num_frames"])}


def collate(items):
    length = max(len(x["x"]) for x in items); b = len(items); n, c = items[0]["x"].shape[1:]
    features = torch.zeros(b, length, n, c, dtype=torch.float16)
    valid = torch.zeros(b, length, dtype=torch.bool)
    frames = torch.zeros(b, length, dtype=torch.long)
    positions = torch.zeros(b, length)
    for j, item in enumerate(items):
        t = len(item["x"]); features[j, :t] = item["x"]; valid[j, :t] = True
        frames[j, :t] = item["frame_numbers"]; positions[j, :t] = item["normalized_positions"]
        if t < length:
            frames[j, t:] = item["frame_numbers"][-1]; positions[j, t:] = 1.0
    result = {"x": features, "time_valid": valid, "frame_numbers": frames, "normalized_positions": positions}
    for key in ("entry_index", "collision_index", "entry_frame", "collision_frame", "entry_side", "evasion", "num_available_frames"):
        result[key] = torch.tensor([x[key] for x in items])
    for key in ("sample_id", "source_id"): result[key] = [x[key] for x in items]
    return result
