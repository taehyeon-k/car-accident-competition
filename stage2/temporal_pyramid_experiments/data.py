"""Frame-count-only adaptive sampling from a frozen FPS-blind feature cache."""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from io import BytesIO
import json
from zipfile import ZipFile
import numpy as np
import torch
from torch.utils.data import Dataset

from stage2.spotting_experiments.data import collate
from stage2.spotting_experiments.sampling import sample_from_cache


@lru_cache(maxsize=600)
def cached_features(path: str):
    # The NumPy array reader is pathologically slow on this image for some
    # 10–20 MB members. These caches are uncompressed NPY members; parse the
    # standard header and make a zero-copy view of the payload.
    def member(archive, name):
        raw = archive.read(name)
        stream = BytesIO(raw)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else: raise ValueError(f"Unsupported NPY version: {version}")
        if fortran or dtype.hasobject: raise ValueError("Unexpected cache array format")
        return np.frombuffer(raw, dtype=dtype, offset=stream.tell()).reshape(shape)
    with ZipFile(path) as archive:
        frames = member(archive, "frame_numbers.npy")
        features = member(archive, "features.npy")
    return frames, features


def adaptive_count(n: int) -> int:
    if n <= 128: return n
    if n <= 256: return 128
    if n <= 512: return 192
    if n <= 768: return 256
    return 320


class PyramidFeatures(Dataset):
    def __init__(self, rows, cache_dir, experiment="P1", train=False, seed=0,
                 override_count=None, view=0,
                 low_cache_dir="stage2/artifacts/spotting_features/fpsblind_192_7x10_349"):
        self.rows, self.cache_dir = rows, Path(cache_dir)
        self.low_cache_dir = Path(low_cache_dir)
        self.experiment, self.train, self.seed = experiment, train, seed
        self.override_count, self.view = override_count, view
        meta = json.loads((self.cache_dir / "meta.json").read_text())
        if meta.get("mode") != "fps_blind" or meta.get("fps_accessed"):
            raise ValueError("Only strictly FPS-blind caches are permitted")
        low_meta = json.loads((self.low_cache_dir / "meta.json").read_text())
        if low_meta.get("mode") != "fps_blind" or low_meta.get("fps_accessed"):
            raise ValueError("Low-resolution cache is not strictly FPS-blind")

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        n = int(row["num_frames"])
        count = 128 if self.experiment == "P1" else adaptive_count(n)
        if self.override_count is not None:
            count = int(self.override_count) if self.experiment == "P1" else min(int(self.override_count), n)
        rng = np.random.default_rng(self.seed + i * 1000003 + self.view * 991 +
                                    (np.random.randint(1 << 20) if self.train else 0))
        if self.train and self.experiment in {"P3", "P4"}:
            options = sorted(set([max(32, round(count * .75 / 32) * 32), count,
                                  min(320, round(count * 1.25 / 32) * 32)]))
            count = min(n, int(rng.choice(options)))
        chosen_cache = self.low_cache_dir if count <= 192 else self.cache_dir
        all_frames, all_features = cached_features(str(chosen_cache / f"{row['sample_id']}.npz"))
        jitter = .35 if self.train and self.experiment in {"P3", "P4"} else 0.0
        drop = .025 if self.train and self.experiment in {"P3", "P4"} else 0.0
        positions = sample_from_cache(all_frames, count, jitter=jitter, drop_probability=drop, rng=rng)
        if self.train and self.experiment in {"P3", "P4"} and len(positions) > 2:
            u = np.linspace(0, 1, len(positions))
            displacement = rng.uniform(-.02, .02) * (len(all_frames) - 1) * np.sin(np.pi * u)
            positions = np.maximum.accumulate(np.rint(positions + displacement).clip(0, len(all_frames) - 1)).astype(np.int64)
        # The 383-grid is nested around the original 192-grid. For clips shorter
        # than 128, remove duplicate frame numbers so all available frames appear.
        if self.experiment != "P1" and n <= 128 and self.override_count is None:
            positions = np.unique(all_frames, return_index=True)[1]
        frames = all_frames[positions]
        normalized = (frames - frames[0]).astype(np.float32) / max(int(frames[-1] - frames[0]), 1)
        # Fancy indexing already allocates a fresh array; avoid a second copy.
        selected_features = all_features[positions]
        return {"x": torch.from_numpy(selected_features),
                "frame_numbers": torch.from_numpy(frames.copy()),
                "normalized_positions": torch.from_numpy(normalized),
                "entry_index": int(np.abs(frames - int(row["entry_frame"])).argmin()),
                "collision_index": int(np.abs(frames - int(row["collision_frame"])).argmin()),
                "entry_frame": int(row["entry_frame"]), "collision_frame": int(row["collision_frame"]),
                "entry_side": int(row["entry_side"] == "RIGHT"), "evasion": int(row["evasion_space"]),
                "sample_id": row["sample_id"], "source_id": row.get("source_id", "unknown"),
                "num_available_frames": n}
