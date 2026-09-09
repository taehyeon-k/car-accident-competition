"""Native-frame datasets with stochastic sampling and frozen observation caches."""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from torchvision.io import read_image, ImageReadMode

from stage2.data.cache_geometry import frame_paths
from stage2.data.preparation import build_window_geometry, load_observation
from stage2.data.sampling import (
    build_coarse_bins,
    event_bin,
    event_preserving_crop,
    temporal_rate_augment,
    sample_fine_window,
    recover_region,
    sliding_windows,
)
from stage2.data.transforms import ClipPhotometric, letterbox
from stage2.utils.utils import read_manifest


class Stage2Dataset(Dataset):
    """Prepared tensor records, available only as an explicit cached-feature ablation."""

    def __init__(
        self,
        manifest: str,
    ):
        self.rows = read_manifest(manifest)

    def __len__(self):
        return len(self.rows)

    def __getitem__(
        self,
        i,
    ):
        row = self.rows[i]
        if "feature_path" not in row:
            raise ValueError("Cached-feature mode requires feature_path records")

        item = torch.load(
            row["feature_path"],
            map_location="cpu",
            weights_only=True,
        )
        item.update({k: v for k, v in row.items() if k != "feature_path"})
        return item


class NativeDataset(Dataset):
    """Sample native RGB and rebuild all track-dependent inputs on each access.

    A shared epoch tensor keeps worker sampling reproducible across resumed runs.
    Validation uses full native sequences and deterministic fine-window selection.
    ``geometry_only`` enables training-only normalization fitting without RGB I/O.
    """

    def __init__(
        self,
        manifest,
        stage,
        tracking,
        training,
        seed=42,
        geometry_only=False,
        coarse_t_max=32,
    ):
        self.rows = read_manifest(manifest)
        self.stage = stage
        self.coarse_t_max = coarse_t_max
        self.tracking = tracking
        self.training = training
        self.seed = seed
        self.geometry_only = geometry_only
        self.epoch = torch.zeros(
            (),
            dtype=torch.long,
        ).share_memory_()
        self.frames = [frame_paths(row["frames_dir"]) for row in self.rows]
        for row, (_, ids) in zip(
            self.rows,
            self.frames,
        ):
            if not row.get("source_id"):
                raise ValueError(
                    "Native manifests require source_id to prevent split leakage"
                )
            if row["entry_frame"] not in ids or row["collision_frame"] not in ids:
                raise ValueError(
                    "Event labels must reference original filename frame IDs"
                )
            if ids.index(row["entry_frame"]) > ids.index(row["collision_frame"]):
                raise ValueError("Entry label must occur no later than collision")

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        self.epoch.fill_(epoch)

    def __len__(self) -> int:
        return len(self.rows) * (2 if self.stage == "fine" else 1)

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        row_index = index // 2 if self.stage == "fine" else index
        row = self.rows[row_index]
        paths, ids = self.frames[row_index]
        entry = ids.index(row["entry_frame"])
        collision = ids.index(row["collision_frame"])
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(self.epoch), index])
        )

        if self.stage == "coarse":
            candidates = np.arange(len(ids))
            if self.training:
                candidates = event_preserving_crop(
                    len(ids),
                    entry,
                    collision,
                    rng,
                )
                candidates = temporal_rate_augment(
                    candidates,
                    entry,
                    collision,
                    row.get("native_fps"),
                    rng,
                )
            bins = build_coarse_bins(
                candidates,
                ids,
                self.training,
                rng,
                num_bins=self.coarse_t_max,
            )
            positions = bins.representative_native_pos
            valid = bins.valid
            side = row["entry_side"]
            if side not in ("LEFT", "RIGHT", 0, 1) or row["evasion_space"] not in (
                0,
                1,
            ):
                raise ValueError("Invalid side/evasion label convention")
            item = {
                "entry_bin": event_bin(
                    entry,
                    bins,
                ),
                "collision_bin": event_bin(
                    collision,
                    bins,
                ),
                "entry_side": (
                    int(side == "RIGHT")
                    if isinstance(
                        side,
                        str,
                    )
                    else side
                ),
                "evasion": float(row["evasion_space"]),
                "bin_valid": torch.from_numpy(valid),
                "bin_native_pos_start": torch.from_numpy(bins.native_start),
                "bin_native_pos_end": torch.from_numpy(bins.native_end),
                "bin_candidate_start": torch.from_numpy(bins.candidate_start),
                "bin_candidate_end": torch.from_numpy(bins.candidate_end),
            }
        else:
            event_type = index % 2
            event = collision if event_type else entry
            if self.training:
                window = sample_fine_window(
                    ids,
                    event,
                    rng,
                    num_bins=self.coarse_t_max,
                )
            else:
                bins = build_coarse_bins(
                    list(range(len(ids))),
                    ids,
                    training=False,
                    num_bins=self.coarse_t_max,
                )
                region = recover_region(
                    bins,
                    event_bin(
                        event,
                        bins,
                    ),
                )
                window = next(
                    window for window in sliding_windows(region) if event in window
                )
            valid = np.arange(64) < len(window)
            positions = np.pad(
                window,
                (0, 64 - len(window)),
                mode="edge",
            )
            item = {
                "event_type": event_type,
                "event_local_index": int(np.flatnonzero(window == event)[0]),
                "time_valid": torch.from_numpy(valid),
            }

        # Repeated padded/representative positions share deterministic observations
        # and RGB decoding within this sample. Association still runs per window.
        unique_positions = np.unique(positions)
        records = {
            int(position): load_observation(
                row["geometry_dir"],
                ids[position],
            )
            for position in unique_positions
        }
        item.update(
            build_window_geometry(
                [records[int(position)] for position in positions],
                valid,
                self.tracking,
                self.stage == "coarse",
            )
        )
        item["sample_id"] = str(row["sample_id"])
        item["source_id"] = str(row["source_id"])
        item["representative_native_pos"] = torch.as_tensor(positions.copy())
        item["representative_frame_id"] = torch.tensor(
            [ids[position] for position in positions]
        )
        if self.geometry_only:
            return item

        augmentation = ClipPhotometric(rng) if self.training else None
        images = {}
        for position in unique_positions:
            image = read_image(
                str(paths[position]),
                mode=ImageReadMode.RGB,
            )
            if list(image.shape[-2:][::-1]) != records[int(position)]["size"]:
                raise ValueError("Cached geometry dimensions differ from source RGB")
            if augmentation is not None:
                image = augmentation(image)
            images[int(position)] = letterbox(
                image,
                384 if self.stage == "coarse" else 336,
            )[0]
        rgb = torch.stack([images[int(position)] for position in positions])
        if self.stage == "coarse":
            item["coarse_rgb"] = rgb.permute(
                1,
                0,
                2,
                3,
            )
        else:
            item["fine_rgb"] = rgb.masked_fill(
                ~item["time_valid"][:, None, None, None],
                0,
            )
        return item


def collate(batch):
    """Use PyTorch collation: scalar labels become tensors, strings remain lists."""
    return default_collate(batch)
