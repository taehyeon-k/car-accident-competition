"""DataLoader construction for live native-frame and cached-feature datasets."""

import torch
from torch.utils.data import DataLoader
from .dataset import NativeDataset, Stage2Dataset, collate
from .joint import JointFeatureDataset, joint_collate


def get_data(
    manifest,
    batch_size,
    num_workers,
    shuffle,
    *,
    config=None,
):
    if config is not None and config["stage"] == "joint":
        ds = JointFeatureDataset(
            manifest,
            config["data"]["feature_dir"],
            training=shuffle,
            censor_probability=config["data"].get(
                "censored_entry_probability", 0.15
            ),
        )
        return ds, DataLoader(
            ds,
            batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            collate_fn=joint_collate,
        )
    if (
        config is not None
        and config["model"].get(
            "training_mode",
            "lora",
        )
        == "lora"
    ):
        ds = NativeDataset(
            manifest,
            config["stage"],
            config["tracking"],
            shuffle,
            config["seed"],
            coarse_t_max=config["model"].get("T_max", 32),
        )
    else:
        ds = Stage2Dataset(manifest)
    return ds, DataLoader(
        ds,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate,
    )
