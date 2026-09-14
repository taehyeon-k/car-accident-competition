"""DataLoader construction for the joint cached-feature dataset."""

import torch
from torch.utils.data import DataLoader
from .joint import JointFeatureDataset, joint_collate


def get_data(
    manifest,
    batch_size,
    num_workers,
    shuffle,
    *,
    config=None,
):
    ds = JointFeatureDataset(
        manifest,
        training=shuffle,
        seed=config["seed"],
        config=config,
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
