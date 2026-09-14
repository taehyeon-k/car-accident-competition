"""DataLoader construction for the joint dataset.

Each sample carries a decoded RGB stack, so the number of samples held in flight
(``num_workers x prefetch_factor x batch_size``) sets host memory, not GPU memory.
"""

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
    prefetch_factor = (config.get("data", {}) or {}).get("prefetch_factor", 1)
    if num_workers and (
        isinstance(prefetch_factor, bool)
        or not isinstance(prefetch_factor, int)
        or prefetch_factor < 1
    ):
        raise ValueError("data.prefetch_factor must be a positive integer")
    return ds, DataLoader(
        ds,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=joint_collate,
        # Only valid with worker processes; PyTorch rejects it at num_workers=0.
        **({"prefetch_factor": prefetch_factor} if num_workers else {}),
    )
