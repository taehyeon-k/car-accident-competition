"""Fit separate coarse/fine normalization from training geometry only.

Run ``python -m stage2.data.geometry_stats --config ... --output ...`` after
creating frozen geometry caches. A bounded random-priority reservoir avoids
keeping every observation in a large dataset in memory.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from stage2.data.dataset import NativeDataset
from stage2.utils.utils import atomic_save, load_config


def fit_statistics(
    samples,
    stage: str,
    max_observations: int = 100000,
    seed: int = 42,
) -> dict:
    """Ignore padding and fit mean/std or median/MAD according to channel type."""
    if max_observations < 1:
        raise ValueError("max_observations must be positive")
    rng = np.random.default_rng(seed)
    observations = np.empty(
        (0, 9),
        dtype=np.float32,
    )
    priorities = np.empty(0)
    for sample in samples:
        values = sample["geometry"][sample["object_valid"]].numpy()
        if not np.isfinite(values).all():
            raise ValueError("Training geometry has nonfinite values")
        observations = np.concatenate((observations, values))
        priorities = np.concatenate((priorities, rng.random(len(values))))
        if len(priorities) > max_observations:
            retained = np.argpartition(
                priorities,
                max_observations - 1,
            )[:max_observations]
            observations = observations[retained]
            priorities = priorities[retained]
    if len(observations) == 0:
        raise ValueError(
            "No observed training objects available for geometry statistics"
        )

    center = observations.mean(axis=0)
    scale = observations.std(axis=0)
    for channel in (5, 7, 8):
        center[channel] = np.median(observations[:, channel])
        scale[channel] = np.median(np.abs(observations[:, channel] - center[channel]))
    # Constant channels must remain well-behaved at validation/inference.
    scale = np.where(
        scale > 1e-6,
        scale,
        1.0,
    )
    return {
        "stage": stage,
        "split": "train",
        "center": torch.as_tensor(
            center,
            dtype=torch.float32,
        ),
        "scale": torch.as_tensor(
            scale,
            dtype=torch.float32,
        ),
        "observations": len(observations),
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    dataset = NativeDataset(
        config["data"]["manifest"],
        config["stage"],
        config["tracking"],
        training=True,
        seed=config["seed"],
        geometry_only=True,
        coarse_t_max=config["model"].get("T_max", 32),
    )

    def sampled_geometry():
        for epoch in range(arguments.passes):
            dataset.set_epoch(epoch)
            for index in range(len(dataset)):
                yield dataset[index]

    statistics = fit_statistics(
        sampled_geometry(),
        config["stage"],
        seed=config["seed"],
    )
    statistics["coarse_t_max"] = config["model"].get("T_max", 32)
    statistics["source_ids"] = sorted({str(row["source_id"]) for row in dataset.rows})
    atomic_save(
        statistics,
        arguments.output,
    )


if __name__ == "__main__":
    main()
