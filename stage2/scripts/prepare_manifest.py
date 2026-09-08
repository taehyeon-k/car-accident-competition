"""Validate native labels and split whole source groups before augmentation."""

import argparse
import json
from pathlib import Path

import numpy as np

from stage2.data.cache_geometry import frame_paths
from stage2.utils.utils import read_manifest


def split_rows(
    rows: list[dict],
    validation_fraction: float,
    seed: int,
) -> tuple[list, list]:
    """Keep every crop/variant from an original source in the same split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    groups = sorted({str(row["source_id"]) for row in rows})
    if len(groups) < 2:
        raise ValueError("Need at least two original source groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    validation_count = min(
        len(groups) - 1,
        max(
            1,
            round(len(groups) * validation_fraction),
        ),
    )
    validation_groups = set(groups[:validation_count])
    train = [row for row in rows if str(row["source_id"]) not in validation_groups]
    validation = [row for row in rows if str(row["source_id"]) in validation_groups]
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    arguments = parser.parse_args()
    rows = read_manifest(arguments.manifest)
    for row in rows:
        _, ids = frame_paths(row["frames_dir"])
        entry = ids.index(row["entry_frame"])
        collision = ids.index(row["collision_frame"])
        if (
            entry > collision
            or row["entry_side"] not in {"LEFT", "RIGHT"}
            or row["evasion_space"] not in {0, 1}
        ):
            raise ValueError(f"Invalid labels in sample {row['sample_id']}")
    train, validation = split_rows(
        rows,
        arguments.validation_fraction,
        arguments.seed,
    )
    directory = Path(arguments.output_dir)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    for name, partition in (("train", train), ("val", validation)):
        with (directory / f"{name}.jsonl").open(
            "x",
            encoding="utf-8",
        ) as stream:
            for row in partition:
                stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
