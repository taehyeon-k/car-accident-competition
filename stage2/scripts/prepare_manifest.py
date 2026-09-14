"""Validate native labels and split whole source groups before augmentation."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from stage2.data.cache_geometry import frame_paths
from stage2.utils.utils import read_manifest, source_name


def stratum_key(row: dict) -> tuple[str, str, str]:
    """Joint stratification key: dataset source, entry side and evasion space.

    The dataset source is the prefix of ``source_id`` written by
    ``prepare_workspace``; manifests without that prefix stratify on the whole
    identifier, which degrades to grouping rather than failing.
    """
    source = source_name(row["source_id"])
    return source, str(row["entry_side"]), str(row["evasion_space"])


def split_rows(
    rows: list[dict],
    validation_fraction: float,
    seed: int,
) -> tuple[list, list]:
    """Stratified split keeping every crop/variant of a source in one split.

    Source groups stay indivisible, so they are the unit that is assigned. Each
    group takes the stratum of its most common member key, and every stratum
    contributes its proportional share of validation groups. Both splits
    therefore keep approximately the full source and label distribution.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["source_id"]), []).append(row)
    if len(groups) < 2:
        raise ValueError("Need at least two original source groups")
    strata: dict[tuple[str, str, str], list[str]] = {}
    for group, members in groups.items():
        keys = Counter(stratum_key(row) for row in members)
        # Mixed-label groups resolve to their most common key, ties by sort order.
        strata.setdefault(min(keys, key=lambda key: (-keys[key], key)), []).append(
            group
        )
    target = min(
        len(groups) - 1,
        max(
            1,
            round(len(groups) * validation_fraction),
        ),
    )
    # Largest-remainder apportionment: each stratum keeps its proportional share
    # while the total validation-group count stays exactly on target.
    quotas = {key: len(value) * target / len(groups) for key, value in strata.items()}
    counts = {key: int(value) for key, value in quotas.items()}
    ranked = sorted(quotas, key=lambda key: (counts[key] - quotas[key], key))
    for key in ranked[: target - sum(counts.values())]:
        counts[key] += 1
    rng = np.random.default_rng(seed)
    validation_groups: set[str] = set()
    for key in sorted(strata):
        members = sorted(strata[key])
        rng.shuffle(members)
        validation_groups.update(members[: counts[key]])
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
