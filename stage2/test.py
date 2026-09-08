"""Submission-side decoding utilities; model loading is deliberately local only."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[1]),
    )


def merge_window_logits(
    windows: list[np.ndarray],
    logits: list[np.ndarray],
) -> tuple[int, float]:
    values = defaultdict(list)
    if len(windows) != len(logits):
        raise ValueError("Each fine window requires a matching logit array")
    for frames, window_logits in zip(
        windows,
        logits,
    ):
        if (
            len(window_logits) < len(frames)
            or not np.isfinite(window_logits[: len(frames)]).all()
        ):
            raise ValueError("Every native frame needs a finite raw logit")
        for frame, logit in zip(
            frames,
            window_logits[: len(frames)],
        ):
            values[int(frame)].append(float(logit))
    if not values:
        raise ValueError("No fine logits to merge")
    best = max(
        ((np.mean(v), frame) for frame, v in values.items()),
        key=lambda x: x[0],
    )
    return best[1], float(best[0])


def side_label(direction_logit: np.ndarray) -> str:
    return "LEFT" if int(np.argmax(direction_logit)) == 0 else "RIGHT"


def evasion_label(logit: float) -> int:
    """sigmoid(logit) >= .5 is equivalent to logit >= 0, without overflow."""
    return int(logit >= 0)


def main() -> None:
    from stage2.model.inference import Stage2Pipeline
    from stage2.utils.checkpoint import load_inference_system
    from stage2.utils.utils import read_manifest

    parser = argparse.ArgumentParser(
        description="Run offline Stage 2 native-frame inference"
    )
    parser.add_argument(
        "--coarse-ckpt",
        required=True,
    )
    parser.add_argument(
        "--fine-ckpt",
        required=True,
    )
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    arguments = parser.parse_args()

    # Training checkpoints contain Python/NumPy RNG state. Load only trusted
    # checkpoints produced by this repository.
    coarse, config = load_inference_system(
        arguments.coarse_ckpt,
        "coarse",
    )
    fine, fine_config = load_inference_system(
        arguments.fine_ckpt,
        "fine",
    )
    if fine_config["tracking"] != config["tracking"]:
        raise ValueError("Coarse and fine tracking configurations must match")
    pipeline = Stage2Pipeline(
        coarse,
        fine,
        config,
        device=arguments.device,
    )
    with open(
        arguments.output,
        "x",
        encoding="utf-8",
    ) as stream:
        for row in read_manifest(arguments.manifest):
            stream.write(
                json.dumps({"sample_id": row["sample_id"], **pipeline.predict(row)})
                + "\n"
            )


if __name__ == "__main__":
    main()
