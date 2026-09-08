"""Cache frozen per-frame observations, never sampled tracks or LoRA features.

Run ``python -m stage2.data.cache_geometry --config ... --manifest ...``.
Factories must construct locally loaded inference adapters. Detector output is a
list of dictionaries with native xyxy boxes, scores and canonical class names.
Depth output is a list of native-resolution relative-depth tensors. Both receive
the original RGB CHW uint8 tensors, with no photometric augmentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.io import read_image, ImageReadMode

from stage2.model.backbones import FrozenAdapter
from stage2.model.tracking import VEHICLE_CLASSES
from stage2.data.sampling import sort_frame_paths
from stage2.utils.utils import atomic_save, file_digest, load_config, read_manifest


def frame_paths(directory: str) -> tuple[list[Path], list[int]]:
    """Discover supported images and enforce a unique numeric frame mapping."""
    paths = [
        path
        for path in Path(directory).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    paths, ids = sort_frame_paths(paths)
    if not paths:
        raise ValueError(f"No RGB frames in {directory}")
    return paths, ids


def compact_observations(
    detections: dict,
    depth: torch.Tensor,
    width: int,
    height: int,
    threshold: float,
    closer_is_larger: bool,
) -> dict:
    """Keep normalized per-detection proximity instead of a full dense depth map.

    Relative object ranks are deliberately deferred until the window's top-12
    tracks have been selected. This compact cache is reusable across all windows.
    """
    depth = torch.as_tensor(depth).detach().float().cpu().numpy()
    if depth.shape != (height, width) or not np.isfinite(depth).all():
        raise ValueError("Depth adapter must return a finite map in native coordinates")
    if not closer_is_larger:
        depth = -depth
    median = float(np.median(depth))
    mad = max(
        float(np.median(np.abs(depth - median))),
        1e-6,
    )
    boxes = []
    scores = []
    labels = []
    proximity = []
    if not (
        len(detections["boxes"])
        == len(detections["scores"])
        == len(detections["labels"])
    ):
        raise ValueError("Detector outputs have mismatched lengths")
    for box, score, label in zip(
        detections["boxes"],
        detections["scores"],
        detections["labels"],
    ):
        box = torch.as_tensor(box).detach().float().cpu().numpy()
        score = float(score)
        if label not in VEHICLE_CLASSES or not np.isfinite(score) or score < threshold:
            continue
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("Detector boxes must be finite native xyxy coordinates")
        box = np.clip(
            box,
            [0, 0, 0, 0],
            [width, height, width, height],
        )
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        left = int(x1 + 0.25 * (x2 - x1))
        right = int(x2 - 0.25 * (x2 - x1))
        top = int(y1 + 0.55 * (y2 - y1))
        bottom = int(y2 - 0.10 * (y2 - y1))
        patch = depth[top:bottom, left:right]
        value = float(np.median(patch)) if patch.size else median
        boxes.append(box.tolist())
        scores.append(score)
        labels.append(label)
        proximity.append((value - median) / mad)
    return {
        "boxes": torch.tensor(
            boxes,
            dtype=torch.float32,
        ).reshape(
            -1,
            4,
        ),
        "scores": torch.tensor(scores),
        "labels": labels,
        "proximity": torch.tensor(proximity),
        "size": [width, height],
    }


@torch.inference_mode()
def cache_manifest(
    config: dict,
    manifest: str,
    device: str = "cpu",
) -> None:
    """Create compact caches with source and checkpoint provenance checks."""
    model_config = config["model"]
    orientation = model_config.get("depth_closer_is_larger")
    if not isinstance(
        orientation,
        bool,
    ):
        raise ValueError(
            "Verify the depth adapter's orientation, then set depth_closer_is_larger"
        )
    detector = FrozenAdapter(
        model_config["rfdetr_factory"],
        model_config["rfdetr_checkpoint"],
    ).to(device)
    depth_model = FrozenAdapter(
        model_config["depth_factory"],
        model_config["depth_checkpoint"],
    ).to(device)
    provenance = {
        "schema": 1,
        "detector_sha256": file_digest(model_config["rfdetr_checkpoint"]),
        "depth_sha256": file_digest(model_config["depth_checkpoint"]),
        "detector_factory": model_config["rfdetr_factory"],
        "depth_factory": model_config["depth_factory"],
        "depth_closer_is_larger": orientation,
        "threshold": config["tracking"]["detection_threshold"],
    }
    chunk_size = model_config.get(
        "geometry_batch_size",
        4,
    )
    if chunk_size < 1:
        raise ValueError("geometry_batch_size must be positive")

    for row in read_manifest(manifest):
        paths, ids = frame_paths(row["frames_dir"])
        directory = Path(row["geometry_dir"])
        metadata_path = directory / "metadata.pt"
        if metadata_path.exists():
            stored = torch.load(
                metadata_path,
                weights_only=True,
            )
            if stored != provenance:
                raise ValueError(
                    f"Cache provenance changed; choose a new geometry_dir: {directory}"
                )
        else:
            atomic_save(
                provenance,
                metadata_path,
            )

        pending = []
        for path, original_id in zip(
            paths,
            ids,
        ):
            target = directory / f"{original_id}.pt"
            if target.exists():
                saved = torch.load(
                    target,
                    weights_only=True,
                )
                if saved["source_sha256"] != file_digest(path):
                    raise ValueError(
                        f"Source frame changed: {path}; choose a new cache directory"
                    )
            else:
                pending.append((path, original_id))

        for start in range(
            0,
            len(pending),
            chunk_size,
        ):
            chunk = pending[start : start + chunk_size]
            images = [
                read_image(
                    str(path),
                    mode=ImageReadMode.RGB,
                ).to(device)
                for path, _ in chunk
            ]
            detections = detector(images)
            depth_maps = depth_model(images)
            if len(detections) != len(images) or len(depth_maps) != len(images):
                raise ValueError(
                    "Frozen adapters must return one output per original frame"
                )
            for (path, original_id), image, detected, depth in zip(
                chunk,
                images,
                detections,
                depth_maps,
            ):
                observations = compact_observations(
                    detected,
                    depth,
                    image.shape[-1],
                    image.shape[-2],
                    provenance["threshold"],
                    orientation,
                )
                observations["source_sha256"] = file_digest(path)
                observations["frame_id"] = original_id
                atomic_save(
                    observations,
                    directory / f"{original_id}.pt",
                )
        print(
            json.dumps({"sample_id": row["sample_id"], "frames_cached": len(pending)})
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    arguments = parser.parse_args()
    cache_manifest(
        load_config(arguments.config),
        arguments.manifest,
        arguments.device,
    )


if __name__ == "__main__":
    main()
