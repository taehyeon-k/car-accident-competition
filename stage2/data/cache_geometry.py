"""Cache frozen per-frame RF-DETR observations, never sampled tracks or features.

Run ``python -m stage2.data.cache_geometry --config ... --manifest ...``.
The factory must construct a locally loaded inference adapter. Detector output is
a list of dictionaries with native xyxy boxes, scores and canonical class names,
computed on the original RGB CHW uint8 frames with no photometric augmentation.
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


def _host(values):
    """Tensors move to the host in one transfer; lists pass through unchanged."""
    return values.detach().cpu() if isinstance(values, torch.Tensor) else values


def compact_observations(
    detections: dict,
    width: int,
    height: int,
    threshold: float,
) -> dict:
    """Keep compact per-detection boxes reusable across every window."""
    boxes = []
    scores = []
    labels = []
    if not (
        len(detections["boxes"])
        == len(detections["scores"])
        == len(detections["labels"])
    ):
        raise ValueError("Detector outputs have mismatched lengths")
    # Moving each CUDA candidate individually synced the device ~300 times per
    # frame (11 s on a 1,279-frame clip); one transfer yields the same values.
    detected_boxes = _host(detections["boxes"])
    detected_scores = _host(detections["scores"])
    for box, score, label in zip(
        detected_boxes,
        detected_scores,
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
        boxes.append(box.tolist())
        scores.append(score)
        labels.append(label)
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
    detector = FrozenAdapter(
        model_config["rfdetr_factory"],
        model_config["rfdetr_checkpoint"],
    ).to(device)
    # Schema 2 dropped the depth channels; schema-1 caches must not load silently.
    provenance = {
        "schema": 2,
        "detector_sha256": file_digest(model_config["rfdetr_checkpoint"]),
        "detector_factory": model_config["rfdetr_factory"],
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
            if len(detections) != len(images):
                raise ValueError(
                    "The frozen detector must return one output per original frame"
                )
            for (path, original_id), image, detected in zip(
                chunk,
                images,
                detections,
            ):
                observations = compact_observations(
                    detected,
                    image.shape[-1],
                    image.shape[-2],
                    provenance["threshold"],
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
