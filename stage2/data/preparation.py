"""Build window-specific tracks, geometry and ROI boxes from frozen observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from stage2.data.transforms import box_to_grid, letterbox_metadata
from stage2.model.geometry import build_geometry, tubelet_geometry
from stage2.model.tracking import Detection, HungarianTracker, rank_tracks


def load_observation(
    directory: str,
    original_id: int,
) -> dict:
    """Read one compact detector/depth record using the original filename ID."""
    path = Path(directory) / f"{original_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing geometry cache {path}; run stage2.data.cache_geometry"
        )
    record = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    if record["frame_id"] != original_id:
        raise ValueError(f"Mismatched cached original frame ID in {path}")
    return record


def build_window_geometry(
    records: list[dict],
    valid: np.ndarray,
    tracking_config: dict,
    coarse: bool,
) -> dict[str, torch.Tensor]:
    """Rerun association/ranking for this window; never reuse cached track IDs."""
    sizes = [tuple(record["size"]) for record in records]
    if len(set(sizes)) != 1:
        raise ValueError("Frames in a sample must use a common native resolution")
    observations = []
    for record, is_valid in zip(
        records,
        valid,
    ):
        frame = []
        if is_valid:
            for box, score, label, proximity in zip(
                record["boxes"],
                record["scores"],
                record["labels"],
                record["proximity"],
            ):
                frame.append(
                    Detection(
                        np.asarray(box),
                        float(score),
                        label,
                        float(proximity),
                    )
                )
        observations.append(frame)

    tracker_options = {
        key: value for key, value in tracking_config.items() if key != "loom_clip"
    }
    tracker = HungarianTracker(**tracker_options)
    tracks = tracker.track(
        observations,
        sizes,
    )
    tracks = rank_tracks(
        tracks,
        total_frames=32 if coarse else int(valid.sum()),
        sizes=sizes,
        loom_clip=tracking_config["loom_clip"],
    )
    geometry, object_valid = build_geometry(
        tracks,
        None,
        sizes,
        len(records),
    )
    native_boxes = torch.zeros(
        len(records),
        12,
        4,
    )
    for slot, track in enumerate(tracks):
        for position, detection in track.observations.items():
            native_boxes[position, slot] = torch.as_tensor(detection.box)

    geometry = torch.from_numpy(geometry)
    object_valid = torch.from_numpy(object_valid)
    width, height = sizes[0]
    if coarse:
        geometry, tube_valid = tubelet_geometry(
            geometry[None],
            object_valid[None],
        )
        geometry = geometry[0]
        pair_boxes = native_boxes.reshape(
            16,
            2,
            12,
            4,
        )
        pair_valid = object_valid.reshape(
            16,
            2,
            12,
        )
        minimum = (
            pair_boxes[..., :2]
            .masked_fill(
                ~pair_valid[..., None],
                torch.inf,
            )
            .amin(dim=1)
        )
        maximum = (
            pair_boxes[..., 2:]
            .masked_fill(
                ~pair_valid[..., None],
                -torch.inf,
            )
            .amax(dim=1)
        )
        # Expand by 15% TOTAL width/height: 7.5% on each side of the union.
        padding = (maximum - minimum) * 0.075
        native_boxes = torch.cat(
            (minimum - padding, maximum + padding),
            dim=-1,
        )
        object_valid = tube_valid[0]
        native_boxes = native_boxes.masked_fill(
            ~object_valid[..., None],
            0,
        )
        native_boxes[..., 0::2].clamp_(
            0,
            width,
        )
        native_boxes[..., 1::2].clamp_(
            0,
            height,
        )

    metadata = letterbox_metadata(
        width,
        height,
        384 if coarse else 336,
    )
    grid_boxes = box_to_grid(
        native_boxes,
        metadata,
        16 if coarse else 14,
    )
    grid_boxes = grid_boxes.masked_fill(
        ~object_valid[..., None],
        0,
    )
    return {
        "geometry": geometry,
        "object_valid": object_valid,
        "boxes_grid": grid_boxes,
    }
