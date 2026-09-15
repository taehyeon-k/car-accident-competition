"""Online joint Stage 2 data: cached detections, crop-local geometry, live RGB.

DINOv3 and V-JEPA both train with LoRA, so no visual feature may come from a
cache. Only the frozen RF-DETR detections are cached, and even those are never
reused as a finished geometry vector: tracking and motion are rebuilt from the
frames that are actually visible inside the sampled crop, so the first frame of
a synthetic crop cannot inherit motion history from before it.

One augmentation decision is drawn per sample and applied to every frame, so the
RGB stack handed to both backbones is identical and stays consistent with the
detector boxes the geometry was computed from.

Two independent temporal mechanisms act in sequence, in this order:

1. **Semantic temporal augmentation** (70% full video / 15% ordinary crop /
   15% synthetic pre-video ENTRY) decides what the sample teaches.
2. **Training memory cap** (``training_memory.max_frames``) bounds how large an
   autograd graph one sample may build. It is not a fourth augmentation mode and
   never alters a label; it only trims windows longer than the limit.

Neither applies to validation or inference, which always read the complete video.
"""

from pathlib import Path
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from stage2.data.augment import (
    augmentation_config,
    flip_boxes,
    flip_side,
    sample_horizontal_flip,
    sample_photometric,
)
from stage2.data.cache_geometry import frame_paths
from stage2.data.joint_sampling import (
    choose_crop,
    enforce_max_train_frames,
    max_train_frames,
    temporal_probabilities,
)
from stage2.data.transforms import (
    box_to_grid,
    letterbox_metadata,
    letterbox_pad_normalize,
    letterbox_resize,
)
from stage2.model.joint_tracking import GEOMETRY_DIM, object_tensors
from stage2.model.tracking import Detection, HungarianTracker
from stage2.utils.utils import read_manifest

DETECTOR_CACHE_SCHEMA = 2
MAX_OBJECTS = 12
IMAGE_SIZE = 384
PATCH_SIZE = 16


def load_detections(geometry_dir, frame_ids):
    """Read cached frozen RF-DETR observations for exactly these frames."""
    directory = Path(geometry_dir)
    metadata = directory / "metadata.pt"
    if not metadata.is_file():
        raise FileNotFoundError(
            f"Missing detector cache metadata {metadata}; run stage2.data.cache_geometry"
        )
    schema = torch.load(metadata, map_location="cpu", weights_only=True).get("schema")
    if schema != DETECTOR_CACHE_SCHEMA:
        raise ValueError(
            f"Detector cache {directory} is schema {schema}; schema "
            f"{DETECTOR_CACHE_SCHEMA} is required. Regenerate it with "
            "stage2.data.cache_geometry."
        )
    records = []
    for frame_id in frame_ids:
        saved = torch.load(
            directory / f"{int(frame_id)}.pt", map_location="cpu", weights_only=True
        )
        if saved["frame_id"] != int(frame_id):
            raise ValueError(f"Mismatched cached frame id in {directory}")
        records.append(saved)
    return records


def crop_objects(records, flip, tracking):
    """Rebuild tracks and geometry using only the frames inside this crop.

    Flipping the detector boxes here, before association, is what keeps the
    geometry consistent with the flipped pixels: ``center_x`` becomes
    ``1 - center_x`` and ``dx`` negates as a consequence rather than as a patch.
    Mirroring is an isometry on the association costs and leaves detection order
    untouched, so track identities survive the flip.
    """
    sizes = [tuple(record["size"]) for record in records]
    if len(set(sizes)) != 1:
        raise ValueError("Frames in a sample must use a common native resolution")
    frames = []
    for record in records:
        boxes = record["boxes"]
        if flip:
            boxes = flip_boxes(boxes, record["size"][0])
        frames.append(
            [
                Detection(box.numpy(), float(score), label)
                for box, score, label in zip(boxes, record["scores"], record["labels"])
            ]
        )
    tracker = HungarianTracker(
        max_gap=tracking.get("max_gap", 2),
        max_center_distance=tracking.get("max_center_distance", 0.2),
        max_match_cost=tracking.get("max_match_cost", 0.65),
        detection_threshold=tracking.get("detection_threshold", 0.2),
    )
    tracks = tracker.track(frames, sizes)
    boxes, geometry, valid, track_ids = object_tensors(
        tracks,
        sizes,
        max_objects=MAX_OBJECTS,
        track_percentile=tracking.get("track_percentile", 90),
        return_track_ids=True,
    )
    return boxes, geometry, valid, track_ids, sizes[0]


def load_clip_rgb(paths, flip, photometric, generator=None):
    """Decode, augment and letterbox one clip with a single shared configuration.

    Photometric work runs at the backbone's 384px resolution rather than native
    (up to 1920x1080), which is where most of the worker time went. It is applied
    after the resize but before the mean-colour padding, so the padding still
    normalizes to exactly zero. The stack is returned as float16: these are the
    largest tensors crossing the worker boundary, and halving them is what keeps
    the loader inside host memory. Values are ImageNet-normalized, so the range
    is far inside float16's precision.
    """
    from torchvision.io import ImageReadMode, read_image
    from torchvision.transforms import functional as TVF

    frames = []
    for path in paths:
        image = read_image(str(path), mode=ImageReadMode.RGB)
        height, width = image.shape[-2:]
        unit = TVF.convert_image_dtype(image, torch.float32)
        if flip:
            unit = TVF.hflip(unit)
        resized = letterbox_resize(unit, IMAGE_SIZE)
        if photometric is not None:
            resized = photometric(resized, generator=generator)
        frames.append(
            letterbox_pad_normalize(resized, IMAGE_SIZE, width, height)[0].half()
        )
    return torch.stack(frames)


def joint_item(
    row,
    paths,
    frame_ids,
    records,
    start,
    stop,
    entry_index=None,
    collision_index=None,
    flip=False,
    photometric=None,
    tracking=None,
    generator=None,
):
    """Build one training/inference sample for the window ``[start, stop)``."""
    length = stop - start
    if not 0 <= start < stop <= len(paths):
        raise ValueError("Invalid joint crop")
    window = records[start:stop]
    boxes, geometry, valid, track_ids, size = crop_objects(window, flip, tracking or {})
    transform = letterbox_metadata(size[0], size[1], IMAGE_SIZE)
    item = {
        "rgb": load_clip_rgb(paths[start:stop], flip, photometric, generator),
        "roi_boxes": box_to_grid(boxes.reshape(-1, 4), transform, PATCH_SIZE).reshape(
            length, MAX_OBJECTS, 4
        ),
        "geometry": geometry,
        "object_valid": valid,
        "time_valid": torch.ones(length, dtype=torch.bool),
        "local_time": torch.linspace(0, 1, length),
        "frame_ids": torch.as_tensor(frame_ids[start:stop]).long(),
        "track_ids": track_ids,
        "sample_id": row["sample_id"],
        "source_id": row.get("source_id", row["sample_id"]),
        "flipped": bool(flip),
    }
    if entry_index is None:
        return item

    fps = float(row.get("native_fps") or 0)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(
            "Training/evaluation seconds-based losses require native_fps; inference does not"
        )
    if not 0 <= entry_index <= collision_index < length:
        raise ValueError("ENTRY must precede COLLISION inside the crop")
    side = row["entry_side"]
    if side not in ("LEFT", "RIGHT", 0, 1) or row["evasion_space"] not in (0, 1):
        raise ValueError("Invalid joint attribute labels")
    if flip:
        side = flip_side(side)
    item.update(
        entry_index=int(entry_index),
        collision_index=int(collision_index),
        frame_seconds=torch.arange(length, dtype=torch.float32) / fps,
        entry_side=int(side == "RIGHT") if isinstance(side, str) else int(side),
        evasion=float(row["evasion_space"]),
    )
    return item


class JointFeatureDataset(Dataset):
    """Cached detections plus online RGB; no cached DINO/V-JEPA features."""

    def __init__(self, manifest, training=False, seed=42, config=None):
        self.rows = read_manifest(manifest)
        self.training = training
        self.seed = seed
        config = config or {}
        self.tracking = config.get("tracking", {})
        self.augmentation = augmentation_config(config.get("augmentation"))
        self.temporal = temporal_probabilities(config.get("temporal_augmentation"))
        # Memory control only; never applied to validation or inference.
        self.max_frames = max_train_frames(config.get("training_memory"))
        self.epoch = torch.zeros((), dtype=torch.long).share_memory_()

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)

    def __len__(self):
        return len(self.rows)

    def _events(self, row, frame_ids):
        ids = torch.as_tensor(frame_ids).long()
        positions = []
        for event in ("entry", "collision"):
            matches = (ids == int(row[f"{event}_frame"])).nonzero().flatten()
            if len(matches) != 1:
                raise ValueError(f"Missing or duplicated {event} frame")
            positions.append(int(matches[0]))
        if positions[0] > positions[1]:
            raise ValueError("ENTRY must not follow COLLISION")
        return positions

    def __getitem__(self, index):
        row = self.rows[index]
        paths, frame_ids = frame_paths(row["frames_dir"])
        records = load_detections(row["geometry_dir"], frame_ids)
        entry, collision = self._events(row, frame_ids)

        if not self.training:
            # Validation and inference see the complete, unaugmented video.
            return joint_item(
                row,
                paths,
                frame_ids,
                records,
                0,
                len(paths),
                entry,
                collision,
                tracking=self.tracking,
            )

        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(self.epoch), index])
        )
        # 1. Semantic policy decides what the sample teaches.
        start, stop, entry_index, collision_index, mode = choose_crop(
            len(paths), entry, collision, row.get("native_fps"), self.temporal, rng
        )
        semantic_length = stop - start
        # 2. Memory cap bounds the autograd graph, leaving the labels alone.
        start, stop, entry_index, collision_index, capped = enforce_max_train_frames(
            start, stop, entry_index, collision_index, mode, self.max_frames, rng
        )
        # 3. Appearance augmentation follows the final temporal selection.
        flip = sample_horizontal_flip(self.augmentation, rng)
        photometric = sample_photometric(self.augmentation, rng)
        generator = torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))
        item = joint_item(
            row,
            paths,
            frame_ids,
            records,
            start,
            stop,
            entry_index,
            collision_index,
            flip=flip,
            photometric=photometric,
            tracking=self.tracking,
            generator=generator,
        )
        # Cheap per-sample counters; the trainer averages them per interval.
        item["temporal_stats"] = {
            "original_temporal_length": float(len(paths)),
            "semantic_temporal_length": float(semantic_length),
            "final_temporal_length": float(stop - start),
            "memory_crop_applied_fraction": float(capped),
        }
        return item


def joint_collate(items):
    """Pad every temporal tensor, including the shared augmented RGB stack."""
    max_time = max(len(item["time_valid"]) for item in items)
    output = {}
    for name in (
        "rgb",
        "roi_boxes",
        "geometry",
        "object_valid",
        "time_valid",
        "local_time",
        "frame_seconds",
    ):
        if name not in items[0]:
            continue
        first = items[0][name]
        padded = first.new_zeros((len(items), max_time, *first.shape[1:]))
        for i, item in enumerate(items):
            padded[i, : len(item[name])] = item[name]
        output[name] = padded
    for name in ("entry_index", "collision_index", "entry_side", "evasion"):
        if name in items[0]:
            output[name] = torch.tensor([item[name] for item in items])
    for name in (
        "frame_ids",
        "sample_id",
        "source_id",
        "track_ids",
        "flipped",
        "temporal_stats",
    ):
        if name in items[0]:
            output[name] = [item[name] for item in items]
    return output
