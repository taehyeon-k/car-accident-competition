"""Deterministic Hungarian association for persistent object tracks."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


@dataclass
class Detection:
    """One native-coordinate xyxy box with a canonical COCO class name."""

    box: np.ndarray
    score: float
    label: str
    proximity: float | None = None


@dataclass
class Track:
    """Persistent identity with sparse observed frames; gaps are not detections."""

    id: int
    label: str
    observations: dict[int, Detection] = field(default_factory=dict)
    last_t: int = -1


def iou(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    wh = np.maximum(
        0.0,
        np.minimum(
            a[2:],
            b[2:],
        )
        - np.maximum(
            a[:2],
            b[:2],
        ),
    )
    inter = wh[0] * wh[1]
    return float(
        inter
        / max(
            1e-8,
            (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter,
        )
    )


def center_distance(
    a: np.ndarray,
    b: np.ndarray,
    diagonal: float,
) -> float:
    return float(
        np.linalg.norm((a[:2] + a[2:]) / 2 - (b[:2] + b[2:]) / 2)
        / max(
            diagonal,
            1e-8,
        )
    )


class HungarianTracker:
    """Associate compatible detections with explicit geometric rejection gates."""

    def __init__(
        self,
        max_gap: int = 2,
        max_center_distance: float = 0.20,
        max_match_cost: float = 0.65,
        detection_threshold: float = 0.20,
    ):
        self.max_gap, self.max_center_distance, self.max_match_cost = (
            max_gap,
            max_center_distance,
            max_match_cost,
        )
        self.detection_threshold = detection_threshold
        if max_gap < 0 or not 0 <= detection_threshold <= 1:
            raise ValueError("Invalid tracking gap or confidence threshold")

    def track(
        self,
        frames: list[list[Detection]],
        sizes: list[tuple[int, int]],
    ) -> list[Track]:
        """Track chronologically; sizes contain (width, height) for each frame."""
        if len(frames) != len(sizes):
            raise ValueError("Frame detections and image sizes must align")
        tracks: list[Track] = []
        active_tracks: list[Track] = []
        next_id = 0
        for t, detections in enumerate(frames):
            width, height = sizes[t]
            if width <= 0 or height <= 0:
                raise ValueError("Native image dimensions must be positive")
            filtered = []
            for detection in detections:
                if (
                    detection.label not in VEHICLE_CLASSES
                    or not np.isfinite(detection.score)
                    or detection.score < self.detection_threshold
                ):
                    continue
                box = np.asarray(
                    detection.box,
                    dtype=np.float64,
                )
                if box.shape != (4,) or not np.isfinite(box).all():
                    continue
                box = np.clip(
                    box,
                    [0, 0, 0, 0],
                    [width, height, width, height],
                )
                if np.all(box[2:] > box[:2]):
                    filtered.append(
                        Detection(
                            box,
                            detection.score,
                            detection.label,
                            detection.proximity,
                        )
                    )
            detections = filtered

            # Only retain live tracks for association; history stays in tracks.
            live = [x for x in active_tracks if t - x.last_t <= self.max_gap + 1]
            active_tracks = live.copy()
            candidates = [(x, x.observations[x.last_t]) for x in live]
            cost = np.full(
                (len(candidates), len(detections)),
                1e6,
                dtype=np.float64,
            )
            diag = float(np.hypot(*sizes[t]))
            if candidates and detections:
                previous = np.stack([det.box for _, det in candidates])
                current = np.stack([det.box for det in detections])
                # Broadcast all pairwise IoUs and center distances together.
                intersection_wh = np.maximum(
                    0,
                    np.minimum(
                        previous[:, None, 2:],
                        current[None, :, 2:],
                    )
                    - np.maximum(
                        previous[:, None, :2],
                        current[None, :, :2],
                    ),
                )
                intersection = intersection_wh.prod(axis=-1)
                previous_area = (previous[:, 2:] - previous[:, :2]).prod(axis=-1)
                current_area = (current[:, 2:] - current[:, :2]).prod(axis=-1)
                overlap = intersection / np.maximum(
                    previous_area[:, None] + current_area[None, :] - intersection,
                    1e-8,
                )
                centers = (previous[:, :2] + previous[:, 2:]) / 2
                new_centers = (current[:, :2] + current[:, 2:]) / 2
                distance = (
                    np.linalg.norm(
                        centers[:, None] - new_centers[None, :],
                        axis=-1,
                    )
                    / diag
                )
                values = 0.6 * (1 - overlap) + 0.4 * distance
                compatible = (
                    np.asarray([track.label for track, _ in candidates])[:, None]
                    == np.asarray([det.label for det in detections])[None, :]
                )
                admissible = (
                    compatible
                    & (distance <= self.max_center_distance)
                    & (values <= self.max_match_cost)
                )
                cost[admissible] = values[admissible]
            used = set()
            if cost.size:
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(
                    rows,
                    cols,
                ):
                    if cost[r, c] >= 1e6:
                        continue
                    track, _ = candidates[r]
                    track.observations[t] = detections[c]
                    track.last_t = t
                    used.add(c)
            for c, d in enumerate(detections):
                if c not in used:
                    tracks.append(
                        Track(
                            next_id,
                            d.label,
                            {t: d},
                            t,
                        )
                    )
                    active_tracks.append(tracks[-1])
                    next_id += 1
        return tracks


def rank_tracks(
    tracks: list[Track],
    total_frames: int,
    sizes: list[tuple[int, int]],
    loom_clip: float = np.log(2),
    max_tracks: int = 12,
) -> list[Track]:
    """Select top tracks using the fixed five-term relevance formula."""
    if total_frames <= 0 or not np.isfinite(loom_clip) or loom_clip <= 0:
        raise ValueError("Track ranking needs positive frame count and looming scale")

    def score(track: Track) -> float:
        areas = []
        bottoms = []
        conf = []
        for t, d in sorted(track.observations.items()):
            w, h = sizes[t]
            box = d.box
            areas.append((box[2] - box[0]) * (box[3] - box[1]) / (w * h))
            bottoms.append(box[3] / h)
            conf.append(d.score)
        deltas = (
            np.maximum(
                0,
                np.diff(np.log(np.asarray(areas) + 1e-8)),
            )
            if len(areas) > 1
            else np.zeros(1)
        )
        loom = float(
            np.clip(
                deltas.max(initial=0) / loom_clip,
                0,
                1,
            )
        )
        return (
            0.30 * max(areas)
            + 0.20 * max(bottoms)
            + 0.20 * (len(areas) / total_frames)
            + 0.10 * float(np.mean(conf))
            + 0.20 * loom
        )

    return sorted(
        tracks,
        key=score,
        reverse=True,
    )[:max_tracks]
