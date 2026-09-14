"""Video-relative relevance and fixed object slots for the joint model."""

import numpy as np
import torch

GEOMETRY_CHANNELS = (
    "center_x",
    "bottom_y",
    "width",
    "height",
    "dx_per_frame",
    "dy_per_frame",
    "log_area_growth_per_frame",
    "detection_confidence",
    "track_continuity",
)
GEOMETRY_DIM = len(GEOMETRY_CHANNELS)


def ego_lane_score(x, y):
    y = float(np.clip(y, 0, 1))
    if y < 0.4:
        return 0.0
    half_width = 0.06 + (y - 0.4) / 0.6 * 0.24
    return float(np.clip(1 - abs(x - 0.5) / half_width, 0, 1))


def percentile_rank(value, reference, *, presorted=False):
    """Midrank for ties; zero motion/growth is always zero evidence."""
    if value <= 0 or not len(reference):
        return 0.0
    values = np.asarray(reference) if presorted else np.sort(reference)
    low = np.searchsorted(values, value, side="left")
    high = np.searchsorted(values, value, side="right")
    return float((low + high) / (2 * len(values)))


def object_tensors(
    tracks, sizes, max_objects=12, track_percentile=90, return_track_ids=False
):
    """Select tracks once per video. Motion geometry uses frames, never FPS.

    Geometry is the nine bbox/tracking channels named in ``GEOMETRY_CHANNELS``.
    Track ranking additionally uses ego-lane position, an absolute least-squares
    x slope over <=5 observations, positive growth and confidence; those ranking
    statistics are kept separate from the emitted geometry token.
    """
    if not 0 <= track_percentile <= 100:
        raise ValueError("track_percentile must be in [0, 100]")
    observations, motion_values, growth_values = {}, [], []
    for track in tracks:
        history, records = [], []
        for t, detection in sorted(track.observations.items()):
            if not np.isfinite(detection.score) or detection.score < 0.2:
                continue
            width, height = sizes[t]
            box = np.asarray(detection.box, dtype=np.float32)
            x1, y1, x2, y2 = box
            x, y = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
            bw, bh = (x2 - x1) / width, (y2 - y1) / height
            area = float(bw * bh)
            dx = dy = growth = 0.0
            motion = None
            if history:
                pt, px, py, pa = history[-1]
                dt = max(t - pt, 1)
                dx, dy = (x - px) / dt, (y - py) / dt
                growth = float(np.log(area + 1e-8) - np.log(pa + 1e-8)) / dt
                recent = history[-4:] + [(t, x, y, area)]
                frames = np.asarray([h[0] for h in recent], dtype=np.float64)
                xs = np.asarray([h[1] for h in recent], dtype=np.float64)
                motion = (
                    0.0
                    if np.ptp(xs) == 0
                    else float(abs(np.polyfit(frames - frames[0], xs, 1)[0]))
                )
                motion_values.append(motion)
                growth_values.append(max(growth, 0.0))
            confidence = float(detection.score)
            values = [
                x,
                y2 / height,
                bw,
                bh,
                float(np.clip(dx, -5, 5)),
                float(np.clip(dy, -5, 5)),
                float(np.clip(growth, -5, 5)),
                confidence,
                min(1.0, (len(history) + 1) / 5),
            ]
            # Ranking statistics travel beside the token, not inside it.
            records.append(
                (
                    t,
                    box,
                    values,
                    motion,
                    max(growth, 0.0),
                    bool(history),
                    confidence,
                    x,
                    y2 / height,
                )
            )
            history.append((t, x, y, area))
        if records:
            observations[track.id] = records

    motion_values = np.sort(motion_values)
    growth_values = np.sort(growth_values)

    def track_score(track_id):
        scores = []
        for *_, motion, growth, has_history, score, x, bottom in observations[track_id]:
            confidence = float(np.clip((score - 0.2) / 0.8, 0, 1))
            numerator = 0.2 * confidence + 0.4 * ego_lane_score(x, bottom)
            denominator = 0.6
            if has_history:
                numerator += 0.3 * percentile_rank(
                    motion, motion_values, presorted=True
                )
                numerator += 0.1 * percentile_rank(
                    growth, growth_values, presorted=True
                )
                denominator += 0.4
            scores.append(numerator / denominator)
        return float(np.percentile(scores, track_percentile))

    selected = sorted(observations, key=lambda i: (-track_score(i), i))[:max_objects]
    boxes = torch.zeros(len(sizes), max_objects, 4)
    geometry = torch.zeros(len(sizes), max_objects, len(GEOMETRY_CHANNELS))
    valid = torch.zeros(len(sizes), max_objects, dtype=torch.bool)
    track_ids = torch.full((max_objects,), -1, dtype=torch.long)
    for slot, track_id in enumerate(selected):
        track_ids[slot] = track_id
        for t, box, values, *_ in observations[track_id]:
            boxes[t, slot] = torch.from_numpy(box)
            geometry[t, slot] = torch.tensor(values)
            valid[t, slot] = True
    result = (boxes, geometry, valid)
    return (*result, track_ids) if return_track_ids else result
