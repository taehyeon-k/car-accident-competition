"""Shared constants and helpers for geometry-aware DINOv3 pretraining.

Every source is normalized to one 16:9 model input (``INPUT_HW``). Dense
targets live at stride 4 (``LABEL_HW``), the resolution the heads predict.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

INPUT_HW = (448, 800)  # 28 x 50 DINOv3 patches of 16 px
LABEL_STRIDE = 4
LABEL_HW = (INPUT_HW[0] // LABEL_STRIDE, INPUT_HW[1] // LABEL_STRIDE)  # 112 x 200
IGNORE = 255

# Compact Stage-2 object taxonomy (index 0 is background).
OBJECT_CLASSES = ["background", "car", "truck", "bus", "two_wheeler", "pedestrian"]
BDD_OBJECT_MAP = {
    "car": 1,
    "truck": 2,
    "train": 2,
    "bus": 3,
    "motor": 4,
    "bike": 4,
    "rider": 4,
    "person": 5,
}
VEHICLE_CLASSES = (1, 2, 3, 4)  # classes that receive a road-contact target
DRIVABLE_CLASSES = ["background", "ego_direct", "alternative"]

LOG = logging.getLogger("geometry_pretrain")


def setup_logging(path: str | Path | None = None, level=logging.INFO):
    handlers = [logging.StreamHandler()]
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def crop_to_aspect(image: np.ndarray, aspect: float = INPUT_HW[1] / INPUT_HW[0]):
    """Center-crop to the model aspect ratio; returns (crop, (x0, y0, w, h))."""
    h, w = image.shape[:2]
    if abs(w / h - aspect) < 0.01:
        return image, (0, 0, w, h)
    if w / h > aspect:
        nw = int(round(h * aspect))
        x0 = (w - nw) // 2
        return image[:, x0 : x0 + nw], (x0, 0, nw, h)
    nh = int(round(w / aspect))
    y0 = (h - nh) // 2
    return image[y0 : y0 + nh], (0, y0, w, nh)


def to_input(image: np.ndarray) -> np.ndarray:
    """RGB/BGR uint8 frame -> 448x800 with aspect-preserving center crop."""
    crop, _ = crop_to_aspect(image)
    return cv2.resize(crop, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_AREA)


def write_jpeg(path: Path, bgr: np.ndarray, quality: int = 92):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.jpg")
    cv2.imwrite(str(tmp), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    os.replace(tmp, path)


def atomic_savez(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def read_jsonl(path) -> list[dict]:
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


class DiskGuard:
    """Fail gracefully before the filesystem crosses the configured safety margin."""

    def __init__(self, path: str | Path, min_free_gb: float = 12.0):
        self.path = Path(path)
        self.min_free_gb = min_free_gb

    def free_gb(self) -> float:
        probe = self.path
        while not probe.exists():
            probe = probe.parent
        return shutil.disk_usage(probe).free / 1e9

    def check(self, context: str = ""):
        free = self.free_gb()
        if free < self.min_free_gb:
            raise RuntimeError(
                f"Disk guard: only {free:.1f} GB free (< {self.min_free_gb} GB safety margin) {context}"
            )
        return free


def dir_size_gb(path: str | Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total / 1e9


# ----------------------------------------------------------------------------
# Lane geometry helpers (shared by BDD and TuSimple preparation)
# ----------------------------------------------------------------------------


def bdd_poly_points(poly2d, samples_per_curve: int = 16) -> np.ndarray:
    """Expand BDD ``poly2d`` vertices (L = vertex, C = cubic control) to a polyline."""
    pts = [(float(p[0]), float(p[1])) for p in poly2d]
    types = [p[2] for p in poly2d]
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        if types[i + 1] == "C" and i + 3 < len(pts):
            p0, p1, p2, p3 = (np.array(pts[j]) for j in range(i, i + 4))
            t = np.linspace(0, 1, samples_per_curve)[1:, None]
            curve = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
            out.extend(map(tuple, curve))
            i += 3
        else:
            out.append(pts[i + 1])
            i += 1
    return np.asarray(out, np.float32)


def fit_lower_line(points: np.ndarray, image_h: float, min_points: int = 3):
    """Fit x = a*y + b on the lower half of a lane polyline (image coords)."""
    if len(points) < min_points:
        return None
    ys = points[:, 1]
    lower = points[ys >= np.median(ys)]
    if len(lower) < 2 or np.ptp(lower[:, 1]) < 0.04 * image_h:
        return None
    a, b = np.polyfit(lower[:, 1], lower[:, 0], 1)
    return float(a), float(b)


def robust_vanishing_point(lines, image_w: float, image_h: float, tol_frac: float = 0.025):
    """Median pairwise intersection of lane lines, accepted only if consistent.

    Returns (vp_x_norm, vp_y_norm, support) or None. Near-parallel pairs in image
    space (edges of one painted marking) are skipped because they are unstable.
    """
    pts = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a1, b1 = lines[i]
            a2, b2 = lines[j]
            if abs(a1 - a2) < 0.15:
                continue
            y = (b2 - b1) / (a1 - a2)
            x = a1 * y + b1
            if -0.25 * image_w < x < 1.25 * image_w and 0.05 * image_h < y < 0.85 * image_h:
                pts.append((x, y))
    if len(pts) < 2:
        return None
    pts = np.asarray(pts)
    med = np.median(pts, 0)
    dist = np.linalg.norm(pts - med, axis=1)
    inlier = dist < tol_frac * image_w
    if inlier.mean() < 0.5 or inlier.sum() < 2:
        return None
    vp = pts[inlier].mean(0)
    return float(vp[0] / image_w), float(vp[1] / image_h), int(inlier.sum())
