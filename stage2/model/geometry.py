"""Native-coordinate geometry, robust depth features and tubelet aggregation."""

from __future__ import annotations
import numpy as np
from scipy.stats import rankdata
import torch
import torch.nn as nn
from .tracking import Track

EPS = 1e-8


def _slope(values: list[float]) -> float:
    """Least-squares log-area slope over observation indices (never seconds)."""
    if len(values) < 2:
        return 0.0
    x = np.arange(
        len(values),
        dtype=np.float32,
    )
    # For <=4 values this closed-form fit avoids repeated polynomial/SVD setup.
    centered = x - x.mean()
    return float(centered @ np.log(np.asarray(values) + EPS) / (centered @ centered))


def build_geometry(
    tracks: list[Track],
    depth_maps: list[np.ndarray] | None,
    sizes: list[tuple[int, int]],
    length: int,
    slots: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return [T,N,9] features and [T,N] observation validity.

    Depth maps must already be in native image coordinates with larger values
    meaning closer. Compute scene statistics once per frame, not per object.
    """
    if (depth_maps is not None and len(depth_maps) != length) or len(sizes) != length:
        raise ValueError("Depth maps and sizes must align with every temporal position")
    geo = np.zeros(
        (length, slots, 9),
        np.float32,
    )
    mask = np.zeros(
        (length, slots),
        bool,
    )
    depth_statistics = []
    for depth, (width, height) in zip(
        depth_maps or [],
        sizes,
    ):
        if depth.shape != (height, width) or not np.isfinite(depth).all():
            raise ValueError("Depth must be a finite native-resolution map")
        median = float(np.median(depth))
        mad = max(
            float(np.median(np.abs(depth - median))),
            1e-6,
        )
        depth_statistics.append((median, mad))

    for slot, track in enumerate(tracks[:slots]):
        history = []
        for t, det in sorted(track.observations.items()):
            x1, y1, x2, y2 = det.box
            w, h = sizes[t]
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError("Track boxes must be clipped to native image bounds")
            area = (x2 - x1) * (y2 - y1) / (w * h)
            history.append(area)
            if depth_maps is None:
                if det.proximity is None or not np.isfinite(det.proximity):
                    raise ValueError(
                        "Cached detections require finite normalized proximity"
                    )
                normalized_proximity = det.proximity
            else:
                d = depth_maps[t]
                ix1, ix2 = int(x1 + 0.25 * (x2 - x1)), int(x2 - 0.25 * (x2 - x1))
                iy1, iy2 = int(y1 + 0.55 * (y2 - y1)), int(y2 - 0.1 * (y2 - y1))
                patch = d[
                    max(
                        0,
                        iy1,
                    ) : min(
                        d.shape[0],
                        iy2,
                    ),
                    max(
                        0,
                        ix1,
                    ) : min(
                        d.shape[1],
                        ix2,
                    ),
                ]
                median, mad = depth_statistics[t]
                q = float(np.median(patch)) if patch.size else median
                normalized_proximity = (q - median) / mad
            delta = (
                0.0
                if len(history) == 1
                else float(np.log(history[-1] + EPS) - np.log(history[-2] + EPS))
            )
            geo[t, slot] = [
                (x1 + x2) / (2 * w),
                y2 / h,
                (x2 - x1) / w,
                (y2 - y1) / h,
                area,
                normalized_proximity,
                0.0,
                delta,
                _slope(history[-4:]),
            ]
            mask[t, slot] = True
    # rank is within-frame and only observed tracks
    for t in range(length):
        active = np.flatnonzero(mask[t])
        if len(active) == 1:
            geo[t, active, 6] = 0.5
        elif len(active) > 1:
            ranks = rankdata(
                geo[t, active, 5],
                method="average",
            )
            geo[t, active, 6] = (ranks - 1) / (len(active) - 1)
    return geo, mask


def tubelet_geometry(
    frame: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average first 7 features, take latest observed looming features."""
    b, t, n, _ = frame.shape
    assert t == 32
    f = frame.masked_fill(
        ~mask[..., None],
        0,
    ).reshape(
        b,
        16,
        2,
        n,
        9,
    )
    m = mask.reshape(
        b,
        16,
        2,
        n,
    )
    out = torch.zeros_like(f[:, :, 0])
    valid = m.any(2)
    denom = m.sum(2).clamp_min(1).unsqueeze(-1)
    out[..., :7] = (f[..., :7] * m.unsqueeze(-1)).sum(2) / denom
    latest = torch.where(
        m[:, :, 1].unsqueeze(-1),
        f[:, :, 1, ..., 7:],
        f[:, :, 0, ..., 7:],
    )
    out[..., 7:] = latest
    return out, valid


class GeometryMLP(nn.Module):
    """Embed nine normalized scalar channels into 128 geometry dimensions."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "center",
            torch.zeros(9),
        )
        self.register_buffer(
            "scale",
            torch.ones(9),
        )
        self.net = nn.Sequential(
            nn.Linear(
                9,
                64,
            ),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(
                64,
                128,
            ),
            nn.LayerNorm(128),
        )

    def forward(
        self,
        x,
    ):
        return self.net((x - self.center) / self.scale)

    @torch.no_grad()
    def set_statistics(
        self,
        statistics: dict,
    ) -> None:
        """Copy training-only statistics into checkpointed model buffers."""
        center = statistics["center"]
        scale = statistics["scale"]
        if (
            center.shape != (9,)
            or scale.shape != (9,)
            or not torch.isfinite(center).all()
            or not torch.isfinite(scale).all()
            or not (scale > 0).all()
        ):
            raise ValueError("Invalid nine-channel geometry statistics")
        self.center.copy_(center)
        self.scale.copy_(scale)
