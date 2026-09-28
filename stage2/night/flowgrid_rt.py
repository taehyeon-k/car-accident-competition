"""Inference-side flow-grid features for Stage 2 packages (identical computation to stage2/night/flowgrid.py; copied into the package
next to runtime.py). Input: the 160 x 90 grey frames of the given frame list (IMREAD_REDUCED_GRAYSCALE_4 + INTER_AREA). Output:
per-frame features [n, D] (row 0 = zeros), signed-log squashed and float16-rounded as in the training caches.
grid (3, 5) -> 65 features (FG); (4, 6) -> 125 features (FG2: + moving-pixel fraction per cell).
"""
import cv2
import numpy as np

MW, MH = 160, 90
YY, XX = np.mgrid[0:MH, 0:MW].astype(np.float32)
A_ = np.stack([XX.ravel() / MW - .5, YY.ravel() / MH - .5, np.ones(MW * MH, np.float32)], 1)


def ssq(x): return np.sign(x) * np.log1p(20 * np.abs(x))


def pair(a, b, grid=(3, 5)):
    GR, GC = grid
    f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 9, 3, 5, 1.1, 0)
    fx, fy = f[..., 0].ravel(), f[..., 1].ravel(); keep = np.ones(len(fx), bool)
    for _ in range(3):
        cx, *_ = np.linalg.lstsq(A_[keep], fx[keep], rcond=None); cy, *_ = np.linalg.lstsq(A_[keep], fy[keep], rcond=None)
        rx, ry = fx - A_ @ cx, fy - A_ @ cy; m = np.hypot(rx, ry)
        keep = m <= max(np.percentile(m, 80), 0.05)
    rx, ry = rx.reshape(MH, MW), ry.reshape(MH, MW); mag = np.hypot(rx, ry)
    div = np.gradient(rx, axis=1) + np.gradient(ry, axis=0); out = []
    for i in range(GR):
        for j in range(GC):
            s = (slice(i * MH // GR, (i + 1) * MH // GR), slice(j * MW // GC, (j + 1) * MW // GC))
            out += [rx[s].mean(), ry[s].mean(), mag[s].mean(), div[s].mean()] + ([float((mag[s] > 0.5).mean())] if (GR, GC) != (3, 5) else [])
    cam = [cx[2], cy[2], (cx[0] + cy[1]) / 2, (cy[0] - cx[1]) / 2]
    return np.array(out + cam + [np.percentile(mag, 95)], np.float32)


def flowgrid(grays, grid=(3, 5)):
    """grays: list of uint8 [90, 160] frames -> [n, D] float32 (float16-rounded, as the training caches)"""
    D = grid[0] * grid[1] * (4 if tuple(grid) == (3, 5) else 5) + 5; out = np.zeros((len(grays), D), np.float32)
    for t in range(1, len(grays)): out[t] = pair(grays[t - 1], grays[t], tuple(grid))
    return ssq(out).astype(np.float16).astype(np.float32)
