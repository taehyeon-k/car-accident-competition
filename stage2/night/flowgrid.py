"""Night campaign — camera-compensated dense-flow grid features (FG2 variant: GRID rows x cols cells + moving-pixel fractions; env FG_GRID="4x6") (cheap at inference: ~2 ms / frame on the 160 x 90 grey frames the
v8 runtime already decodes for its motion features).

Per retained frame t (vs the previous retained frame; row 0 = zeros), on 160 x 90 grey (IMREAD_REDUCED_GRAYSCALE_4 + INTER_AREA, as
the runtime's motion features): Farneback flow; a global affine camera motion fitted by least squares with 2 rounds of outlier
rejection; residual flow r = flow - affine. Features (65, signed-log squashed):
  3 x 5 grid cells: mean r_x, mean r_y, mean |r|, mean divergence of r (d r_x / dx + d r_y / dy; > 0 = expansion / looming)  (60)
  camera: translation x, y, scale - 1, rotation                                                                                 (4)
  95th percentile of |r|                                                                                                         (1)
Views: k1 = every native frame; k2 / k3 = the retained frames of generalization/cache_stride (kept), flow between consecutive
retained frames (the stride-view layout used by --objmotion / OBJ_CACHE). Output cache_flowgrid/k{k}/<sid>.npy float16.
No labels, no FPS. Usage: python -m stage2.night.flowgrid [--procs 10]
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from stage2.long_context_v2_experiments import common as C

import os
GR, GC = (int(v) for v in os.environ.get("FG_GRID", "3x5").split("x")); OUT = C.REPO / ("stage2/night/cache_flowgrid" if (GR, GC) == (3, 5) else f"stage2/night/cache_flowgrid{GR}x{GC}"); STRIDE = C.REPO / "stage2/generalization/cache_stride"; MW, MH = 160, 90
YY, XX = np.mgrid[0:MH, 0:MW].astype(np.float32); A_ = np.stack([XX.ravel() / MW - .5, YY.ravel() / MH - .5, np.ones(MW * MH, np.float32)], 1)
ssq = lambda x: np.sign(x) * np.log1p(20 * np.abs(x))


def gray(p):
    img = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    return cv2.resize(img, (MW, MH), interpolation=cv2.INTER_AREA)


def pair(a, b):
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


def one(args):
    sid, fdir = args
    if all((OUT / f"k{k}" / f"{sid}.npy").exists() for k in (1, 2, 3)): return
    cv2.setNumThreads(1); paths = sorted(Path(fdir).glob("*.jpg")); g = [gray(p) for p in paths]
    for k in (1, 2, 3):
        kept = np.arange(len(g)) if k == 1 else np.load(STRIDE / f"k{k}" / f"{sid}.npz")["kept"]
        feats = np.zeros((len(kept), GR * GC * (4 if (GR, GC) == (3, 5) else 5) + 5), np.float32)
        for t in range(1, len(kept)): feats[t] = pair(g[kept[t - 1]], g[kept[t]])
        np.save(OUT / f"k{k}" / f"{sid}.npy", ssq(feats).astype(np.float16))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=10); a = ap.parse_args()
    for k in (1, 2, 3): (OUT / f"k{k}").mkdir(parents=True, exist_ok=True)
    rows = sorted(C.rows("all"), key=lambda r: -int(r["num_frames"]))
    with Pool(a.procs) as p:
        for i, _ in enumerate(p.imap_unordered(one, [(r["sample_id"], r["frames_dir"]) for r in rows]), 1):
            if i % 25 == 0: print(i, len(rows), flush=True)
    print("done", flush=True)


if __name__ == "__main__": main()
