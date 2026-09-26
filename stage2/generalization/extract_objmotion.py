"""Object-level independent motion per native frame (H4): where tracks that do NOT follow the camera are, and which way they move.

Same sparse Lucas-Kanade tracks + RANSAC similarity camera fit as extract_residual.py (160x90 grey, frame t vs t-1). Tracks whose
camera-compensated displacement exceeds 1 px are "independent". 12 dims (row 0 zeros):
  0-5   log1p(count) of independent tracks in a 3x2 grid: (left, centre, right) x (far = upper half, near = lower half)
  6-7   mean lateral velocity TOWARD the image centre of independent tracks in the left / right half (px, signed, >0 = inward)
  8-9   mean downward velocity (approaching) of independent tracks in the left / right half (px, signed)
  10    max independent residual speed in the near-centre cell (px)
  11    fraction of all tracks that are independent
Output cache_objmotion/k{k}/<id>.npy float32 [n, 12] for native frames[::k] (k = 1, 2, 3; consecutive retained frames). No FPS.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments.extract_motion import load, W, H
from stage2.spotting_experiments.extract_features import indexed_frames

OUT = C.REPO / "stage2/generalization/cache_objmotion"
DIM = 12


def pair_features(prev, cur):
    f = np.zeros(DIM, np.float32)
    p8, c8 = (prev * 255).astype(np.uint8), (cur * 255).astype(np.uint8)
    pts = cv2.goodFeaturesToTrack(p8, maxCorners=200, qualityLevel=0.01, minDistance=5)
    if pts is None or len(pts) < 12: return f
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(p8, c8, pts, None, winSize=(15, 15), maxLevel=2)
    ok = st[:, 0] == 1
    if ok.sum() < 12: return f
    a, b = pts[ok, 0], nxt[ok, 0]
    m, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if m is None: return f
    pred = a @ m[:, :2].T + m[:, 2]          # where the camera alone would move each point (prev -> cur)
    resid = b - pred; speed = np.linalg.norm(resid, axis=1); ind = speed > 1.0
    f[11] = ind.mean()
    if not ind.any(): return f
    x, y, dx, dy, sp = a[ind, 0], a[ind, 1], resid[ind, 0], resid[ind, 1], speed[ind]
    col = np.clip((x / (W / 3)).astype(int), 0, 2); row = (y >= H / 2).astype(int)
    for c in range(3):
        for r in range(2): f[c * 2 + r] = np.log1p(((col == c) & (row == r)).sum())
    left, right = x < W / 2, x >= W / 2
    if left.any(): f[6] = dx[left].mean(); f[8] = dy[left].mean()
    if right.any(): f[7] = -dx[right].mean(); f[9] = dy[right].mean()
    nc = (col == 1) & (row == 1)
    if nc.any(): f[10] = sp[nc].max()
    return f


def clip(args):
    sid, frames_dir, k = args
    target = OUT / f"k{k}" / f"{sid}.npy"
    if target.exists(): return sid
    cv2.setNumThreads(1)
    pairs = indexed_frames(frames_dir)[::k]; feats = np.zeros((len(pairs), DIM), np.float32); prev = None
    for t, (_, path) in enumerate(pairs):
        cur = load(path)
        if prev is not None: feats[t] = pair_features(prev, cur)
        prev = cur
    target.parent.mkdir(parents=True, exist_ok=True); np.save(target, feats)
    return sid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3]); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    jobs = [(r["sample_id"], r["frames_dir"], k) for k in a.strides for r in sorted(C.rows("all"), key=lambda r: -int(r["num_frames"]))]
    with ProcessPoolExecutor(a.workers) as ex:
        for n, sid in enumerate(ex.map(clip, jobs), 1):
            if n % 100 == 0: print(n, len(jobs), sid, flush=True)
    print("done", flush=True)


if __name__ == "__main__": main()
