"""Ego-motion-compensated residual motion (residual_motion_features), per native frame t vs t-1. CPU only, no FPS.

Same 160x90 grayscale frames as long_context_v2_experiments.extract_motion. Dominant camera motion is a similarity transform
(translation + rotation + zoom) from sparse Lucas-Kanade tracks + RANSAC (cv2.estimateAffinePartial2D); fallback: phase
correlation translation. Frame t is warped onto t-1 and the absolute residual summarised.

36 dims (already squashed; row 0 = zeros):
  0-1   residual mean, residual p95                              log1p(100 x)
  2-26  5x5 grid residual means                                   log1p(100 x)
  27    left-right residual asymmetry (left half - right half)    signed log1p(100 |x|)
  28    centre-road residual (lower half, middle third)           log1p(100 x)
  29-32 camera zoom (s - 1), rotation (rad), tx, ty (pixels)      signed log1p(100 |x|)
  33    RANSAC inlier ratio                                        raw
  34    fraction of tracks inconsistent with camera motion        raw
  35    mean residual displacement of those tracks (pixels)       log1p(x)
Output <out>/<sample_id>.residual.npy float32 [num_frames, 36].
"""
from __future__ import annotations

import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from stage2.long_context_v2_experiments.extract_motion import load, W, H
from stage2.spotting_experiments.extract_features import indexed_frames

DIM = 36
sq = lambda x: np.log1p(100 * np.maximum(x, 0))
ssq = lambda x: np.sign(x) * np.log1p(100 * np.abs(x))


def camera(prev, cur, win):
    """2x3 matrix mapping cur -> prev coordinates, inlier ratio, outlier fraction, outlier residual magnitude."""
    pts = cv2.goodFeaturesToTrack((prev * 255).astype(np.uint8), maxCorners=200, qualityLevel=0.01, minDistance=5)
    if pts is not None and len(pts) >= 12:
        nxt, st, _ = cv2.calcOpticalFlowPyrLK((prev * 255).astype(np.uint8), (cur * 255).astype(np.uint8), pts, None,
                                             winSize=(15, 15), maxLevel=2)
        ok = st[:, 0] == 1
        if ok.sum() >= 12:
            a, b = pts[ok, 0], nxt[ok, 0]
            m, inl = cv2.estimateAffinePartial2D(b, a, method=cv2.RANSAC, ransacReprojThreshold=1.0)
            if m is not None:
                inl = inl[:, 0].astype(bool)
                pred = b @ m[:, :2].T + m[:, 2]
                res = np.linalg.norm(pred - a, axis=1)
                out = ~inl
                return m, float(inl.mean()), float(out.mean()), float(res[out].mean()) if out.any() else 0.0
    (dx, dy), _ = cv2.phaseCorrelate(prev, cur, win)
    return np.float32([[1, 0, -dx], [0, 1, -dy]]), 0.0, 0.0, 0.0


def clip(args):
    sample_id, frames_dir, out = args
    target = Path(out) / f"{sample_id}.residual.npy"
    if target.exists(): return sample_id
    cv2.setNumThreads(1)
    pairs = indexed_frames(frames_dir)
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    feats = np.zeros((len(pairs), DIM), np.float32)
    prev = load(pairs[0][1])
    for t in range(1, len(pairs)):
        cur = load(pairs[t][1])
        m, inl, out_frac, out_res = camera(prev, cur, win)
        aligned = cv2.warpAffine(cur, m, (W, H), borderMode=cv2.BORDER_REFLECT)
        d = np.abs(aligned - prev)[4:-4, 4:-4]; h, w = d.shape
        grid = [d[i * h // 5:(i + 1) * h // 5, j * w // 5:(j + 1) * w // 5].mean() for i in range(5) for j in range(5)]
        scale = float(np.hypot(m[0, 0], m[1, 0])); rot = float(np.arctan2(m[1, 0], m[0, 0]))
        feats[t, 0:2] = sq(np.array([d.mean(), np.percentile(d, 95)]))
        feats[t, 2:27] = sq(np.array(grid))
        feats[t, 27] = ssq(d[:, :w // 2].mean() - d[:, w // 2:].mean())
        feats[t, 28] = sq(d[h // 2:, w // 3:2 * w // 3].mean())
        feats[t, 29:33] = ssq(np.array([scale - 1, rot, m[0, 2], m[1, 2]]))
        feats[t, 33:36] = [inl, out_frac, np.log1p(out_res)]
        prev = cur
    tmp = target.with_suffix(".tmp.npy"); np.save(tmp, feats); os.replace(tmp, target)
    return sample_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/workspace/data/stage2/manifests/all.jsonl")
    ap.add_argument("--out", default="stage2/aux_signal_experiments/cache_residual")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in open(a.manifest)]
    jobs = [(r["sample_id"], r["frames_dir"], a.out) for r in sorted(rows, key=lambda r: -int(r["num_frames"]))]
    if a.limit: jobs = jobs[:a.limit]
    with ProcessPoolExecutor(a.workers) as ex:
        for k, sid in enumerate(ex.map(clip, jobs), 1):
            if k % 25 == 0: print(k, sid, flush=True)
    print("done", len(jobs), flush=True)


if __name__ == "__main__": main()
