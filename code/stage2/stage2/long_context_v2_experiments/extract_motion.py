"""Per-native-frame motion descriptors (CPU, no FPS): frame t vs t-1 on 160x90 grayscale.

14 dims: phase-correlation dx, dy, |shift|, peak response; aligned mean |diff|; 3x3 grid aligned mean |diff|.
Output <out>/<sample_id>.motion.npy float32 [num_frames, 14] (row 0 = zeros).
"""
from __future__ import annotations

import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from stage2.spotting_experiments.extract_features import indexed_frames

W, H = 160, 90


def load(path):
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def clip(args):
    sample_id, frames_dir, out = args
    target = Path(out) / f"{sample_id}.motion.npy"
    if target.exists(): return sample_id
    cv2.setNumThreads(1)
    pairs = indexed_frames(frames_dir)
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    feats = np.zeros((len(pairs), 14), np.float32)
    prev = load(pairs[0][1])
    for t in range(1, len(pairs)):
        cur = load(pairs[t][1])
        (dx, dy), resp = cv2.phaseCorrelate(prev, cur, win)
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        aligned = cv2.warpAffine(cur, m, (W, H), borderMode=cv2.BORDER_REFLECT)
        d = np.abs(aligned - prev)[3:-3, 3:-3]
        grid = [d[i * d.shape[0] // 3:(i + 1) * d.shape[0] // 3, j * d.shape[1] // 3:(j + 1) * d.shape[1] // 3].mean()
                for i in range(3) for j in range(3)]
        feats[t] = [dx, dy, np.hypot(dx, dy), resp, d.mean(), *grid]
        prev = cur
    tmp = target.with_suffix(".tmp.npy"); np.save(tmp, feats); os.replace(tmp, target)
    return sample_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/workspace/data/stage2/manifests/all.jsonl")
    ap.add_argument("--out", default="stage2/long_context_v2_experiments/cache_dense")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(a.manifest)]
    jobs = [(r["sample_id"], r["frames_dir"], a.out) for r in sorted(rows, key=lambda r: -int(r["num_frames"]))]
    with ProcessPoolExecutor(a.workers) as ex:
        for k, sid in enumerate(ex.map(clip, jobs), 1):
            if k % 25 == 0: print(k, sid, flush=True)
    print("done", len(jobs), flush=True)


if __name__ == "__main__": main()
