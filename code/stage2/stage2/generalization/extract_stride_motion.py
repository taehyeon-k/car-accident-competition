"""Frame-rate robustness benchmark inputs: global (14-d) and residual (36-d) motion recomputed on each clip's frames[::k].

A clip seen at 1/k of its native frame rate = its native frames with stride k. Motion is computed between CONSECUTIVE RETAINED frames
(exactly what a lower-FPS video would give) with the same code as extract_motion.py / extract_residual.py (separate frame chains,
since cv2.phaseCorrelate modifies its inputs in place). Output cache_stride/k{k}/<id>.npz: kept (native frame indices), motion [n,14],
residual [n,36] (row 0 zeros). No FPS is read.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments.extract_motion import load, W, H
from stage2.aux_signal_experiments.extract_residual import camera, sq, ssq, DIM as RDIM
from stage2.spotting_experiments.extract_features import indexed_frames

OUT = C.REPO / "stage2/generalization/cache_stride"


def clip(args):
    sid, frames_dir, k, o = args if len(args) == 4 else (*args, 0)
    target = OUT / (f"k{k}" if o == 0 else f"k{k}o{o}") / f"{sid}.npz"  # offset o: native frames[o::k]
    if target.exists(): return sid
    cv2.setNumThreads(1)
    pairs = indexed_frames(frames_dir)[o::k]
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    n = len(pairs); mot = np.zeros((n, 14), np.float32); res = np.zeros((n, RDIM), np.float32)
    prev = prev_r = None
    for t, (_, path) in enumerate(pairs):
        g0 = load(path); g, gr_ = g0.copy(), g0.copy()
        if prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(prev, g, win)
            al = cv2.warpAffine(g, np.float32([[1, 0, -dx], [0, 1, -dy]]), (W, H), borderMode=cv2.BORDER_REFLECT)
            d = np.abs(al - prev)[3:-3, 3:-3]
            grid = [d[i * d.shape[0] // 3:(i + 1) * d.shape[0] // 3, j * d.shape[1] // 3:(j + 1) * d.shape[1] // 3].mean() for i in range(3) for j in range(3)]
            mot[t] = [dx, dy, np.hypot(dx, dy), resp, d.mean(), *grid]
            m, inl, of, orr = camera(prev_r, gr_, win)
            a2 = cv2.warpAffine(gr_, m, (W, H), borderMode=cv2.BORDER_REFLECT); r = np.abs(a2 - prev_r)[4:-4, 4:-4]; h, w = r.shape
            gr = [r[i * h // 5:(i + 1) * h // 5, j * w // 5:(j + 1) * w // 5].mean() for i in range(5) for j in range(5)]
            sc = float(np.hypot(m[0, 0], m[1, 0])); rot = float(np.arctan2(m[1, 0], m[0, 0]))
            res[t, 0:2] = sq(np.array([r.mean(), np.percentile(r, 95)])); res[t, 2:27] = sq(np.array(gr))
            res[t, 27] = ssq(r[:, :w // 2].mean() - r[:, w // 2:].mean()); res[t, 28] = sq(r[h // 2:, w // 3:2 * w // 3].mean())
            res[t, 29:33] = ssq(np.array([sc - 1, rot, m[0, 2], m[1, 2]])); res[t, 33:36] = [inl, of, np.log1p(orr)]
        prev, prev_r = g, gr_
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, kept=np.array([i for i, _ in pairs], np.int64), motion=mot, residual=res)
    return sid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--strides", type=int, nargs="+", default=[2, 3]); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--offsets", action="store_true", help="also extract every non-zero start offset o = 1..k-1 (frames[o::k])")
    a = ap.parse_args()
    jobs = [(r["sample_id"], r["frames_dir"], k, o) for k in a.strides for o in (range(k) if a.offsets else [0])
            for r in sorted(C.rows("all"), key=lambda r: -int(r["num_frames"]))]
    with ProcessPoolExecutor(a.workers) as ex:
        for n, sid in enumerate(ex.map(clip, jobs), 1):
            if n % 50 == 0: print(n, len(jobs), sid, flush=True)
    print("done", flush=True)


if __name__ == "__main__": main()
