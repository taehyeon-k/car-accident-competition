"""Per-frame object-track features from cache_tracks (RF-DETR + ByteTrack), in the k1/k2/k3 layout read by train.py --objmotion
--obj-cache stage2/objtrack/cache_objfeat. For stride k the sequence is native frames[::k] and all temporal differences are taken
between consecutive RETAINED frames (like extract_stride_motion.py). Frame indices only; no FPS.

Per frame: the K = 4 largest road users (by box area), each described by 10 values
  cx - 0.5, |cx - 0.5|, bottom y, width, height, log area,
  growth = log area - log area at the track's previous retained frame (0 if none; clipped to +-1),
  approach = |cx_prev - 0.5| - |cx - 0.5| (> 0 = moving toward the image centre; clipped to +-0.2, x5),
  edge = signed distance of the nearest box edge to the centre line (< 0 = box covers the centre line),
  vru = person / bicycle / motorcycle
plus 4 global values: log1p(count), max growth, max approach, min edge over objects in the lower half.  -> 44 dims.
Empty slots are zeros with a presence flag folded into width (0).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path("/workspace/car-accident")
TRACKS = REPO / "stage2/objtrack/cache_tracks"
OUT = REPO / "stage2/objtrack/cache_objfeat"
K, PER, GLOBAL = 4, 10, 4
DIM = K * PER + GLOBAL
VRU = (1, 2, 4)


def clip_features(d, k):
    n = int(d["nframes"]); kept = np.arange(0, n, k)
    fr, box, cls, tr = d["frame"], d["box"], d["cls"], d["track"]
    m = fr % k == 0; fr, box, cls, tr = fr[m] // k, box[m], cls[m], tr[m]  # retained-frame index
    out = np.zeros((len(kept), DIM), np.float32)
    last = {}  # track id -> (retained index, log area, |cx - .5|)
    order = np.argsort(fr, kind="stable")
    fr, box, cls, tr = fr[order], box[order], cls[order], tr[order]
    bounds = np.searchsorted(fr, np.arange(len(kept) + 1))
    for t in range(len(kept)):
        s, e = bounds[t], bounds[t + 1]
        if s == e: continue
        b = box[s:e]; c = cls[s:e]; ids = tr[s:e]
        cx = (b[:, 0] + b[:, 2]) / 2; w = b[:, 2] - b[:, 0]; h = b[:, 3] - b[:, 1]
        la = np.log(np.clip(w * h, 1e-5, None)); dc = np.abs(cx - 0.5)
        growth = np.zeros(len(b), np.float32); appr = np.zeros(len(b), np.float32)
        for j, tid in enumerate(ids):
            if tid >= 0 and tid in last and t - last[tid][0] <= 3:
                growth[j] = np.clip(la[j] - last[tid][1], -1, 1); appr[j] = np.clip(last[tid][2] - dc[j], -.2, .2) * 5
        for j, tid in enumerate(ids):
            if tid >= 0: last[tid] = (t, la[j], dc[j])
        edge = np.where(cx < 0.5, 0.5 - b[:, 2], b[:, 0] - 0.5)
        feats = np.stack([cx - 0.5, dc, b[:, 3], w, h, la / 5, growth, appr, edge, np.isin(c, VRU).astype(np.float32)], 1)
        top = np.argsort(-(w * h))[:K]
        out[t, :len(top) * PER] = feats[top].reshape(-1)
        low = b[:, 3] > 0.5
        out[t, K * PER:] = [np.log1p(len(b)), growth.max(), appr.max(), edge[low].min() if low.any() else 0.5]
    return out


def one(args):
    p, k = args
    target = OUT / f"k{k}" / f"{p.stem}.npy"
    if target.exists(): return
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, clip_features(np.load(p), k))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    jobs = [(p, k) for k in (1, 2, 3) for p in sorted(TRACKS.glob("*.npz")) if not p.name.endswith(".tmp.npz")]
    with ProcessPoolExecutor(a.workers) as ex: list(ex.map(one, jobs))
    print("done", len(jobs))


if __name__ == "__main__": main()
