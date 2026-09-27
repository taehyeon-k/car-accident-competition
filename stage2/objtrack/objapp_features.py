"""Per-retained-frame object-appearance features (option 3) in the k1/k2/k3 layout read by train.py --objmotion --obj-cache.

From cache_objcrop (per native frame, K = 3 most prominent vehicles: DINOv3-S crop embedding, box, track id). Per slot (40 values):
  present, cx - 0.5, bottom y, width, height, log area / 5,
  growth  = log area - log area of the same track at the previous retained frame (0 if none; clipped +-1),
  approach = |cx_prev - 0.5| - |cx - 0.5| (x5, clipped +-1),
  32-d PCA of the crop embedding (PCA fitted without labels on embeddings pooled over all clips; scaled to unit variance).
For stride k the sequence is native frames[::k] and differences use consecutive retained frames. -> 120 dims. No FPS.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path("/workspace/car-accident")
CROPS = REPO / "stage2/objtrack/cache_objcrop"
OUT = REPO / "stage2/objtrack/cache_objapp"
K, P = 3, 32


def fit_pca(files, per_clip=200, seed=0):
    rng = np.random.default_rng(seed); xs = []
    for f in files:
        d = np.load(f); e = d["emb"][d["present"]].astype(np.float32)
        if len(e): xs.append(e[rng.choice(len(e), min(per_clip, len(e)), replace=False)])
    x = np.concatenate(xs); mu = x.mean(0); u, s, vt = np.linalg.svd(x - mu, full_matrices=False)
    comp = vt[:P]; scale = (s[:P] / np.sqrt(len(x) - 1)).astype(np.float32)
    return mu.astype(np.float32), comp.astype(np.float32), scale


def clip_features(d, mu, comp, scale, k):
    pres, box, tid = d["present"], d["box"], d["track"]; n = len(pres); kept = np.arange(0, n, k)
    z = ((d["emb"].astype(np.float32) - mu) @ comp.T) / scale; z[~pres] = 0
    out = np.zeros((len(kept), K * 40), np.float32); last = {}
    for t, f in enumerate(kept):
        for j in range(K):
            if not pres[f, j]: continue
            b = box[f, j]; cx = (b[0] + b[2]) / 2; w, h = b[2] - b[0], b[3] - b[1]; la = np.log(max(w * h, 1e-5)); dc = abs(cx - .5)
            g = a = 0.0; key = int(tid[f, j])
            if key >= 0 and key in last and t - last[key][0] <= 3:
                g = float(np.clip(la - last[key][1], -1, 1)); a = float(np.clip((last[key][2] - dc) * 5, -1, 1))
            out[t, j * 40:(j + 1) * 40] = [1.0, cx - .5, b[3], w, h, la / 5, g, a, *z[f, j]]
        for j in range(K):
            if pres[f, j] and tid[f, j] >= 0:
                b = box[f, j]; last[int(tid[f, j])] = (t, np.log(max((b[2] - b[0]) * (b[3] - b[1]), 1e-5)), abs((b[0] + b[2]) / 2 - .5))
    return out


def main():
    files = sorted(p for p in CROPS.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    mu, comp, scale = fit_pca(files); OUT.mkdir(exist_ok=True); np.savez(OUT / "pca.npz", mu=mu, comp=comp, scale=scale)
    for f in files:
        d = np.load(f)
        for k in (1, 2, 3):
            (OUT / f"k{k}").mkdir(exist_ok=True); np.save(OUT / f"k{k}" / f"{f.stem}.npy", clip_features(d, mu, comp, scale, k))
    print("done", len(files))


if __name__ == "__main__": main()
