"""Experiment 2 — per-object, ego-lane-relative features for the object/lane ENTRY branch (k1..k4 layout, --obj-cache).

Inputs: cache_objcrop (per native frame, K = 3 most prominent vehicles: box, track id, DINOv3-S crop embedding) and cache_corridor
(ego-lane left/right per label row per native frame, original-frame normalised coordinates). Per retained frame and slot:
  geometry (8): present, cx - .5, bottom y, width, height, log area / 5, growth, approach           (as objapp_features)
  lane (7):     corridor defined at the contact row, signed intrusion depth of the box edge nearest the lane centre past the lane
                boundary (/ half-width; > 0 = inside the ego lane), fraction of the box width inside the corridor, centre offset
                (cx - lane centre) / half-width, and their changes vs the same track at the previous retained frame:
                d(intrusion), d(overlap), lateral velocity toward the lane centre (-d|offset|)
  appearance (33, variant 'full' only): 32-d PCA of the crop embedding + its change vs the same track (L2)
Variants: geo (15 / slot -> 45) and full (48 / slot -> 144). Lane changes are ego-motion compensated by construction (the corridor
moves with the ego camera). Differences use consecutive retained frames of native frames[::k]. No labels, no FPS.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path("/workspace/car-accident")
CROPS = REPO / "stage2/objtrack/cache_objcrop"
CORR = REPO / "stage2/objtrack/cache_corridor"
PCA = REPO / "stage2/objtrack/cache_objapp/pca.npz"
K = 3


def lane_at(left, right, row_y, y):
    """corridor (L, R) at the row nearest to y, searching +-10 rows for a defined one; None if undefined."""
    r0 = int(np.abs(row_y - y).argmin())
    for d in range(0, 11):
        for r in (r0 - d, r0 + d):
            if 0 <= r < len(row_y) and np.isfinite(left[r]) and np.isfinite(right[r]) and right[r] - left[r] > 0.01:
                return float(left[r]), float(right[r])
    return None


def conf_lookup(tracks):
    """(native frame, track id) -> detector confidence, from cache_tracks."""
    return {(int(f), int(t)): float(s) for f, t, s in zip(tracks["frame"], tracks["track"], tracks["score"]) if t >= 0}


def clip_features(d, corr, k, variant, pca, conf=None):
    pres, box, tid = d["present"], d["box"], d["track"]; n = len(pres); kept = np.arange(0, n, k)
    L, R, row_y = corr["left"].astype(np.float32), corr["right"].astype(np.float32), corr["row_y"]
    if variant in ("full", "fullq"):
        z = ((d["emb"].astype(np.float32) - pca["mu"]) @ pca["comp"].T) / pca["scale"]; z[~pres] = 0
    D = {"geo": 15, "full": 48, "fullq": 51}[variant]; seen = {}
    out = np.zeros((len(kept), K * D), np.float32); last = {}
    for t, f in enumerate(kept):
        cur = {}
        for j in range(K):
            if not pres[f, j]: continue
            b = box[f, j]; x1, y1, x2, y2 = [float(v) for v in b]; cx = (x1 + x2) / 2; w, h = x2 - x1, y2 - y1
            la = np.log(max(w * h, 1e-5)); dc = abs(cx - .5); key = int(tid[f, j])
            lane = lane_at(L[f], R[f], row_y, y2)
            if variant == "fullq":  # lane reliability q = defined fraction of near-field corridor rows in this frame (current only)
                nr = slice(int(len(row_y) * 0.55), len(row_y)); q = float((np.isfinite(L[f][nr]) & np.isfinite(R[f][nr])).mean())
            if lane:
                Lb, Rb = lane; cc, hw = (Lb + Rb) / 2, (Rb - Lb) / 2
                depth = ((x2 - Lb) if cx < cc else (Rb - x1)) / hw
                overlap = max(0.0, min(x2, Rb) - max(x1, Lb)) / max(w, 1e-4); off = (cx - cc) / hw
                lv = [1.0, float(np.clip(depth, -3, 3)), overlap, float(np.clip(off, -3, 3))]
            else:
                lv = [0.0, 0.0, 0.0, 0.0]
            g = a = dd = dov = lat = dz = 0.0
            if key >= 0 and key in last and t - last[key]["t"] <= 3:
                p = last[key]; g = float(np.clip(la - p["la"], -1, 1)); a = float(np.clip((p["dc"] - dc) * 5, -1, 1))
                if lane and p["lane"]:
                    dd = float(np.clip(lv[1] - p["lv"][1], -1, 1)); dov = float(np.clip(lv[2] - p["lv"][2], -1, 1))
                    lat = float(np.clip(abs(p["lv"][3]) - abs(lv[3]), -1, 1))
                if variant in ("full", "fullq"): dz = float(np.linalg.norm(z[f, j] - p["z"]) / 8)
            vec = [1.0, cx - .5, y2, w, h, la / 5, g, a] + lv[:1] + lv[1:] + [dd, dov, lat]
            if variant == "fullq": vec[8:15] = [v * q for v in vec[8:15]]  # lane features scaled by lane reliability
            if variant in ("full", "fullq"): vec += list(z[f, j]) + [dz]
            if variant == "fullq":  # reliability inputs: detector confidence, causal persistence (last 10 retained frames), q
                hist = seen.setdefault(key, []); persist = sum(1 for tt in hist if t - tt <= 10) / 10.0
                vec += [conf.get((int(f), key), 0.0), min(persist, 1.0), q]
            out[t, j * D:(j + 1) * D] = vec
            cur[key] = {"t": t, "la": la, "dc": dc, "lane": bool(lane), "lv": lv, "z": z[f, j] if variant in ("full", "fullq") else None}
            if key >= 0: seen.setdefault(key, []).append(t)
        for key_, v in cur.items():
            if key_ >= 0: last[key_] = v
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", choices=["geo", "full", "fullq"], required=True); a = ap.parse_args()
    out = REPO / f"stage2/objtrack/cache_objlane_{a.variant}"; pca = dict(np.load(PCA))
    files = sorted(p for p in CROPS.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    for f in files:
        d = np.load(f); corr = np.load(CORR / f"{f.stem}.npz")
        conf = conf_lookup(np.load(REPO / "stage2/objtrack/cache_tracks" / f"{f.stem}.npz")) if a.variant == "fullq" else None
        for k in (1, 2, 3, 4):
            (out / f"k{k}").mkdir(parents=True, exist_ok=True); np.save(out / f"k{k}" / f"{f.stem}.npy", clip_features(d, corr, k, a.variant, pca, conf))
    print("done", a.variant, len(files))


if __name__ == "__main__": main()
