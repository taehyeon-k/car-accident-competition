"""Stage2_experiments D1 / D5 — per-retained-frame actor features for the object branch (K slots x D, obj-cache k-layout).

Per retained frame of native frames[::k], for the collision actor (cache_actor_rank; identity kept by track linking):
  geometry (8): present, cx - .5, bottom y, width, height, log area / 5, growth, approach      (objlane_features layout)
  lane (7):     lane defined, intrusion depth, overlap, centre offset, d(depth), d(overlap), lateral velocity toward the lane centre
  appearance:   a50 / a48p3: 32-d objapp PCA of the TIGHT crop + its change (dz)
                a147 / a147c: 64-d PCA of the TIGHT crop + dz, 64-d PCA of the 2x CONTEXT crop + dz (high-res actor ROI, D5)
  extras (a50, a147*): interpolated-box flag, detector confidence
Layouts: a50 (K 1, D 50; D1), a48p3 (K 4, D 48: actor slot + OB_D's 3 prominence slots from cache_objlane_full; D1b),
         a147 (K 1, D 147; D5 global + ROI), a147c (a147 with geometry / lane zeroed except the present flag; D5 crop-only).
Caches: cache_actorfeat/<layout>/train/k{1,2,3} (actor from GT COLLISION; training clips), val/k1 (actor from the E4_sa out-of-fold
COLLISION; validation clips inside training runs), eval/{k1,k2,k3,k1c50,k1c25} (actor from the v8 out-of-fold COLLISION of that
entry_suite condition). PCA (64-d) fitted label-free on the GT-actor crops of all clips. No FPS.
Usage: python -m stage2.actor.actor_features [--layouts a50 a48p3 a147 a147c]
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np

from stage2.long_context_v2_experiments import common as C
from stage2.objtrack.objlane_features import lane_at, conf_lookup

A = C.REPO / "stage2/actor/cache_actor_rank"; CROP = C.REPO / "stage2/actor/cache_actorcrop"; CORR = C.REPO / "stage2/objtrack/cache_corridor"
TR = C.REPO / "stage2/objtrack/cache_tracks"; OUT = C.REPO / "stage2/actor/cache_actorfeat"; PROM = C.REPO / "stage2/objtrack/cache_objlane_full"
PCA32 = C.REPO / "stage2/objtrack/cache_objapp/pca.npz"; PCA64 = C.REPO / "stage2/actor/cache_actorcrop_pca64.npz"
DIM = {"a50": 50, "a48p3": 48, "a147": 147, "a147c": 147}
SPLITS = {"train": [("gt", 1, "k1"), ("gt", 2, "k2"), ("gt", 3, "k3")], "val": [("e4v", 1, "k1")],
          "eval": [("v8_k1", 1, "k1"), ("v8_k2", 2, "k2"), ("v8_k3", 3, "k3"), ("v8_crop50", 1, "k1c50"), ("v8_crop25", 1, "k1c25")]}


def fit_pca64():
    xs = {"tight": [], "ctx": []}
    for p in sorted(CROP.glob("*.npz")):
        d = np.load(p); i = d["idx_gt"]; i = i[i >= 0][::3]
        for k in xs: xs[k].append(d[k][i].astype(np.float32))
    out = {}
    for k, v in xs.items():
        x = np.concatenate(v); mu = x.mean(0); u, s, vt = np.linalg.svd(x - mu, full_matrices=False)
        comp = vt[:64]; z = (x - mu) @ comp.T; out[f"{k}_mu"], out[f"{k}_comp"], out[f"{k}_scale"] = mu, comp, z.std(0) + 1e-6
    np.savez(PCA64, **out); return out


def actor_vecs(sid, variant, k, layout, p32, p64):
    a = np.load(A / variant / f"{sid}.npz"); cr = np.load(CROP / f"{sid}.npz"); corr = np.load(CORR / f"{sid}.npz")
    conf = conf_lookup(np.load(TR / f"{sid}.npz")); idx = cr[f"idx_{variant}"]
    pres, box, tid = a["present"], a["box"], a["tid"]; n = len(pres); kept = np.arange(0, n, k)
    L, R, row_y = corr["left"].astype(np.float32), corr["right"].astype(np.float32), corr["row_y"]
    D = DIM[layout]; out = np.zeros((len(kept), D), np.float32); last = None
    for t, f in enumerate(kept):
        if not pres[f]: continue
        x1, y1, x2, y2 = [float(v) for v in box[f]]; cx = (x1 + x2) / 2; w, h = x2 - x1, y2 - y1
        la = np.log(max(w * h, 1e-5)); dc = abs(cx - .5)
        lane = lane_at(L[f], R[f], row_y, y2)
        if lane:
            Lb, Rb = lane; cc, hw = (Lb + Rb) / 2, (Rb - Lb) / 2
            lv = [1.0, float(np.clip(((x2 - Lb) if cx < cc else (Rb - x1)) / hw, -3, 3)), max(0.0, min(x2, Rb) - max(x1, Lb)) / max(w, 1e-4),
                  float(np.clip((cx - cc) / hw, -3, 3))]
        else: lv = [0.0] * 4
        tight = cr["tight"][idx[f]].astype(np.float32); ctx = cr["ctx"][idx[f]].astype(np.float32)
        if layout in ("a50", "a48p3"): app = [((tight - p32["mu"]) @ p32["comp"].T) / p32["scale"]]
        else: app = [((tight - p64["tight_mu"]) @ p64["tight_comp"].T) / p64["tight_scale"], ((ctx - p64["ctx_mu"]) @ p64["ctx_comp"].T) / p64["ctx_scale"]]
        g = ap = dd = dov = lat = 0.0; dz = [0.0] * len(app)
        if last is not None and t - last["t"] <= 3:
            g = float(np.clip(la - last["la"], -1, 1)); ap = float(np.clip((last["dc"] - dc) * 5, -1, 1))
            if lane and last["lane"]:
                dd = float(np.clip(lv[1] - last["lv"][1], -1, 1)); dov = float(np.clip(lv[2] - last["lv"][2], -1, 1)); lat = float(np.clip(abs(last["lv"][3]) - abs(lv[3]), -1, 1))
            dz = [float(np.linalg.norm(z - zp) / 8) for z, zp in zip(app, last["app"])]
        geo = [1.0, cx - .5, y2, w, h, la / 5, g, ap] + lv + [dd, dov, lat]
        if layout == "a147c": geo = [1.0] + [0.0] * 14
        vec = geo
        for z, d_ in zip(app, dz): vec = vec + list(z) + [d_]
        if layout != "a48p3": vec = vec + [1.0 if pres[f] == 2 else 0.0, conf.get((int(f), int(tid[f])), 0.0) if pres[f] == 1 else 0.0]
        out[t] = vec; last = {"t": t, "la": la, "dc": dc, "lane": bool(lane), "lv": lv, "app": app}
    return out


def one(args):
    sid, layout, p32, p64 = args
    for split, jobs in SPLITS.items():
        for variant, k, sub in jobs:
            if not (A / variant / f"{sid}.npz").exists(): continue
            v = actor_vecs(sid, variant, k, layout, p32, p64)
            if layout == "a48p3":  # actor slot first, then OB_D's 3 prominence slots (crop windows are sliced by the evaluator)
                v = np.concatenate([v, np.load(PROM / f"k{k}" / f"{sid}.npy")], 1)
            np.save(OUT / layout / split / sub / f"{sid}.npy", v.astype(np.float16))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layouts", nargs="+", default=list(DIM)); a = ap.parse_args()
    p32 = dict(np.load(PCA32)); p64 = dict(np.load(PCA64)) if PCA64.exists() else fit_pca64()
    for L in a.layouts:
        for split, jobs in SPLITS.items():
            for _, _, sub in jobs: (OUT / L / split / sub).mkdir(parents=True, exist_ok=True)
    sids = sorted(p.stem for p in (A / "gt").glob("*.npz"))
    with Pool(8) as p: p.map(one, [(s, L, p32, p64) for L in a.layouts for s in sids], chunksize=4)
    print("done", a.layouts, len(sids))


if __name__ == "__main__": main()
