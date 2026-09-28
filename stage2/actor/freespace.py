"""Deepening Exp 3 — free-space evidence for `evasion_space` (label-free per-frame features; the classifier is freespace_eval.py).

Road head of the geometry-adapted DINOv3-S (same model / preprocessing as ego_corridor.py; drivable classes 0 background, 1 ego lane,
2 alternative drivable; channel 3 lane-marking logit, channel 4 curb logit), near field = bottom 45 % of the 112-row label grid. Per
native frame (16 values):
  0 ego defined fraction of near rows          1 mean ego width / 200
  2 / 3  left / right alternative-drivable width beside the ego lane (pixels in the row / ego width, clipped to 3), near-row mean
  4 / 5  left / right curb within one ego width of the ego boundary (fraction of near rows)
  6 alternative-drivable fraction of the near field      7 background fraction of the near field
  8 / 9 / 10  vehicles with bottom y > .55 whose centre is left of / inside / right of the ego corridor (cache_corridor, cache_tracks)
  11 / 12 / 13  largest log box area of those vehicles (/ 5, 0 = none; left / inside / right)
  14 / 15  lane-marking pixels left / right within one ego width (fraction of near rows)
Output cache_freespace/<sid>.npy float16 [n, 16]. No labels, no FPS. Usage: python -m stage2.actor.freespace
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.objtrack.ego_corridor import Frames
from stage2.objtrack.objlane_features import lane_at
from stage2.geometry_pretrain.visualize import load_run
from stage2.aux_signal_experiments.pseudo_lane import RUN
from stage2.geometry_pretrain.pseudo_labels.teachers import MEAN, STD

OUT = C.REPO / "stage2/actor/cache_freespace"; TR = C.REPO / "stage2/objtrack/cache_tracks"; CORR = C.REPO / "stage2/objtrack/cache_corridor"
VEH = (2, 3, 4, 6, 8); NEAR = 0.55


def road_feats(drv, lane, curb):
    """drv [T,112,200] argmax classes, lane / curb bool -> [T, 8] + [T, 2]"""
    T, H, W = drv.shape; y0 = int(H * NEAR); out = np.zeros((T, 10), np.float32); cols = np.arange(W)
    for t in range(T):
        d = drv[t, y0:]; ego = d == 1; alt = d == 2; ok = ego.sum(1) >= 3; rows = np.flatnonzero(ok)
        out[t, 6] = alt.mean(); out[t, 7] = (d == 0).mean(); out[t, 0] = ok.mean()
        if not len(rows): continue
        L = np.array([cols[ego[r]].min() for r in rows]); R = np.array([cols[ego[r]].max() for r in rows]); w = np.maximum(R - L + 1, 1)
        out[t, 1] = w.mean() / W
        la = np.array([alt[r, :L[i]].sum() for i, r in enumerate(rows)]) / w; ra = np.array([alt[r, R[i] + 1:].sum() for i, r in enumerate(rows)]) / w
        out[t, 2], out[t, 3] = np.clip(la, 0, 3).mean(), np.clip(ra, 0, 3).mean()
        cu, ln = curb[t, y0:], lane[t, y0:]
        out[t, 4] = np.mean([cu[r, max(0, L[i] - w[i]):L[i]].any() for i, r in enumerate(rows)])
        out[t, 5] = np.mean([cu[r, R[i] + 1:R[i] + 1 + w[i]].any() for i, r in enumerate(rows)])
        out[t, 8] = np.mean([ln[r, max(0, L[i] - w[i]):L[i]].any() for i, r in enumerate(rows)])
        out[t, 9] = np.mean([ln[r, R[i] + 1:R[i] + 1 + w[i]].any() for i, r in enumerate(rows)])
    return out


def vehicle_feats(sid, n):
    d = np.load(TR / f"{sid}.npz"); corr = np.load(CORR / f"{sid}.npz"); out = np.zeros((n, 6), np.float32)
    Lc, Rc, row_y = corr["left"].astype(np.float32), corr["right"].astype(np.float32), corr["row_y"]
    for f, b, c in zip(d["frame"], d["box"], d["cls"]):
        if c not in VEH: continue
        x1, y1, x2, y2 = [float(v) for v in b]
        if y2 < NEAR: continue
        lane = lane_at(Lc[f], Rc[f], row_y, y2)
        if not lane: continue
        cx = (x1 + x2) / 2; side = 0 if cx < lane[0] else (2 if cx > lane[1] else 1)
        out[f, side] += 1; out[f, 3 + side] = max(out[f, 3 + side], (np.log(max((x2 - x1) * (y2 - y1), 1e-5)) + 12) / 5)
    return out


@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=True); model, _ = load_run(RUN); dev = torch.device("cuda")
    rows = sorted(C.rows("all"), key=lambda r: -int(r["num_frames"]))
    for n_, r in enumerate(rows, 1):
        sid = r["sample_id"]; target = OUT / f"{sid}.npy"
        if target.exists(): continue
        paths = sorted(Path(r["frames_dir"]).glob("*.jpg")); drv, lane, curb = [], [], []
        for x, _, _ in torch.utils.data.DataLoader(Frames(paths), batch_size=48, num_workers=3):
            x = (x.to(dev).float() / 255.0 - MEAN.to(dev)) / STD.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model.forward_static(x, tasks=["road"])["road"].float()
            drv.append(p[:, :3].argmax(1).cpu().numpy().astype(np.int8)); lane.append((p[:, 3] > 0).cpu().numpy()); curb.append((p[:, 4] > 0).cpu().numpy())
        rf = road_feats(np.concatenate(drv), np.concatenate(lane), np.concatenate(curb)); vf = vehicle_feats(sid, len(paths))
        np.save(target, np.concatenate([rf[:, :8], vf, rf[:, 8:]], 1).astype(np.float16))
        if n_ % 25 == 0: print(n_, len(rows), sid, flush=True)
    print("done", flush=True)


if __name__ == "__main__": main()
