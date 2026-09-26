"""Per-sampled-frame geometry features from the geometry-DINOv3-S heads (the backbone the Stage-2 features come from).

For each clip's adaptive sampled frames (the positions the temporal model sees), at the heads' 112x200 resolution:
  grid (lower 70 % of the image, 3 rows x 5 cols = 15 cells), per cell the fraction of:
      ego-lane drivable, alternative drivable, vehicle (car/truck/bus/two-wheeler), vulnerable (pedestrian + two-wheeler),
      lane marking                                                                                 -> 75 dims
  corridor (time-smoothed ego-lane corridor as in pseudo_lane v2, lower 60 %):
      vehicle occupancy inside / left band / right band, max vehicle closeness (inverse depth) inside the corridor
      and anywhere in the lower image, alternative-drivable fraction left / right half, contact pixels inside corridor
                                                                                                   -> 8 dims
Output cache_geo/<sample_id>.geo.npz: frames [T], feats [T, 83] float16. Frame numbers / pixels only; no FPS.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from stage2.geometry_pretrain.visualize import load_run
from stage2.geometry_pretrain.pseudo_labels.teachers import MEAN, STD
from stage2.long_context_v2_experiments import common as C
from stage2.spotting_experiments.extract_features import indexed_frames
from .pseudo_lane import Frames, RUN, corridor
from scipy.ndimage import median_filter

OUT = C.REPO / "stage2/aux_signal_experiments/cache_geo"
DIM = 83


@torch.no_grad()
def maps(model, x):
    x = (x.cuda().float() / 255.0 - MEAN.cuda()) / STD.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = model.forward_static(x, tasks=["road", "objects", "contact", "depth"])
    road = p["road"].float(); drv = road[:, :3].argmax(1)
    obj = p["objects"].float().argmax(1)
    return {"ego": (drv == 1), "alt": (drv == 2), "veh": (obj >= 1) & (obj <= 4), "vul": (obj == 4) | (obj == 5),
            "lane": road[:, 3] > 0, "contact": p["contact"][:, 0].float() > 0, "depth": p["depth"][:, 0].float()}


def grid_feats(m):
    B, H, W = m["ego"].shape; y0 = int(H * 0.3); out = []
    for key in ("ego", "alt", "veh", "vul", "lane"):
        x = m[key][:, y0:].float()[:, None]
        out.append(torch.nn.functional.adaptive_avg_pool2d(x, (3, 5)).flatten(1))
    return torch.cat(out, 1).cpu().numpy()  # [B, 75]


def corridor_feats(ego, veh, contact, depth, alt):
    T, H, W = ego.shape; y0 = int(H * 0.4); cols = np.arange(W)
    left, right = corridor(ego)
    for arr in (left, right):
        med = np.nanmedian(arr, 0) if np.isfinite(arr).any() else np.full(H, np.nan)
        filled = np.where(np.isfinite(arr), arr, med[None])
        arr[:] = np.where(np.isfinite(filled), median_filter(np.nan_to_num(filled, nan=-1), size=(9, 1), mode="nearest"), np.nan)
        arr[arr < 0] = np.nan
    f = np.zeros((T, 8), np.float32)
    d = (depth - depth.reshape(T, -1).min(1)[:, None, None]) / (np.ptp(depth.reshape(T, -1), 1)[:, None, None] + 1e-6)  # per-frame 0..1
    for t in range(T):
        L, R = left[t, y0:], right[t, y0:]; ok = np.isfinite(L) & np.isfinite(R) & (R - L >= 4)
        v = veh[t, y0:]; dd = d[t, y0:]
        f[t, 4] = float((dd * v).max()) if v.any() else 0.0
        f[t, 5] = float(alt[t, y0:, :W // 2].mean()); f[t, 6] = float(alt[t, y0:, W // 2:].mean())
        if ok.mean() < 0.3: continue
        vv = v[ok]; Lk, Rk = L[ok][:, None], R[ok][:, None]; w = (Rk - Lk) / 2
        inside = (cols[None] >= Lk) & (cols[None] <= Rk)
        lb = (cols[None] >= Lk - w) & (cols[None] < Lk); rb = (cols[None] > Rk) & (cols[None] <= Rk + w)
        f[t, 0] = float(((vv & inside).sum(1) / inside.sum(1).clip(1)).mean())
        f[t, 1] = float(((vv & lb).sum(1) / lb.sum(1).clip(1)).mean()); f[t, 2] = float(((vv & rb).sum(1) / rb.sum(1).clip(1)).mean())
        f[t, 3] = float((dd[ok] * (vv & inside)).max()) if (vv & inside).any() else 0.0
        f[t, 7] = float((contact[t, y0:][ok] & inside).mean())
    return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    model, _ = load_run(RUN)
    rows = [r for r in C.rows("all") if not (OUT / f"{r['sample_id']}.geo.npz").exists()]
    for n, r in enumerate(rows, 1):
        frames_all, _ = C.dense(r["sample_id"]); frames = frames_all[C.select_adaptive(frames_all)]
        paths = dict(indexed_frames(r["frames_dir"]))
        dl = torch.utils.data.DataLoader(Frames([(r["sample_id"], int(f), paths[int(f)]) for f in frames]), batch_size=a.batch, num_workers=a.workers)
        g, parts = [], {k: [] for k in ("ego", "veh", "contact", "depth", "alt")}
        for x, _ in dl:
            m = maps(model, x); g.append(grid_feats(m))
            for k in parts: parts[k].append(m[k].cpu().numpy())
        parts = {k: np.concatenate(v) for k, v in parts.items()}
        feats = np.concatenate([np.concatenate(g), corridor_feats(parts["ego"], parts["veh"], parts["contact"], parts["depth"], parts["alt"])], 1)
        np.savez_compressed(OUT / f"{r['sample_id']}.geo.npz", frames=frames, feats=feats.astype(np.float16))
        if n % 25 == 0: print(n, r["sample_id"], flush=True)
    print("done", len(rows), flush=True)


if __name__ == "__main__": main()
