"""Offline victim / ego-lane intrusion pseudo-labels from the geometry-DINOv3-S heads (road / objects / contact).

Only for training targets (Exp 2); nothing here runs at Stage-2 inference. Frames = each clip's adaptive sampled frames
(the positions the temporal model sees). Per frame at the heads' 112x200 label resolution:
  ego       = drivable argmax == ego_direct
  contact   = contact logit > 0 inside the (dilated) vehicle mask (car/truck/bus/two-wheeler), near field only
              (lower NEAR_FRAC of the image, where a cutting-in vehicle is close to the ego car)
  corridor  = per image row, [leftmost, rightmost] ego pixel; median-smoothed over +-4 sampled frames (robust to a vehicle
              occluding the lane), rows with too little ego lane are undefined
  d         = min over contact pixels of |x - corridor centre| / corridor half-width
  state     = INSIDE (3) d < 0.5 | CROSSING (2) 0.5 <= d < 1.0 | APPROACHING (1) 1.0 <= d < 1.75 | OUTSIDE (0) otherwise / no contact
  score     = clip((1.75 - d) / 1.75, 0, 1)            (-1 / state -1 where the corridor is undefined)
Output cache_lane/<sample_id>.lane.npz: frames, state, score, dmin, corridor_rows, n_contact, valid_clip.

--variant occupancy (v2, after the contact variant failed QA): near field = lower 60 %; score = mean over valid near-field rows
of the fraction of the smoothed ego corridor covered by vehicle masks; approach band = the corridor widened by 50 % on each side.
  INSIDE occ >= 0.35 | CROSSING 0.05 <= occ < 0.35 | APPROACHING occ < 0.05 and band occupancy >= 0.05 | OUTSIDE otherwise.
Output cache_lane_v2/.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import median_filter

from stage2.geometry_pretrain.common import INPUT_HW, crop_to_aspect
from stage2.geometry_pretrain.pseudo_labels.teachers import MEAN, STD
from stage2.geometry_pretrain.visualize import load_run
from stage2.long_context_v2_experiments import common as C
from stage2.spotting_experiments.extract_features import indexed_frames

RUN = "/workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor"
OUT = C.REPO / "stage2/aux_signal_experiments/cache_lane"
NEAR_FRAC = 0.45
LH, LW = INPUT_HW[0] // 4, INPUT_HW[1] // 4


class Frames(torch.utils.data.Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        sid, fnum, path = self.items[i]
        img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)[..., ::-1]
        img = cv2.resize(crop_to_aspect(img)[0], (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1), i


@torch.no_grad()
def heads(model, x, with_vehicles=False):
    x = (x.cuda().float() / 255.0 - MEAN.cuda()) / STD.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = model.forward_static(x, tasks=["road", "objects", "contact"])
    ego = p["road"][:, :3].float().argmax(1) == 1
    obj = p["objects"].float().argmax(1); veh = (obj >= 1) & (obj <= 4)
    veh = torch.nn.functional.max_pool2d(veh[:, None].float(), 5, 1, 2)[:, 0] > 0
    contact = (p["contact"][:, 0].float() > 0) & veh
    if with_vehicles: return ego.cpu().numpy(), veh.cpu().numpy()
    return ego.cpu().numpy(), contact.cpu().numpy()


def corridor(ego):
    """per-row (left, right) of the ego lane, nan where fewer than 3 ego pixels."""
    T, Hh, Ww = ego.shape; cols = np.arange(Ww)[None, None]
    cnt = ego.sum(-1)
    left = np.where(ego, cols, Ww).min(-1).astype(float); right = np.where(ego, cols, -1).max(-1).astype(float)
    left[cnt < 3] = np.nan; right[cnt < 3] = np.nan
    return left, right


def label_clip(ego, contact):
    T, Hh, Ww = ego.shape; y0 = int(Hh * (1 - NEAR_FRAC))
    left, right = corridor(ego)
    # temporal median smoothing (nan-aware via fill with the clip median of that row)
    for arr in (left, right):
        med = np.nanmedian(arr, 0) if np.isfinite(arr).any() else np.full(Hh, np.nan)
        filled = np.where(np.isfinite(arr), arr, med[None])
        arr[:] = np.where(np.isfinite(filled), median_filter(np.nan_to_num(filled, nan=-1), size=(9, 1), mode="nearest"), np.nan)
        arr[arr < 0] = np.nan
    state = np.full(T, -1, np.int8); score = np.full(T, -1.0, np.float32); dmin = np.full(T, np.inf, np.float32)
    rows_ok = np.zeros(T, np.float32); ncont = np.zeros(T, np.int32)
    for t in range(T):
        L, R = left[t, y0:], right[t, y0:]
        ok = np.isfinite(L) & np.isfinite(R) & (R - L >= 4)
        rows_ok[t] = ok.mean()
        if rows_ok[t] < 0.3: continue
        ys, xs = np.nonzero(contact[t, y0:]); keep = ok[ys]; ys, xs = ys[keep], xs[keep]; ncont[t] = len(xs)
        if len(xs):
            c = (L[ys] + R[ys]) / 2; w = (R[ys] - L[ys]) / 2
            dmin[t] = float(np.min(np.abs(xs - c) / w))
        d = dmin[t]
        state[t] = 3 if d < 0.5 else 2 if d < 1.0 else 1 if d < 1.75 else 0
        score[t] = float(np.clip((1.75 - d) / 1.75, 0, 1)) if np.isfinite(d) else 0.0
    return state, score, dmin, rows_ok, ncont


def label_occupancy(ego, veh, near=0.60):
    T, Hh, Ww = ego.shape; y0 = int(Hh * (1 - near))
    left, right = corridor(ego)
    for arr in (left, right):
        med = np.nanmedian(arr, 0) if np.isfinite(arr).any() else np.full(Hh, np.nan)
        filled = np.where(np.isfinite(arr), arr, med[None])
        arr[:] = np.where(np.isfinite(filled), median_filter(np.nan_to_num(filled, nan=-1), size=(9, 1), mode="nearest"), np.nan)
        arr[arr < 0] = np.nan
    cols = np.arange(Ww)
    state = np.full(T, -1, np.int8); score = np.full(T, -1.0, np.float32); band_occ = np.zeros(T, np.float32); rows_ok = np.zeros(T, np.float32)
    for t in range(T):
        L, R = left[t, y0:], right[t, y0:]; ok = np.isfinite(L) & np.isfinite(R) & (R - L >= 4); rows_ok[t] = ok.mean()
        if rows_ok[t] < 0.3: continue
        v = veh[t, y0:][ok]; Lk, Rk = L[ok][:, None], R[ok][:, None]; w = (Rk - Lk) / 2
        inside = (cols[None] >= Lk) & (cols[None] <= Rk)
        band = ((cols[None] >= Lk - w) & (cols[None] < Lk)) | ((cols[None] > Rk) & (cols[None] <= Rk + w))
        occ = float(((v & inside).sum(1) / inside.sum(1).clip(1)).mean())
        bo = float(((v & band).sum(1) / band.sum(1).clip(1)).mean()); band_occ[t] = bo
        score[t] = min(occ / 0.5, 1.0)
        state[t] = 3 if occ >= 0.35 else 2 if occ >= 0.05 else 1 if bo >= 0.05 else 0
    return state, score, band_occ, rows_ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--variant", choices=["contact", "occupancy"], default="contact")
    a = ap.parse_args()
    out_dir = OUT if a.variant == "contact" else OUT.parent / "cache_lane_v2"; out_dir.mkdir(parents=True, exist_ok=True)
    model, _ = load_run(RUN)
    rows = [r for r in C.rows("all") if not (out_dir / f"{r['sample_id']}.lane.npz").exists()]
    if a.limit: rows = rows[:a.limit]
    for n, r in enumerate(rows, 1):
        frames_all, _ = C.dense(r["sample_id"]); frames = frames_all[C.select_adaptive(frames_all)]
        paths = dict(indexed_frames(r["frames_dir"]))
        items = [(r["sample_id"], int(f), paths[int(f)]) for f in frames]
        dl = torch.utils.data.DataLoader(Frames(items), batch_size=a.batch, num_workers=a.workers)
        egos, conts = [], []
        for x, _ in dl:
            e, c = heads(model, x, with_vehicles=a.variant == "occupancy"); egos.append(e); conts.append(c)
        ego, cont = np.concatenate(egos), np.concatenate(conts)
        if a.variant == "occupancy":
            state, score, dmin, rows_ok = label_occupancy(ego, cont); ncont = np.zeros(len(state), np.int32)
        else:
            state, score, dmin, rows_ok, ncont = label_clip(ego, cont)
        valid_clip = bool((state >= 0).mean() >= 0.7)
        np.savez_compressed(out_dir / f"{r['sample_id']}.lane.npz", frames=frames, state=state, score=score, dmin=dmin,
                            corridor_rows=rows_ok, n_contact=ncont, valid_clip=valid_clip,
                            ego=np.packbits(ego, axis=-1), contact=np.packbits(cont, axis=-1))
        if n % 20 == 0: print(n, r["sample_id"], flush=True)
    print("done", len(rows), flush=True)


if __name__ == "__main__": main()
