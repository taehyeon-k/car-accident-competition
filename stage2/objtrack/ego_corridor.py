"""Ego-lane corridor on EVERY native frame (Experiment 2 of stage2/Stage2_Entry_Robustness_Experiments.md).

Geometry-adapted DINOv3-S road head (same model / preprocessing as aux_signal_experiments/pseudo_lane.py: 2x-reduced JPEG, centre crop
to the 800:448 aspect, 112 x 200 label grid, ego lane = road argmax class 1). Per frame and per label row: leftmost / rightmost ego
pixel (nan with < 3 ego pixels), then a temporal median over +-4 native frames per row (robust to a vehicle occluding the lane).
Coordinates are mapped back to the ORIGINAL frame, normalised to [0, 1] (the same frame the RF-DETR boxes use).
Output cache_corridor/<sid>.npz: left, right float16 [n, 112] (nan = undefined), row_y float32 [112] (normalised y of each row).
No labels, no FPS.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import median_filter

from stage2.geometry_pretrain.common import INPUT_HW, crop_to_aspect
from stage2.geometry_pretrain.visualize import load_run
from stage2.aux_signal_experiments.pseudo_lane import heads, corridor, RUN
from stage2.long_context_v2_experiments import common as C

OUT = C.REPO / "stage2/objtrack/cache_corridor"


class Frames(torch.utils.data.Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = cv2.imread(str(self.paths[i]), cv2.IMREAD_REDUCED_COLOR_2)[..., ::-1]
        crop, box = crop_to_aspect(img)
        img = cv2.resize(crop, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1), torch.tensor([*box, *img.shape[:2]]), torch.tensor(crop.shape[:2])


def smooth(arr, size=9):
    med = np.nanmedian(arr, 0) if np.isfinite(arr).any() else np.full(arr.shape[1], np.nan)
    filled = np.where(np.isfinite(arr), arr, med[None])
    out = np.where(np.isfinite(filled), median_filter(np.nan_to_num(filled, nan=-1), size=(size, 1), mode="nearest"), np.nan)
    out[out < 0] = np.nan
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3); ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args(); OUT.mkdir(exist_ok=True)
    model, _ = load_run(RUN)
    rows = sorted(C.rows("all"), key=lambda r: -int(r["num_frames"]))
    for n_, r in enumerate(rows, 1):
        sid = r["sample_id"]; target = OUT / f"{sid}.npz"
        if target.exists(): continue
        paths = sorted(Path(r["frames_dir"]).glob("*.jpg"))
        egos, meta = [], None
        for x, m, cs in torch.utils.data.DataLoader(Frames(paths), batch_size=a.batch, num_workers=a.workers):
            e, _ = heads(model, x); egos.append(e); meta = (m[0].numpy(), cs[0].numpy())
        ego = np.concatenate(egos); left, right = corridor(ego)   # label-grid columns [n, 112]
        left, right = smooth(left), smooth(right)
        (x0, y0, cw, ch, rh, rw), _ = (meta[0][:4].tolist() + [None, None], None) if False else ((*meta[0][:4].tolist(), 0, 0), None)
        # 2x-reduced image: crop box (x0, y0, w, h) is in reduced pixels; the reduced image is half the original resolution
        img0 = cv2.imread(str(paths[0])); H0, W0 = img0.shape[:2]; Hr, Wr = (H0 + 1) // 2, (W0 + 1) // 2
        lw = INPUT_HW[1] // 4; lh = INPUT_HW[0] // 4
        to_x = lambda c: (x0 + (c + 0.5) / lw * cw) / Wr
        row_y = ((y0 + (np.arange(lh) + 0.5) / lh * ch) / Hr).astype(np.float32)
        np.savez(OUT / f"{sid}.tmp.npz", left=to_x(left).astype(np.float16), right=to_x(right).astype(np.float16), row_y=row_y)
        (OUT / f"{sid}.tmp.npz").rename(target)
        if n_ % 25 == 0: print(n_, len(rows), sid, flush=True)
    print("done", flush=True)


if __name__ == "__main__": main()
