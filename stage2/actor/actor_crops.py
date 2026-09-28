"""Stage2_experiments D1/D5 — actor crops on every native frame where the actor is present (detected or interpolated).

For the union of actor boxes over all cache_actor_rank variants of a clip: a TIGHT crop (box enlarged 1.25x, square, 224 x 224; the
obj_crops recipe, so the existing objapp PCA applies) and a CONTEXT crop (box enlarged 2x, square, 224 x 224: actor + surrounding
lane/road) from the ORIGINAL-resolution frame, encoded with the Stage 2 geometry-adapted DINOv3 ViT-S (mean of patch tokens, 384-d).
Output cache_actorcrop/<sid>.npz: tight, ctx float16 [U, 384] (U unique boxes) and idx_<variant> int32 [n] (-1 = no actor).
No labels, no FPS. Usage: python -m stage2.actor.actor_crops [--workers 3]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone
from stage2.objtrack.obj_crops import CKPT, MEAN, STD, S

A = C.REPO / "stage2/actor/cache_actor_rank"; OUT = C.REPO / "stage2/actor/cache_actorcrop"
SCALES = (1.25, 2.0)


def crop(im, box, scale):
    h, w = im.shape[:2]; x1, y1, x2, y2 = box * [w, h, w, h]; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(max(x2 - x1, y2 - y1) * scale, 16)
    a, b = int(round(cx - side / 2)), int(round(cy - side / 2)); c_, d_ = a + int(round(side)), b + int(round(side))
    pad = cv2.copyMakeBorder(im, max(0, -b), max(0, d_ - h), max(0, -a), max(0, c_ - w), cv2.BORDER_CONSTANT, value=(104, 116, 124))
    cr = pad[b + max(0, -b):d_ + max(0, -b), a + max(0, -a):c_ + max(0, -a)]
    return cv2.resize(cr, (S, S), interpolation=cv2.INTER_AREA)[:, :, ::-1]


class Crops(torch.utils.data.Dataset):
    def __init__(self, frames_dir, jobs): self.d, self.jobs = frames_dir, jobs  # jobs: [(frame, [boxes])]
    def __len__(self): return len(self.jobs)

    def __getitem__(self, i):
        f, boxes = self.jobs[i]; im = cv2.imread(str(Path(self.d) / f"{f:06d}.jpg"))
        return torch.from_numpy(np.stack([crop(im, b, s) for b in boxes for s in SCALES])).permute(0, 3, 1, 2)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    dev = torch.device("cuda"); OUT.mkdir(exist_ok=True); variants = sorted(p.name for p in A.iterdir() if p.is_dir())
    model = DinoBackbone("vits16", CKPT).to(dev).eval()
    rows = sorted(C.rows("all"), key=lambda r: -int(r["num_frames"]))
    for n_, r in enumerate(rows, 1):
        sid = r["sample_id"]; target = OUT / f"{sid}.npz"
        if target.exists(): continue
        acts = {v: np.load(A / v / f"{sid}.npz") for v in variants if (A / v / f"{sid}.npz").exists()}
        n = len(next(iter(acts.values()))["present"]); uniq = {}; idx = {v: np.full(n, -1, np.int32) for v in acts}
        per_frame = [[] for _ in range(n)]
        for v, d in acts.items():
            for f in np.flatnonzero(d["present"] > 0):
                key = (int(f), tuple(np.round(d["box"][f], 4).tolist()))
                if key not in uniq: uniq[key] = len(uniq); per_frame[f].append(np.array(key[1], np.float32))
                idx[v][f] = uniq[key]
        jobs = [(f, bs) for f, bs in enumerate(per_frame) if bs]
        tight = np.zeros((len(uniq), 384), np.float16); ctx = np.zeros((len(uniq), 384), np.float16)
        order = [uniq[(f, tuple(np.round(b, 4).tolist()))] for f, bs in jobs for b in bs]; k = 0
        loader = torch.utils.data.DataLoader(Crops(r["frames_dir"], jobs), batch_size=16, num_workers=a.workers, collate_fn=lambda x: torch.cat(x))
        for x in loader:
            x = (x.to(dev).float() - MEAN.to(dev)) / STD.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = model(x)["patch"].float().flatten(2).mean(-1).cpu().numpy()   # [2m, 384]: (tight, ctx) per box
            m = len(e) // 2; ids = order[k:k + m]; tight[ids] = e[0::2]; ctx[ids] = e[1::2]; k += m
        assert k == len(order)
        np.savez(OUT / f"{sid}.tmp.npz", tight=tight, ctx=ctx, **{f"idx_{v}": i for v, i in idx.items()}); (OUT / f"{sid}.tmp.npz").rename(target)
        if n_ % 25 == 0: print(n_, len(rows), sid, flush=True)
    print("done", flush=True)


if __name__ == "__main__": main()
