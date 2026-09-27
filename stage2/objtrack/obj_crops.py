"""Option 3: high-resolution appearance of the most prominent road users (object-centric ENTRY evidence).

Per native frame, the K = 3 most prominent detected vehicles (RF-DETR + ByteTrack, cache_tracks; prominence = box area x centrality,
no labels used) are cropped on EVERY native frame (box enlarged 25 %, square, 224 x 224) and encoded with the Stage 2
geometry-adapted DINOv3 ViT-S (mean of patch tokens, 384-d). Output cache_objcrop/<sid>.npz:
  emb float16 [n, K, 384], present bool [n, K], box float32 [n, K, 4] (normalised x1, y1, x2, y2), slot_cls int16 [K]
No FPS. Usage: python -m stage2.objtrack.obj_crops [--workers 3]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from stage2.long_context_v2_experiments import common as C
from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone

TRACKS = C.REPO / "stage2/objtrack/cache_tracks"
OUT = C.REPO / "stage2/objtrack/cache_objcrop"
CKPT = "/workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor/backbone_best.pth"
K, S, VEH = 3, 224, (2, 3, 4, 6, 8)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def select_tracks(d):
    fr, box, cls, tr = d["frame"], d["box"], d["cls"], d["track"]
    m = np.isin(cls, VEH) & (tr >= 0)
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]); cen = 1 - np.abs((box[:, 0] + box[:, 2]) / 2 - 0.5)
    score = {}
    for t, a, c in zip(tr[m], area[m], cen[m]): score[int(t)] = score.get(int(t), 0.0) + float(a * c)
    return sorted(score, key=lambda t: -score[t])[:K]


def per_frame_slots(d, n):
    """Per native frame, the K vehicles with the largest area x centrality (slots filled whenever any vehicle is detected;
    distant vehicles included). Track ids kept per slot for motion features."""
    fr, box, cls, tr = d["frame"], d["box"], d["cls"], d["track"]
    m = np.isin(cls, VEH); fr, box, tr = fr[m], box[m], tr[m]
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]); cen = 1 - np.abs((box[:, 0] + box[:, 2]) / 2 - 0.5); sc = area * cen
    present = np.zeros((n, K), bool); boxes = np.zeros((n, K, 4), np.float32); tids = np.full((n, K), -1, np.int32)
    order = np.lexsort((-sc, fr)); fr, box, tr = fr[order], box[order], tr[order]
    bounds = np.searchsorted(fr, np.arange(n + 1))
    for t in range(n):
        s_, e_ = bounds[t], min(bounds[t + 1], bounds[t] + K)
        k = e_ - s_; present[t, :k] = True; boxes[t, :k] = box[s_:e_]; tids[t, :k] = tr[s_:e_]
    return present, boxes, tids


class Crops(torch.utils.data.Dataset):
    def __init__(self, paths, boxes, present): self.paths, self.boxes, self.present = paths, boxes, present
    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        out = np.zeros((K, S, S, 3), np.uint8)
        if self.present[i].any():
            im = cv2.imread(str(self.paths[i])); h, w = im.shape[:2]
            for k in range(K):
                if not self.present[i, k]: continue
                x1, y1, x2, y2 = self.boxes[i, k] * [w, h, w, h]; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                side = max(x2 - x1, y2 - y1) * 1.25; side = max(side, 16)
                a, b = int(round(cx - side / 2)), int(round(cy - side / 2)); c_, d_ = a + int(round(side)), b + int(round(side))
                pad = cv2.copyMakeBorder(im, max(0, -b), max(0, d_ - h), max(0, -a), max(0, c_ - w), cv2.BORDER_CONSTANT, value=(104, 116, 124))
                crop = pad[b + max(0, -b):d_ + max(0, -b), a + max(0, -a):c_ + max(0, -a)]
                out[k] = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        return torch.from_numpy(out).permute(0, 3, 1, 2)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    dev = torch.device("cuda"); OUT.mkdir(exist_ok=True)
    model = DinoBackbone("vits16", CKPT).to(dev).eval()
    rows = sorted(C.rows("all"), key=lambda r: -int(r["num_frames"]))
    for n_, r in enumerate(rows, 1):
        sid = r["sample_id"]; target = OUT / f"{sid}.npz"
        if target.exists(): continue
        d = np.load(TRACKS / f"{sid}.npz"); n = int(d["nframes"]); present, boxes, slot_tid = per_frame_slots(d, n); slot_cls = np.zeros(K, np.int16)
        paths = [Path(r["frames_dir"]) / f"{i:06d}.jpg" for i in range(n)]
        emb = np.zeros((n, K, 384), np.float16); i0 = 0
        for batch in torch.utils.data.DataLoader(Crops(paths, boxes, present), batch_size=32, num_workers=a.workers):
            b = batch.shape[0]; x = batch.reshape(b * K, 3, S, S).to(dev).float()
            x = (x - MEAN.to(dev)) / STD.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                patch = model(x)["patch"]
            e = patch.float().flatten(2).mean(-1).reshape(b, K, -1)  # mean of patch tokens [b, K, 384]
            emb[i0:i0 + b] = (e * torch.from_numpy(present[i0:i0 + b]).to(dev)[..., None]).half().cpu().numpy(); i0 += b
        np.savez(OUT / f"{sid}.tmp.npz", emb=emb, present=present, box=boxes, track=slot_tid); (OUT / f"{sid}.tmp.npz").rename(target)
        if n_ % 25 == 0: print(n_, len(rows), sid, flush=True)
    print("done", flush=True)


if __name__ == "__main__": main()
