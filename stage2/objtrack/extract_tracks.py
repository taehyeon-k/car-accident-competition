"""RF-DETR Small (COCO, project checkpoint) detections + ByteTrack identities on EVERY native frame of each clip.

Runs in the detector venv (/workspace/venvs/det: rfdetr + supervision; separate torch). No FPS / timestamps: ByteTrack keeps its
default frame-count buffers. Only road users are kept (person, bicycle, car, motorcycle, bus, truck), score >= 0.3.
Output cache_tracks/<sample_id>.npz:
  frame  int32 [N]   native frame index of each detection
  box    float32 [N,4] x1,y1,x2,y2 normalised by image width / height
  score  float32 [N]
  cls    int16 [N]   COCO id
  track  int32 [N]   ByteTrack id (-1 = unmatched detection)
  nframes, width, height
Usage: /workspace/venvs/det/bin/python -m stage2.objtrack.extract_tracks [--source DIRLIST] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path("/workspace/car-accident")
OUT = REPO / "stage2/objtrack/cache_tracks"
KEEP = (1, 2, 3, 4, 6, 8)
WEIGHTS = "/workspace/pretrained/rfdetr_small/rf-detr-small.pth"


class Frames(torch.utils.data.Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return np.asarray(Image.open(self.paths[i]).convert("RGB"))


def clip_dirs(manifest):
    with open(manifest) as f:
        for r in csv.DictReader(f):
            yield r["sample_id"], Path("/workspace/data/stage2/frames") / r["sample_id"]


def run_clip(model, sid, d, out, bs=16, workers=3):
    import supervision as sv
    target = out / f"{sid}.npz"
    if target.exists(): return
    paths = sorted(d.glob("*.jpg")); assert paths, d
    idx = [int(p.stem) for p in paths]; assert idx == list(range(len(idx))), f"{sid}: frame files not contiguous"
    tracker = sv.ByteTrack()
    loader = torch.utils.data.DataLoader(Frames(paths), batch_size=bs, num_workers=workers, collate_fn=list)
    F, B, S, K, T = [], [], [], [], []; t = 0; w = h = None
    for batch in loader:
        dets = model.predict([Image.fromarray(a) for a in batch], threshold=0.3)
        dets = dets if isinstance(dets, list) else [dets]
        for a, det in zip(batch, dets):
            h, w = a.shape[:2]
            det = det[np.isin(det.class_id, KEEP)]
            det = tracker.update_with_detections(det) if len(det) else tracker.update_with_detections(det)
            n = len(det)
            if n:
                F.append(np.full(n, t, np.int32)); B.append((det.xyxy / [w, h, w, h]).astype(np.float32))
                S.append(det.confidence.astype(np.float32)); K.append(det.class_id.astype(np.int16))
                T.append(det.tracker_id.astype(np.int32) if det.tracker_id is not None else np.full(n, -1, np.int32))
            t += 1
    cat = lambda xs, shape, dt: np.concatenate(xs) if xs else np.zeros(shape, dt)
    tmp = out / f"{sid}.tmp.npz"
    np.savez(tmp, frame=cat(F, (0,), np.int32), box=cat(B, (0, 4), np.float32), score=cat(S, (0,), np.float32),
             cls=cat(K, (0,), np.int16), track=cat(T, (0,), np.int32), nframes=t, width=w, height=h)
    tmp.rename(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/workspace/data/stage2/usable/usable_only.csv")
    ap.add_argument("--out", default=str(OUT)); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    k, n = map(int, a.shard.split("/"))
    from rfdetr import RFDETRSmall
    model = RFDETRSmall(pretrain_weights=WEIGHTS)
    jobs = list(clip_dirs(a.manifest))[k::n]
    for i, (sid, d) in enumerate(jobs, 1):
        run_clip(model, sid, d, out)
        if i % 20 == 0: print(i, len(jobs), sid, flush=True)
    print("done", flush=True)


if __name__ == "__main__": sys.exit(main())
