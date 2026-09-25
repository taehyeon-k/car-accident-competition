"""Dense native-frame cache: geometry-DINOv3-S 7x10 pooled tokens for EVERY frame.

Same preprocessing/backbone/pooling as spotting_experiments.extract_features (letterbox 448x800,
bf16 autocast, adaptive avg-pool to 7x10). No FPS/duration is read. Output per clip:
<out>/<sample_id>.npy  float16 [num_frames, 70, 384]  and  <sample_id>.frames.npy int64 frame numbers.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone
from stage2.spotting_experiments.extract_features import MEAN, STD, Images, indexed_frames, read_jsonl


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/workspace/data/stage2/manifests/all.jsonl")
    p.add_argument("--backbone-checkpoint", default="/workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor/backbone_best.pth")
    p.add_argument("--output", default="stage2/long_context_v2_experiments/cache_dense")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = DinoBackbone("vits16", args.backbone_checkpoint).to(device).eval()
    rows = sorted(read_jsonl(args.manifest), key=lambda r: -int(r["num_frames"]))
    for k, row in enumerate(rows):
        target = out / f"{row['sample_id']}.npy"
        if target.exists(): continue
        pairs = indexed_frames(row["frames_dir"])
        feats = []
        for images in DataLoader(Images([x[1] for x in pairs]), batch_size=args.batch_size,
                                 num_workers=args.workers, pin_memory=True):
            images = (images.to(device, non_blocking=True).float() - MEAN.to(device)) / STD.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                patch = model(images)["patch"]
            feats.append(F.adaptive_avg_pool2d(patch.float(), (7, 10)).flatten(2).transpose(1, 2).half().cpu())
        np.save(out / f"{row['sample_id']}.frames.npy", np.asarray([x[0] for x in pairs], dtype=np.int64))
        tmp = out / f"{row['sample_id']}.tmp.npy"
        np.save(tmp, torch.cat(feats).numpy()); os.replace(tmp, target)
        print(f"{k+1}/{len(rows)} {row['sample_id']} {len(pairs)}", flush=True)
    (out / "meta.json").write_text(json.dumps({"mode": "dense_native", "fps_accessed": False, "grid": [7, 10], "dim": 384,
        "checkpoint": args.backbone_checkpoint}, indent=2) + "\n")


if __name__ == "__main__": main()
