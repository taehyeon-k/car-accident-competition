"""Higher-resolution frozen features: geometry-DINOv3-S patch tokens adaptive-avg-pooled to 14x25 (350 tokens) instead of 7x10,
at each clip's adaptive sampled frames (same letterbox 448x800 / bf16 / backbone as extract_dense). Output cache_hr/<id>.npy
float16 [T, 350, 384] and <id>.frames.npy (sampled frame numbers). No FPS.
"""
from __future__ import annotations

import argparse, os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone
from stage2.long_context_v2_experiments import common as C
from stage2.spotting_experiments.extract_features import MEAN, STD, Images, indexed_frames

OUT = C.REPO / "stage2/aux_signal_experiments/cache_hr"


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda")
    model = DinoBackbone("vits16", "/workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor/backbone_best.pth").to(dev).eval()
    for n, r in enumerate(C.rows("all"), 1):
        if (OUT / f"{r['sample_id']}.npy").exists(): continue
        fa, _ = C.dense(r["sample_id"]); fr = fa[C.select_adaptive(fa)]; paths = dict(indexed_frames(r["frames_dir"]))
        feats = []
        for x in DataLoader(Images([paths[int(f)] for f in fr]), batch_size=a.batch, num_workers=a.workers):
            x = (x.to(dev).float() - MEAN.to(dev)) / STD.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                patch = model(x)["patch"]
            feats.append(F.adaptive_avg_pool2d(patch.float(), (14, 25)).flatten(2).transpose(1, 2).half().cpu())
        np.save(OUT / f"{r['sample_id']}.frames.npy", fr.astype(np.int64))
        tmp = OUT / f"{r['sample_id']}.tmp.npy"; np.save(tmp, torch.cat(feats).numpy()); os.replace(tmp, OUT / f"{r['sample_id']}.npy")
        if n % 25 == 0: print(n, r["sample_id"], flush=True)
    print("done", flush=True)


if __name__ == "__main__": main()
