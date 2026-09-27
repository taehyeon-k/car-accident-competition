"""Frozen DINOv3 ViT-B 7x10 pooled tokens for EVERY native frame of the unlabelled pool (cache_unl), so ViT-B families can use the
metadata-anchored extras and the unlabelled rate-consistency loss. Same decode + letterbox as unl_expand.py / nexar_expand.py
(parity-checked against the JPEG caches), same bf16 autocast and adaptive avg-pool as extract_dense.py. Frame count must equal
the existing <sid>.motion.npy. Output cache_unl_vitb/<sid>.npy float16 [n, 70, 768]. No FPS is read.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from stage2.long_context_v2_experiments import common as C
from stage2.aux_signal_experiments.nexar_expand import letterbox_bgr
from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone
from stage2.spotting_experiments.extract_features import MEAN, STD

UNL = C.REPO / "stage2/generalization/cache_unl"
OUT = C.REPO / "stage2/generalization/cache_unl_vitb"
VID = Path("/workspace/data/stage2/unlabelled/videos")
CKPT = "/workspace/pretrained/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only-labelled-pool", action="store_true"); a = ap.parse_args()
    dev = torch.device("cuda"); OUT.mkdir(exist_ok=True)
    model = DinoBackbone("vitb16", CKPT).to(dev).eval()
    sids = sorted(p.name[:-len(".motion.npy")] for p in UNL.glob("*.motion.npy"))
    for i, sid in enumerate(sids, 1):
        target = OUT / f"{sid}.npy"
        if target.exists(): continue
        path = VID / (sid.split("_", 1)[1] + ".mp4" if sid.startswith("mmauu_") else sid + ".mp4")
        n_expect = int(np.load(UNL / f"{sid}.motion.npy", mmap_mode="r").shape[0])
        cap = cv2.VideoCapture(str(path)); feats, batch = [], []
        def flush():
            if not batch: return
            x = (torch.stack(batch).to(dev).float() - MEAN.to(dev)) / STD.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                patch = model(x)["patch"]
            feats.append(F.adaptive_avg_pool2d(patch.float(), (7, 10)).flatten(2).transpose(1, 2).half().cpu()); batch.clear()
        while True:
            ok, fr = cap.read()
            if not ok: break
            batch.append(letterbox_bgr(fr))
            if len(batch) == 32: flush()
        flush(); cap.release()
        arr = torch.cat(feats).numpy() if feats else np.zeros((0, 70, 768), np.float16)
        assert len(arr) == n_expect, f"{sid}: {len(arr)} frames vs motion {n_expect}"
        np.save(OUT / f"{sid}.tmp.npy", arr); (OUT / f"{sid}.tmp.npy").rename(target)
        if i % 100 == 0: print(i, len(sids), sid, flush=True)
    print("done", flush=True)


if __name__ == "__main__": main()
