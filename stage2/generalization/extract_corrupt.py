"""Image-quality robustness inputs: geometry-DINOv3-S 7x10 tokens of each clip's adaptive sampled frames under a corruption.
  lowres : frame downscaled 4x (INTER_AREA) and back up (INTER_LINEAR) before the usual letterbox  -> low-resolution camera
  jpeg   : frame re-encoded as JPEG quality 15 before the letterbox                                  -> heavy compression
Same letterbox 448x800 / bf16 / backbone / pooling as extract_dense. Output cache_corrupt/<kind>/<id>.npy [T,70,384] fp16 and <id>.frames.npy.
Motion inputs are left clean (they come from 160x90 grey frames, already a low-resolution representation). No FPS.
"""
from __future__ import annotations

import argparse, os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone
from stage2.long_context_v2_experiments import common as C
from stage2.spotting_experiments.extract_features import MEAN, STD, indexed_frames
from stage2.aux_signal_experiments.nexar_expand import letterbox_bgr

OUT = C.REPO / "stage2/generalization/cache_corrupt"


def corrupt(bgr, kind):
    if kind == "lowres":
        h, w = bgr.shape[:2]
        return cv2.resize(cv2.resize(bgr, (w // 4, h // 4), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 15]); return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class Frames(torch.utils.data.Dataset):
    def __init__(self, paths, kind): self.paths, self.kind = paths, kind
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return letterbox_bgr(corrupt(cv2.imread(str(self.paths[i])), self.kind))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kinds", nargs="+", default=["lowres", "jpeg"]); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(); dev = torch.device("cuda")
    model = DinoBackbone("vits16", "/workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor/backbone_best.pth").to(dev).eval()
    for kind in a.kinds:
        (OUT / kind).mkdir(parents=True, exist_ok=True)
        for n, r in enumerate(C.rows("all"), 1):
            if (OUT / kind / f"{r['sample_id']}.npy").exists(): continue
            fa, _ = C.dense(r["sample_id"]); fr = fa[C.select_adaptive(fa)]; P = dict(indexed_frames(r["frames_dir"]))
            feats = []
            for x in torch.utils.data.DataLoader(Frames([P[int(f)] for f in fr], kind), batch_size=32, num_workers=a.workers):
                x = (x.to(dev).float() - MEAN.to(dev)) / STD.to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    patch = model(x)["patch"]
                feats.append(F.adaptive_avg_pool2d(patch.float(), (7, 10)).flatten(2).transpose(1, 2).half().cpu())
            np.save(OUT / kind / f"{r['sample_id']}.frames.npy", fr.astype(np.int64))
            tmp = OUT / kind / f"{r['sample_id']}.tmp.npy"; np.save(tmp, torch.cat(feats).numpy()); os.replace(tmp, OUT / kind / f"{r['sample_id']}.npy")
            if n % 50 == 0: print(kind, n, flush=True)
    print("done", flush=True)


if __name__ == "__main__": main()
