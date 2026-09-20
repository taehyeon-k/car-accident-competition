"""Cache geometry-DINO patch grids without using FPS in strict mode."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone
from .sampling import normalized_indices

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) * 255


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def indexed_frames(folder):
    result = []
    for path in Path(folder).iterdir():
        match = re.search(r"(\d+)$", path.stem)
        if path.is_file() and match and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            result.append((int(match.group(1)), path))
    result.sort()
    if not result or len({x[0] for x in result}) != len(result):
        raise ValueError(f"Missing or duplicate frames: {folder}")
    return result


def letterbox(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Cannot read {path}")
    h, w = bgr.shape[:2]
    scale = min(448 / h, 800 / w)
    nh, nw = round(h * scale), round(w * scale)
    out = np.empty((448, 800, 3), np.uint8)
    out[:] = (124, 116, 104)
    y, x = (448 - nh) // 2, (800 - nw) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(out[:, :, ::-1])).permute(2, 0, 1)


class Images(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return letterbox(self.paths[i])


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/workspace/data/stage2/manifests/all.jsonl")
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["fps_blind", "legacy_10hz"], default="fps_blind")
    p.add_argument("--base-count", type=int, default=192)
    p.add_argument("--grid", default="7x10")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    gh, gw = map(int, args.grid.lower().split("x"))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoBackbone("vits16", args.backbone_checkpoint).to(device).eval()
    peak = 0
    for row in tqdm(read_jsonl(args.manifest), desc=f"extract {args.mode} {args.grid}"):
        target = out / f"{row['sample_id']}.npz"
        if target.exists():
            continue
        pairs = indexed_frames(row["frames_dir"])
        if args.mode == "fps_blind":
            positions = normalized_indices([x[0] for x in pairs], args.base_count)
        else:
            # Deliberately isolated legacy reproduction; strict mode never enters here.
            stride = max(1, round(float(row["native_fps"]) / 10.0))
            positions = np.arange(0, len(pairs), stride, dtype=np.int64)
        chosen = [pairs[i] for i in positions]
        unique, inverse = np.unique(positions, return_inverse=True)
        paths = [pairs[i][1] for i in unique]
        feats = []
        for images in DataLoader(Images(paths), batch_size=args.batch_size, num_workers=args.workers,
                                 pin_memory=device.type == "cuda"):
            images = (images.to(device, non_blocking=True).float() - MEAN.to(device)) / STD.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                patch = model(images)["patch"]
            grid = F.adaptive_avg_pool2d(patch.float(), (gh, gw)).flatten(2).transpose(1, 2)
            feats.append(grid.half().cpu())
            if device.type == "cuda": peak = max(peak, torch.cuda.max_memory_allocated())
        features = torch.cat(feats).numpy()[inverse]
        frames = np.asarray([x[0] for x in chosen], dtype=np.int64)
        tmp = target.with_name(target.name + ".tmp.npz")
        np.savez(tmp, features=features, frame_numbers=frames)
        os.replace(tmp, target)
    meta = {"mode": args.mode, "base_count": args.base_count if args.mode == "fps_blind" else None,
            "grid": [gh, gw], "dim": model.embed_dim, "checkpoint": str(Path(args.backbone_checkpoint).resolve()),
            "fps_accessed": args.mode != "fps_blind", "peak_vram_mb": peak / 2**20}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__": main()
