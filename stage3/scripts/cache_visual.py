"""Cache frozen DINOv3-S patch tokens aligned to existing motion-cache frames (Stage 3 v2, Exp. D).

Frames are decoded with the same PTS->10 Hz selection as ``cache_motion``; the
selected ``actual_times`` must equal the motion cache's. Each frame (and its
horizontal mirror, for flip augmentation) is resized to 208x336, encoded by the
frozen backbone, and the 13x21 final-norm patch grid is average-pooled to
``grid`` tokens. Output per clip: ``<clip_id>.npz`` with float16
``tokens``/``tokens_flip`` of shape [T, gh*gw, 384].
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone
from stage3.data.timing import decode_external_training_video
from stage3.utils.checkpoint import load_artifact
from stage3.utils.config import read_jsonl

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def decode(row):
    meta = row["metadata"]
    video = decode_external_training_video(row["video_path"], 10.0, 0.04, 0.2,
                                           start_time=meta.get("segment_start_time"), end_time=meta.get("segment_end_time"))
    cache = load_artifact(row["cache_path"])
    if len(video.actual_times) != len(cache["actual_times"]) or not np.allclose(video.actual_times, cache["actual_times"].numpy(), atol=1e-6):
        raise ValueError(f"Frame selection differs from motion cache: {row['clip_id']}")
    return row, np.stack(video.frames)


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default="/workspace/pretrained/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-timm-converted.pth")
    p.add_argument("--size", default="208x336")
    p.add_argument("--grid", default="4x7")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--decoders", type=int, default=12)
    args = p.parse_args()
    h, w = map(int, args.size.split("x")); gh, gw = map(int, args.grid.split("x"))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model = DinoBackbone("vits16", args.checkpoint).cuda().eval()
    rows = [r for r in read_jsonl(args.manifest) if not (out / f"{r['clip_id']}.npz").exists()]
    mean, std = MEAN.cuda(), STD.cuda()

    def encode(frames: np.ndarray) -> np.ndarray:
        result = []
        for start in range(0, len(frames), args.batch):
            x = torch.from_numpy(frames[start:start + args.batch]).cuda().permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, (h, w), mode="bilinear", antialias=True, align_corners=False)
            x = (x - mean) / std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                patch = model(x)["patch"]
            result.append(F.adaptive_avg_pool2d(patch.float(), (gh, gw)).flatten(2).transpose(1, 2).half().cpu())
        return torch.cat(result).numpy()

    def bounded(pool):
        # Keep at most 2*decoders decoded clips in memory.
        chunk = 2 * args.decoders
        for start in range(0, len(rows), chunk):
            yield from pool.map(decode, rows[start:start + chunk])

    with ThreadPoolExecutor(args.decoders) as pool:
        for i, (row, frames) in enumerate(bounded(pool)):
            tokens = encode(frames)
            flipped = encode(np.ascontiguousarray(frames[:, :, ::-1]))
            target = out / f"{row['clip_id']}.npz"
            tmp = target.with_name(target.name + ".tmp.npz")
            np.savez(tmp, tokens=tokens, tokens_flip=flipped)
            os.replace(tmp, target)
            if i % 100 == 0:
                print(f"{i}/{len(rows)} {row['clip_id']} {tokens.shape}", flush=True)
    (out / "meta.json").write_text(json.dumps({"checkpoint": args.checkpoint, "size": [h, w], "grid": [gh, gw],
                                               "dim": 384, "frozen": True}, indent=2))


if __name__ == "__main__":
    main()
