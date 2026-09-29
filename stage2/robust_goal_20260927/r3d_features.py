"""Frozen Kinetics R3D layer2/layer3 features aligned to Stage2 sampled frames.

Feasibility pilot only. The input is the same adaptively selected frame sequence
used by Stage2; no labels, FPS, timestamps, or source are model inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.video import R3D_18_Weights, r3d_18

from stage2.long_context_v2_experiments import common as C

ROOT = C.REPO / "stage2/robust_goal_20260927/r3d_pilot"
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)


def selected_frames(sample_id, stride=1):
    frame_numbers, _ = C.dense(sample_id)
    if stride > 1:
        path = C.REPO / f"stage2/generalization/cache_stride/k{stride}/{sample_id}.npz"
        with np.load(path) as data:
            frame_numbers = frame_numbers[data["kept"]]
    return frame_numbers[C.select_adaptive(frame_numbers)]


def other_gpu_job():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        return any(int(pid) != os.getpid() for pid in output.split() if pid.isdigit())
    except (OSError, subprocess.CalledProcessError):
        return True


def images(sample_id, frames):
    folder = Path("/workspace/data/stage2/frames") / sample_id
    arr = []
    for frame in frames:
        image = cv2.imread(str(folder / f"{int(frame):06d}.jpg"), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"{folder}/{int(frame):06d}.jpg")
        image = cv2.resize(image, (200, 112), interpolation=cv2.INTER_AREA)
        arr.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).float() / 255


@torch.inference_mode()
def extract(model, video, device):
    n = len(video)
    result = torch.zeros(n, 384, device=device)
    count = torch.zeros(n, 1, device=device)
    starts = list(range(0, max(n - 64, 0) + 1, 48))
    last = max(n - 64, 0)
    if not starts or starts[-1] != last:
        starts.append(last)
    for start in starts:
        end = min(start + 64, n)
        chunk = video[start:end].to(device)
        chunk = (chunk - MEAN.to(device)) / STD.to(device)
        x = chunk.permute(1, 0, 2, 3)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = model.stem(x)
            x = model.layer1(x)
            x = model.layer2(x)
            l2 = x.float().mean((-2, -1))
            l3 = model.layer3(x).float().mean((-2, -1))
        l2 = F.interpolate(l2, size=end-start, mode="linear", align_corners=False)
        l3 = F.interpolate(l3, size=end-start, mode="linear", align_corners=False)
        feat = torch.cat((l2, l3), 1)[0].transpose(0, 1)
        result[start:end] += feat
        count[start:end] += 1
    return (result / count).half().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", nargs="*")
    parser.add_argument("--all", action="store_true", help="extract all 349 labelled clips")
    parser.add_argument("--strides", nargs="+", type=int, default=[1])
    args = parser.parse_args()
    if args.all and args.samples:
        parser.error("Choose explicit samples or --all")
    samples = [r["sample_id"] for r in C.rows("all")] if args.all else args.samples
    if not samples:
        parser.error("No samples selected")
    ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1).to(device).eval()
    report = []
    for stride in args.strides:
        dest = ROOT / "cache" / f"k{stride}"
        dest.mkdir(parents=True, exist_ok=True)
        for index, sample_id in enumerate(samples):
            target = dest / f"{sample_id}.npy"
            if target.exists():
                continue
            while other_gpu_job():
                print("GPU busy; waiting before next R3D clip", flush=True)
                time.sleep(15)
            t0 = time.perf_counter()
            frames = selected_frames(sample_id, stride)
            video = images(sample_id, frames)
            decode_s = time.perf_counter() - t0
            torch.cuda.reset_peak_memory_stats()
            feat = extract(model, video, device)
            torch.cuda.synchronize()
            record = {"sample_id": sample_id, "stride": stride, "frames": len(frames), "shape": list(feat.shape),
                      "decode_seconds": decode_s, "total_seconds": time.perf_counter() - t0,
                      "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1048576,
                      "finite": bool(np.isfinite(feat).all()), "mean_abs": float(np.abs(feat).mean())}
            np.save(target, feat)
            report.append(record)
            if index % 20 == 0 or not record["finite"]:
                print(json.dumps(record), flush=True)
            if not record["finite"]:
                raise RuntimeError(f"Non-finite features for {sample_id} k{stride}")
        (ROOT / f"extraction_k{stride}.json").write_text(json.dumps([r for r in report if r["stride"] == stride], indent=2) + "\n")


if __name__ == "__main__":
    main()
