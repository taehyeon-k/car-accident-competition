"""Expand the existing 192-grid cache to a nested 383-grid, encoding missing frames only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone
from stage2.spotting_experiments.extract_features import Images, MEAN, STD, indexed_frames
from stage2.spotting_experiments.sampling import normalized_indices
from stage2.spotting_experiments.data import read_rows


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", default="/workspace/data/stage2/manifests/all.jsonl")
    p.add_argument("--batch-size", type=int, default=12)
    args = p.parse_args()
    old, out = Path(args.old), Path(args.output)
    meta = json.loads((old / "meta.json").read_text())
    if meta.get("mode") != "fps_blind" or meta.get("base_count") != 192 or meta.get("fps_accessed"):
        raise ValueError("Expected 192-position FPS-blind cache")
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free < 3 * 2**30: raise RuntimeError("Less than 3 GiB free; will not compete with active jobs")
        torch.cuda.set_per_process_memory_fraction(.05)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoBackbone("vits16", meta["checkpoint"]).to(device).eval()
    out.mkdir(parents=True, exist_ok=True)
    added = 0
    for row in tqdm(read_rows(args.manifest), desc="expand 192→383"):
        target = out / f"{row['sample_id']}.npz"
        if target.exists(): continue
        pairs = indexed_frames(row["frames_dir"])
        positions = normalized_indices([x[0] for x in pairs], 383)
        frames = np.asarray([pairs[i][0] for i in positions], dtype=np.int64)
        with np.load(old / f"{row['sample_id']}.npz") as previous:
            old_frames, old_features = previous["frame_numbers"], previous["features"]
        feature_by_frame = {int(f): old_features[j] for j, f in enumerate(old_frames)}
        missing_positions = sorted({int(i) for i in positions if pairs[i][0] not in feature_by_frame})
        if missing_positions:
            paths = [pairs[i][1] for i in missing_positions]
            encoded = []
            for images in DataLoader(Images(paths), batch_size=args.batch_size, num_workers=2,
                                     pin_memory=device.type == "cuda"):
                images = (images.to(device, non_blocking=True).float() - MEAN.to(device)) / STD.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    patch = model(images)["patch"]
                encoded.append(F.adaptive_avg_pool2d(patch.float(), (7, 10)).flatten(2).transpose(1, 2).half().cpu().numpy())
            new = np.concatenate(encoded)
            feature_by_frame.update({pairs[i][0]: x for i, x in zip(missing_positions, new)})
            added += len(missing_positions)
        features = np.stack([feature_by_frame[int(f)] for f in frames])
        temp = target.with_name(target.name + ".tmp.npz")
        np.savez(temp, features=features, frame_numbers=frames)
        os.replace(temp, target)
    meta.update(base_count=383, derived_from=str(old.resolve()), newly_encoded_unique_frames=added,
                peak_vram_mb=torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__": main()
