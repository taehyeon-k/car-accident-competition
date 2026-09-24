"""Benchmark raw-frame frozen-DINO inference on one long validation clip."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone
from stage2.spotting_experiments.data import read_rows
from stage2.spotting_experiments.extract_features import Images, MEAN, STD, indexed_frames
from stage2.spotting_experiments.sampling import normalized_indices


@torch.inference_mode()
def main():
    if not torch.cuda.is_available(): raise RuntimeError("GPU required for raw-frame benchmark")
    free, _ = torch.cuda.mem_get_info()
    if free < 3 * 2**30: raise RuntimeError("Insufficient spare GPU memory")
    torch.cuda.set_per_process_memory_fraction(.05)
    torch.set_num_threads(4)
    rows = read_rows("/workspace/data/stage2/manifests/val.jsonl")
    row = next(x for x in rows if x["source_id"].startswith("NEXAR:") and x["num_frames"] > 1000)
    meta = json.loads(Path("stage2/temporal_pyramid_experiments/cache_383/meta.json").read_text())
    model = DinoBackbone("vits16", meta["checkpoint"]).cuda().eval()
    pairs = indexed_frames(row["frames_dir"])
    result = {"sample_id": row["sample_id"], "num_frames": len(pairs), "batch_size": 12, "runs": {}}
    for count in (128, 320):
        indices = normalized_indices([x[0] for x in pairs], count)
        paths = [pairs[i][1] for i in sorted(set(indices))]
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for images in DataLoader(Images(paths), batch_size=12, num_workers=2, pin_memory=True):
            images = (images.cuda(non_blocking=True).float() - MEAN.cuda()) / STD.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                patch = model(images)["patch"]
            _ = F.adaptive_avg_pool2d(patch.float(), (7, 10))
        torch.cuda.synchronize()
        result["runs"][str(count)] = {"unique_encoded_frames": len(paths),
                                      "end_to_end_seconds": time.perf_counter() - start,
                                      "peak_allocated_vram_mb": torch.cuda.max_memory_allocated() / 2**20}
        print(count, result["runs"][str(count)], flush=True)
    out = Path("stage2/temporal_pyramid_experiments/results/raw_benchmark.json")
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__": main()
