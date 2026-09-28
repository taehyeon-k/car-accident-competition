"""Stage2_experiments — approximate inference cost of the object / actor front-end on this GPU (per-component throughput on a few
clips; JPEG decoding included). Components: RF-DETR Small + ByteTrack on every native frame (detector venv; run with
/workspace/venvs/det/bin/python -m stage2.actor.timing det), ego-lane road head on every native frame, DINOv3-S crop encoding (224 x 224).
The per-video cost is then (frames x per-frame cost) + (crops x per-crop cost); reported per 150-frame clip (the median clip).
Usage: python -m stage2.actor.timing {det|lane|crop}
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import torch

CLIPS = ["aihub_bb_1_000606_vehicle_142_131", "nexar_00136", "mmau_008965"]
FR = Path("/workspace/data/stage2/frames")


def frames(sid):
    import csv
    for r in csv.DictReader(open("/workspace/data/stage2/usable/usable_only.csv")):
        if r["sample_id"] == sid: return sorted(Path(FR / sid).glob("*.jpg"))
    return sorted(Path(FR / sid).glob("*.jpg"))


def det():
    from stage2.objtrack.extract_tracks import run_clip, WEIGHTS
    from rfdetr import RFDETRSmall
    import tempfile
    m = RFDETRSmall(pretrain_weights=WEIGHTS); out = Path(tempfile.mkdtemp()); n = 0
    run_clip(m, CLIPS[0], FR / CLIPS[0], out)  # warm-up
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    for s in CLIPS[1:]: run_clip(m, s, FR / s, out); n += len(list((FR / s).glob("*.jpg")))
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    print(f"RF-DETR Small + ByteTrack: {n / dt:.1f} frames/s ({1000 * dt / n:.1f} ms/frame), peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


@torch.no_grad()
def lane():
    from stage2.objtrack.ego_corridor import Frames
    from stage2.geometry_pretrain.visualize import load_run
    from stage2.aux_signal_experiments.pseudo_lane import heads, RUN
    model, _ = load_run(RUN); n = 0
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    for s in CLIPS:
        for x, m, cs in torch.utils.data.DataLoader(Frames(sorted((FR / s).glob("*.jpg"))), batch_size=48, num_workers=3):
            heads(model, x); n += len(x)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    print(f"ego-lane road head: {n / dt:.1f} frames/s ({1000 * dt / n:.1f} ms/frame), peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


@torch.inference_mode()
def crop():
    from stage2.actor.actor_crops import Crops
    from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone
    from stage2.objtrack.obj_crops import CKPT, MEAN, STD
    dev = torch.device("cuda"); model = DinoBackbone("vits16", CKPT).to(dev).eval(); n = 0
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    for s in CLIPS:
        fs = sorted((FR / s).glob("*.jpg")); jobs = [(i, [np.array([.4, .5, .5, .6], np.float32)]) for i in range(len(fs))]  # 2 crops / frame
        for x in torch.utils.data.DataLoader(Crops(FR / s, jobs), batch_size=16, num_workers=3, collate_fn=lambda b: torch.cat(b)):
            x = (x.to(dev).float() - MEAN.to(dev)) / STD.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): model(x)["patch"]
            n += len(x)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    print(f"DINOv3-S crop encoder (incl. JPEG decode, 2 crops/frame): {n / dt:.1f} crops/s ({1000 * dt / n:.2f} ms/crop), peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__": {"det": det, "lane": lane, "crop": crop}[sys.argv[1]]()
