"""Unlabelled multi-domain pool for semi-supervised consistency (AIHUB / CCD / MMAU dashcam clips without Stage-2 labels).

Per video (decoded directly, same code paths as aux_signal_experiments/nexar_expand.py, which is parity-checked):
  <id>.npy          geometry-DINOv3-S 7x10 pooled tokens for EVERY native frame (so any frame-rate view can be built)
  <id>.motion.npy   14-d global motion per native frame;  <id>.residual.npy  36-d residual motion per native frame
  <id>.k2.npz / <id>.k3.npz  kept frame indices + global/residual motion recomputed between consecutive frames of frames[::k]
Sources: R2 stage2/unusable/unusable_only.csv (AIHUB, CCD) and stage2/external/mmau/videos (MMAU, minus the labelled ones).
Output cache_unl/. Frame numbers / counts only; no FPS.
"""
from __future__ import annotations

import argparse, csv, io, os, subprocess
from pathlib import Path

import cv2
import numpy as np

from stage2.long_context_v2_experiments import common as C
from stage2.aux_signal_experiments.nexar_expand import letterbox_bgr, W, H

OUT = C.REPO / "stage2/generalization/cache_unl"
VID = Path("/workspace/data/stage2/unlabelled/videos")
R2 = "r2:car-accident-dataset/stage2"


def listing(limit_per_source):
    rows = list(csv.DictReader(io.StringIO(subprocess.run(["rclone", "cat", f"{R2}/unusable/unusable_only.csv"], capture_output=True, text=True).stdout)))
    out = []
    for src in ("AIHUB", "CCD"):
        rs = [r for r in rows if r["source"] == src][:limit_per_source]
        out += [(r["sample_id"], f"{R2}/{r['video_relative_path']}") for r in rs]
    labelled = {r["sample_id"] for r in C.rows("all")}
    mm = subprocess.run(["rclone", "lsf", f"{R2}/external/mmau/videos/"], capture_output=True, text=True).stdout.split()
    mm = [f for f in mm if f.endswith(".mp4") and f"mmau_{Path(f).stem}" not in labelled and Path(f).stem not in labelled][:limit_per_source]
    out += [(f"mmauu_{Path(f).stem}", f"{R2}/external/mmau/videos/{f}") for f in mm]
    return out


def process(sid, key, model, device):
    import torch, torch.nn.functional as F
    from stage2.spotting_experiments.extract_features import MEAN, STD
    from stage2.aux_signal_experiments.extract_residual import camera, sq, ssq, DIM as RDIM
    if (OUT / f"{sid}.npy").exists(): return
    path = VID / Path(key).name
    if not path.exists(): subprocess.run(["rclone", "copy", key, str(VID)], check=True)
    cap = cv2.VideoCapture(str(path)); win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    mot, res, feats, batch = [], [], [], []; prev = prev_r = None
    def pair(p_, pr_, g, gr_):
        m14 = np.zeros(14, np.float32); r36 = np.zeros(RDIM, np.float32)
        (dx, dy), resp = cv2.phaseCorrelate(p_, g, win)
        al = cv2.warpAffine(g, np.float32([[1, 0, -dx], [0, 1, -dy]]), (W, H), borderMode=cv2.BORDER_REFLECT); d = np.abs(al - p_)[3:-3, 3:-3]
        grid = [d[i * d.shape[0] // 3:(i + 1) * d.shape[0] // 3, j * d.shape[1] // 3:(j + 1) * d.shape[1] // 3].mean() for i in range(3) for j in range(3)]
        m14[:] = [dx, dy, np.hypot(dx, dy), resp, d.mean(), *grid]
        m, inl, of, orr = camera(pr_, gr_, win)
        a2 = cv2.warpAffine(gr_, m, (W, H), borderMode=cv2.BORDER_REFLECT); r = np.abs(a2 - pr_)[4:-4, 4:-4]; h, w = r.shape
        gr = [r[i * h // 5:(i + 1) * h // 5, j * w // 5:(j + 1) * w // 5].mean() for i in range(5) for j in range(5)]
        sc = float(np.hypot(m[0, 0], m[1, 0])); rot = float(np.arctan2(m[1, 0], m[0, 0]))
        r36[0:2] = sq(np.array([r.mean(), np.percentile(r, 95)])); r36[2:27] = sq(np.array(gr))
        r36[27] = ssq(r[:, :w // 2].mean() - r[:, w // 2:].mean()); r36[28] = sq(r[h // 2:, w // 3:2 * w // 3].mean())
        r36[29:33] = ssq(np.array([sc - 1, rot, m[0, 2], m[1, 2]])); r36[33:36] = [inl, of, np.log1p(orr)]
        return m14, r36
    strided = {k: {"kept": [], "mot": [], "res": [], "prev": None, "prev_r": None} for k in (2, 3)}; t = -1
    def flush():
        if not batch: return
        with torch.inference_mode():
            x = (torch.stack(batch).to(device).float() - MEAN.to(device)) / STD.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                patch = model(x)["patch"]
            feats.append(F.adaptive_avg_pool2d(patch.float(), (7, 10)).flatten(2).transpose(1, 2).half().cpu())
        batch.clear()
    while True:
        ok, fr = cap.read()
        if not ok: break
        g0 = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        t += 1; g, gr_ = g0.copy(), g0.copy()
        m14, r36 = pair(prev, prev_r, g, gr_) if prev is not None else (np.zeros(14, np.float32), np.zeros(RDIM, np.float32))
        prev, prev_r = g, gr_; mot.append(m14); res.append(r36); batch.append(letterbox_bgr(fr))
        for k, st in strided.items():  # frame-rate views: consecutive retained frames of frames[::k], own chains
            if t % k: continue
            gk, grk = g0.copy(), g0.copy()
            mk, rk = pair(st["prev"], st["prev_r"], gk, grk) if st["prev"] is not None else (np.zeros(14, np.float32), np.zeros(RDIM, np.float32))
            st["prev"], st["prev_r"] = gk, grk; st["kept"].append(t); st["mot"].append(mk); st["res"].append(rk)
        if len(batch) == 48: flush()
    flush(); cap.release()
    if len(mot) < 8: return
    np.save(OUT / f"{sid}.motion.npy", np.stack(mot)); np.save(OUT / f"{sid}.residual.npy", np.stack(res))
    for k, st in strided.items():
        np.savez(OUT / f"{sid}.k{k}.npz", kept=np.array(st["kept"], np.int64), motion=np.stack(st["mot"]), residual=np.stack(st["res"]))
    tmp = OUT / f"{sid}.tmp.npy"; np.save(tmp, torch.cat(feats).numpy()); os.replace(tmp, OUT / f"{sid}.npy")


def worker(jobs):
    import torch
    from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone
    torch.set_num_threads(2); dev = torch.device("cuda")
    model = DinoBackbone("vits16", "/workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor/backbone_best.pth").to(dev).eval()
    for k, (sid, key) in enumerate(jobs):
        try: process(sid, key, model, dev)
        except Exception as e: print("FAIL", sid, repr(e), flush=True)
        if k % 50 == 0: print(os.getpid(), k, len(jobs), sid, flush=True)


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--per-source", type=int, default=400)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); VID.mkdir(parents=True, exist_ok=True)
    jobs = listing(a.per_source); print(len(jobs), "unlabelled videos", flush=True)
    ctx = mp.get_context("spawn"); ps = [ctx.Process(target=worker, args=(jobs[i::a.workers],)) for i in range(a.workers)]
    for p in ps: p.start()
    for p in ps: p.join()
    print("done", flush=True)


if __name__ == "__main__": main()
