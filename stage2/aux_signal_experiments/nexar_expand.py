"""Build training inputs for the 670 unlabelled NEXAR positives (R2 stage2/videos/nexar/, not in the 349 labelled clips).

Per video (decoded directly, no JPEG frames on disk):
  <id>.nframes.npy  native frame count        <id>.frames.npy  adaptive sampled frame numbers (C.select_adaptive, as for labelled clips)
  <id>.npy          geometry-DINOv3-S 7x10 pooled tokens at the sampled frames  (letterbox 448x800, bf16, same as extract_dense)
  <id>.motion.npy   14-d global motion per native frame   (extract_motion algorithm on 160x90 grey)
  <id>.residual.npy 36-d residual motion per native frame (extract_residual algorithm)
Labels are NOT made here (see nexar_labels.py). Frame numbers / counts only; no FPS enters any feature.
"""
from __future__ import annotations

import argparse, csv, os, subprocess
from pathlib import Path

import cv2
import numpy as np

from stage2.long_context_v2_experiments import common as C

OUT = C.REPO / "stage2/aux_signal_experiments/cache_nexar_u"
VID = Path("/workspace/data/stage2/nexar_unlabeled/videos")
CAND = C.REPO / "stage2/aux_signal_experiments/nexar_candidates.csv"  # copy of R2 stage2/manifests/nexar_candidates.csv
W, H = 160, 90


def letterbox_bgr(bgr):
    import torch
    h, w = bgr.shape[:2]; scale = min(448 / h, 800 / w); nh, nw = round(h * scale), round(w * scale)
    out = np.empty((448, 800, 3), np.uint8); out[:] = (124, 116, 104); y, x = (448 - nh) // 2, (800 - nw) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(out[:, :, ::-1])).permute(2, 0, 1)


def process(vid, model, device):
    import torch, torch.nn.functional as F
    from stage2.spotting_experiments.extract_features import MEAN, STD
    from .extract_residual import camera, sq, ssq, DIM as RDIM
    sid = f"nexaru_{vid}"
    if (OUT / f"{sid}.npy").exists(): return sid
    path = VID / f"{vid}.mp4"
    if not path.exists():
        subprocess.run(["rclone", "copy", f"r2:car-accident-dataset/stage2/videos/nexar/{vid}.mp4", str(VID)], check=True)
    cap = cv2.VideoCapture(str(path)); n = 0   # pass 1: count frames without decoding them into memory
    while cap.grab(): n += 1
    cap.release()
    sampled = C.select_adaptive(np.arange(n)); keep = set(int(i) for i in sampled)
    cap = cv2.VideoCapture(str(path))
    def raw_frames():   # pass 2: stream-decode
        while True:
            ok, fr = cap.read()
            if not ok: return
            yield fr
    cv2.setNumThreads(1); win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    mot = np.zeros((n, 14), np.float32); res = np.zeros((n, RDIM), np.float32); imgs = []
    # NOTE cv2.phaseCorrelate modifies its input arrays in place, so the global-motion and residual-motion computations keep
    # separate frame chains, each reproducing the call pattern of extract_motion / extract_residual exactly.
    prev = prev_r = None
    for t, fr in enumerate(raw_frames()):
        g0 = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        g, gr_ = g0.copy(), g0.copy()
        if t in keep: imgs.append(letterbox_bgr(fr))
        if prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(prev, g, win)
            al = cv2.warpAffine(g, np.float32([[1, 0, -dx], [0, 1, -dy]]), (W, H), borderMode=cv2.BORDER_REFLECT)
            d = np.abs(al - prev)[3:-3, 3:-3]
            grid = [d[i * d.shape[0] // 3:(i + 1) * d.shape[0] // 3, j * d.shape[1] // 3:(j + 1) * d.shape[1] // 3].mean() for i in range(3) for j in range(3)]
            mot[t] = [dx, dy, np.hypot(dx, dy), resp, d.mean(), *grid]
            m, inl, of, orr = camera(prev_r, gr_, win)
            a2 = cv2.warpAffine(gr_, m, (W, H), borderMode=cv2.BORDER_REFLECT); r = np.abs(a2 - prev_r)[4:-4, 4:-4]; h, w = r.shape
            gr = [r[i * h // 5:(i + 1) * h // 5, j * w // 5:(j + 1) * w // 5].mean() for i in range(5) for j in range(5)]
            sc = float(np.hypot(m[0, 0], m[1, 0])); rot = float(np.arctan2(m[1, 0], m[0, 0]))
            res[t, 0:2] = sq(np.array([r.mean(), np.percentile(r, 95)])); res[t, 2:27] = sq(np.array(gr))
            res[t, 27] = ssq(r[:, :w // 2].mean() - r[:, w // 2:].mean()); res[t, 28] = sq(r[h // 2:, w // 3:2 * w // 3].mean())
            res[t, 29:33] = ssq(np.array([sc - 1, rot, m[0, 2], m[1, 2]])); res[t, 33:36] = [inl, of, np.log1p(orr)]
        prev, prev_r = g, gr_
    cap.release()
    feats = []
    with torch.inference_mode():
        for i in range(0, len(imgs), 48):
            x = (torch.stack(imgs[i:i + 48]).to(device).float() - MEAN.to(device)) / STD.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                patch = model(x)["patch"]
            feats.append(F.adaptive_avg_pool2d(patch.float(), (7, 10)).flatten(2).transpose(1, 2).half().cpu())
    np.save(OUT / f"{sid}.frames.npy", sampled.astype(np.int64)); np.save(OUT / f"{sid}.nframes.npy", np.array(n))
    np.save(OUT / f"{sid}.motion.npy", mot); np.save(OUT / f"{sid}.residual.npy", res)
    tmp = OUT / f"{sid}.tmp.npy"; np.save(tmp, torch.cat(feats).numpy()); os.replace(tmp, OUT / f"{sid}.npy")
    return sid


def worker(vids):
    import torch
    from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone
    torch.set_num_threads(2); device = torch.device("cuda")
    model = DinoBackbone("vits16", "/workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor/backbone_best.pth").to(device).eval()
    for k, v in enumerate(vids):
        try: process(v, model, device)
        except Exception as e: print("FAIL", v, repr(e), flush=True)
        if k % 20 == 0: print(os.getpid(), k, len(vids), v, flush=True)


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); VID.mkdir(parents=True, exist_ok=True)
    labelled = {r["sample_id"].split("_")[1] for r in C.rows("all") if C.source(r) == "NEXAR"}
    vids = [r["video_id"] for r in csv.DictReader(open(CAND)) if r["video_id"] not in labelled]
    if a.limit: vids = vids[:a.limit]
    print(len(vids), "unlabelled NEXAR videos", flush=True)
    ctx = mp.get_context("spawn"); procs = [ctx.Process(target=worker, args=(vids[i::a.workers],)) for i in range(a.workers)]
    for p in procs: p.start()
    for p in procs: p.join()
    print("done", flush=True)


if __name__ == "__main__": main()
