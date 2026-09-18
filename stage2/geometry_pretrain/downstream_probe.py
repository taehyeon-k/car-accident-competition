"""Controlled Stage-2 comparison: original vs geometry-adapted frozen DINOv3 ViT-S.

extract: frozen backbone -> per-frame final-norm patch tokens (the Stage-2
         ``x_norm_patchtokens``), average-pooled to a 7x10 grid, at ~10 fps
         (stride round(fps/10)); letterboxed to 448x800 so no view is cropped.
run:     the SAME small temporal probe + protocol for every feature set:
         fixed official split (Stage-2 train/val, never seen by geometry
         pretraining) and 5-fold source/label-stratified video-level CV x seeds.
Loss/decoding/metrics are Stage 2's own (joint_loss, constrained_decode,
JointMetricAccumulator), so scores are the competition-style score.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from stage2.geometry_pretrain.common import INPUT_HW, read_jsonl
from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone
from stage2.utils.joint_losses import joint_loss
from stage2.utils.joint_metrics import JointMetricAccumulator, joint_metric_packet

GRID = (7, 10)
FEAT_ROOT = Path("/workspace/cache/geometry_pretrain/stage2_probe_features")
FRAME_CACHE = Path("/workspace/cache/geometry_pretrain/stage2_probe_frames")  # letterboxed 448x800 JPEGs
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def letterbox(bgr):
    h, w = bgr.shape[:2]
    s = min(INPUT_HW[0] / h, INPUT_HW[1] / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((*INPUT_HW, 3), np.uint8)
    out[:] = (124, 116, 104)  # ImageNet mean (BGR) -> zero after normalization
    y0, x0 = (INPUT_HW[0] - nh) // 2, (INPUT_HW[1] - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = img
    return out[:, :, ::-1]


def frame_stride(fps: float) -> int:
    return max(1, int(round(fps / 10.0)))


class Frames(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, cached = self.items[i]
        if cached is not None and cached.is_file():
            rgb = cv2.imread(str(cached))[:, :, ::-1]
        else:
            rgb = letterbox(cv2.imread(path))
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                tmp = cached.with_name(cached.name + ".tmp.jpg")
                cv2.imwrite(str(tmp), np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 95])
                tmp.rename(cached)
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)


@torch.no_grad()
def extract(args):
    rows = read_jsonl(args.manifest)
    out_dir = FEAT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone = DinoBackbone("vits16", args.backbone).cuda().eval()
    for r in tqdm(rows, desc=f"extract {args.name}"):
        out = out_dir / f"{r['sample_id']}.npy"
        if out.is_file():
            continue
        stride = frame_stride(r["native_fps"])
        idx = list(range(0, r["num_frames"], stride))
        frames_dir = r["frames_dir"]
        paths = [(f"{frames_dir}/{i:06d}.jpg", FRAME_CACHE / r["sample_id"] / f"{i:06d}.jpg" if args.frame_cache else None) for i in idx]
        loader = DataLoader(Frames(paths), batch_size=32, num_workers=args.workers)
        feats = []
        for x in loader:
            x = (x.cuda().float() - MEAN.cuda()) / STD.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = backbone(x)["patch"]
            f = F.adaptive_avg_pool2d(f.float(), GRID)  # B,C,7,10
            feats.append(f.flatten(2).transpose(1, 2).half().cpu())
        arr = torch.cat(feats).numpy()
        tmp = out.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        tmp.rename(out)
    (out_dir / "meta.json").write_text(json.dumps({"backbone": args.backbone, "grid": GRID, "fps_target": 10}))


# ----------------------------------------------------------------------------
# Probe
# ----------------------------------------------------------------------------


_FEATURES: dict = {}  # process-wide cache: (name, sample_id) -> tensor


class VideoFeatures(Dataset):
    def __init__(self, rows, name, train=False, max_len=None):
        self.rows, self.name, self.train, self.max_len = rows, name, train, max_len
        self.cache = _FEATURES

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        key = (self.name, r["sample_id"])
        if key not in self.cache:
            self.cache[key] = torch.from_numpy(np.load(FEAT_ROOT / self.name / f"{r['sample_id']}.npy"))
        x = self.cache[key]
        stride = frame_stride(r["native_fps"])
        idx = torch.arange(0, r["num_frames"], stride)
        T = len(idx)
        assert x.shape[0] == T, (r["sample_id"], x.shape, T)
        # Event labels -> nearest sampled frame.
        e = int(torch.argmin((idx - r["entry_frame"]).abs()))
        c = int(torch.argmin((idx - r["collision_frame"]).abs()))
        return {
            "x": x,
            "seconds": idx.float() / float(r["native_fps"]),
            "entry_index": e,
            "collision_index": c,
            "entry_side": int(r["entry_side"] == "RIGHT"),
            "evasion": int(r["evasion_space"]),
            # True event times, for exact scoring against native labels.
            "entry_s": r["entry_frame"] / float(r["native_fps"]),
            "collision_s": r["collision_frame"] / float(r["native_fps"]),
        }


def collate(items):
    T = max(it["x"].shape[0] for it in items)
    B = len(items)
    N, C = items[0]["x"].shape[1:]
    x = torch.zeros(B, T, N, C, dtype=torch.float16)
    valid = torch.zeros(B, T, dtype=torch.bool)
    secs = torch.zeros(B, T)
    for b, it in enumerate(items):
        t = it["x"].shape[0]
        x[b, :t] = it["x"]
        valid[b, :t] = True
        secs[b, :t] = it["seconds"]
        if t < T:  # keep seconds monotone in the padding (masked anyway)
            secs[b, t:] = it["seconds"][-1] + torch.arange(1, T - t + 1) * 0.1
    out = {"x": x, "time_valid": valid, "frame_seconds": secs}
    for k in ("entry_index", "collision_index", "entry_side", "evasion"):
        out[k] = torch.tensor([it[k] for it in items])
    for k in ("entry_s", "collision_s"):
        out[k] = torch.tensor([it[k] for it in items])
    return out


class TemporalProbe(nn.Module):
    """Deliberately small: token projection, flatten, 2 dilated temporal convs."""

    def __init__(self, n_tokens=GRID[0] * GRID[1], dim=384, tok=32, hidden=192, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.tok = nn.Linear(dim, tok)
        self.frame = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_tokens * tok, hidden), nn.GELU())
        self.t1 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.t2 = nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2)
        self.drop = nn.Dropout(dropout)
        self.event = nn.Linear(hidden, 2)
        self.attn = nn.Linear(hidden, 1)
        self.side = nn.Linear(hidden, 2)
        self.evasion = nn.Linear(hidden, 1)

    def forward(self, x, valid):
        B, T, N, C = x.shape
        h = self.tok(self.norm(x.float())).reshape(B, T, -1)
        h = self.frame(h).transpose(1, 2)  # B,H,T
        h = h + F.gelu(self.t1(h))
        h = h + F.gelu(self.t2(self.drop(h)))
        h = h.transpose(1, 2)  # B,T,H
        ev = self.event(self.drop(h))
        neg = torch.finfo(ev.dtype).min / 4
        entry = ev[..., 0].masked_fill(~valid, neg)
        collision = ev[..., 1].masked_fill(~valid, neg)
        a = self.attn(h)[..., 0].masked_fill(~valid, neg).softmax(-1)
        pooled = torch.einsum("bt,bth->bh", a, h)
        return {
            "entry_logits": entry,
            "collision_logits": collision,
            "side_logits": self.side(self.drop(pooled)),
            "evasion_logits": self.evasion(self.drop(pooled))[:, 0],
        }


def to_dev(batch, dev):
    return {k: v.to(dev) for k, v in batch.items()}


def train_eval(train_rows, val_rows, name, seed, cfg, dev="cuda"):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = TemporalProbe(dropout=cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    epochs = cfg["epochs"]
    tl = DataLoader(VideoFeatures(train_rows, name, True), batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate,
                    generator=torch.Generator().manual_seed(seed))
    vl = DataLoader(VideoFeatures(val_rows, name), batch_size=16, collate_fn=collate)
    steps = epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    for ep in range(epochs):
        model.train()
        for batch in tl:
            batch = to_dev(batch, dev)
            x = batch["x"]
            if cfg.get("feat_noise", 0) > 0:
                x = x + cfg["feat_noise"] * torch.randn_like(x.float()).half()
            out = model(x, batch["time_valid"])
            loss, _ = joint_loss(out, batch, entry_sigma_seconds=0.15, collision_sigma_seconds=0.10)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    acc = JointMetricAccumulator()
    exact = defaultdict(float)
    with torch.no_grad():
        for batch in vl:
            batch = to_dev(batch, dev)
            out = model(batch["x"], batch["time_valid"])
            acc.update(joint_metric_packet(out, batch))
            # Exact scoring vs native-frame labels (sampling-grid independent).
            from stage2.utils.joint_losses import constrained_decode

            e, c = constrained_decode(out["entry_logits"], out["collision_logits"])
            es = batch["frame_seconds"].gather(1, e[:, None])[:, 0]
            cs = batch["frame_seconds"].gather(1, c[:, None])[:, 0]
            exact["entry"] += float(((es - batch["entry_s"]).abs() <= 0.3 + 1e-6).sum())
            exact["collision"] += float(((cs - batch["collision_s"]).abs() <= 0.3 + 1e-6).sum())
            exact["n"] += len(es)
    m = acc.compute()
    m["acc_entry_0.3s_native"] = exact["entry"] / exact["n"]
    m["acc_collision_0.3s_native"] = exact["collision"] / exact["n"]
    m["competition_score_native"] = (
        0.35 * m["acc_entry_0.3s_native"] + 0.35 * m["acc_collision_0.3s_native"]
        + 0.15 * m["f1_entry_side_macro"] + 0.15 * m["f1_evasion_space_macro"]
    )
    return m


def stratified_folds(rows, k, seed):
    from stage2.utils.utils import source_name

    strata = defaultdict(list)
    for r in rows:
        strata[(source_name(r["source_id"]), r["entry_side"], r["evasion_space"])].append(r["sample_id"])
    rng = np.random.default_rng(seed)
    fold_of = {}
    offset = 0
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        for i, sid in enumerate(ids):
            fold_of[sid] = (i + offset) % k
        offset += len(ids)
    return fold_of


def run(args):
    cfg = {"epochs": args.epochs, "lr": args.lr, "wd": args.wd, "batch_size": args.batch_size, "dropout": args.dropout,
           "feat_noise": args.feat_noise}
    train = read_jsonl(args.train_manifest)
    val = read_jsonl(args.val_manifest)
    all_rows = train + val
    results = {"config": cfg, "fixed_split": {}, "cv": {}, "features": args.features}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for name in args.features:
        per_seed = [train_eval(train, val, name, s, cfg) for s in args.seeds]
        results["fixed_split"][name] = per_seed
        print(name, "fixed", {k: round(float(np.mean([m[k] for m in per_seed])), 4) for k in per_seed[0]}, flush=True)
    if args.folds > 1:
        fold_of = stratified_folds(all_rows, args.folds, 42)
        for name in args.features:
            runs = []
            for f in range(args.folds):
                tr = [r for r in all_rows if fold_of[r["sample_id"]] != f]
                va = [r for r in all_rows if fold_of[r["sample_id"]] == f]
                for s in args.seeds:
                    m = train_eval(tr, va, name, s, cfg)
                    m.update(fold=f, seed=s)
                    runs.append(m)
            results["cv"][name] = runs
            print(name, "cv", {k: round(float(np.mean([m[k] for m in runs])), 4) for k in runs[0] if k not in ("fold", "seed")}, flush=True)
    results["time_s"] = time.time() - t0
    out.write_text(json.dumps(results, indent=1))
    summarize(results)


def summarize(results):
    keys = ["competition_score_native", "acc_entry_0.3s_native", "acc_collision_0.3s_native", "f1_entry_side_macro", "f1_evasion_space_macro"]
    for block in ("fixed_split", "cv"):
        if not results.get(block):
            continue
        print(f"== {block}")
        for name, runs in results[block].items():
            if block == "cv":
                # Per seed: pool folds (mean over folds), then mean/std across seeds and folds.
                vals = {k: [m[k] for m in runs] for k in keys}
            else:
                vals = {k: [m[k] for m in runs] for k in keys}
            print(name.ljust(12), " ".join(f"{k.replace('_native','')}={np.mean(v):.4f}±{np.std(v):.4f}" for k, v in vals.items()))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--backbone", required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--manifest", default="/workspace/data/stage2/manifests/all.jsonl")
    e.add_argument("--workers", type=int, default=5)
    e.add_argument("--frame-cache", action="store_true", help="read/write letterboxed frames (q95 JPEG) to speed up repeated extraction")
    r = sub.add_parser("run")
    r.add_argument("--features", nargs="+", required=True)
    r.add_argument("--train-manifest", default="/workspace/data/stage2/manifests/train.jsonl")
    r.add_argument("--val-manifest", default="/workspace/data/stage2/manifests/val.jsonl")
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--epochs", type=int, default=40)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--wd", type=float, default=0.05)
    r.add_argument("--batch-size", type=int, default=8)
    r.add_argument("--dropout", type=float, default=0.3)
    r.add_argument("--feat-noise", type=float, default=0.0)
    r.add_argument("--output", default="/workspace/outputs/geometry_pretrain/stage2_probe/results.json")
    args = parser.parse_args()
    extract(args) if args.cmd == "extract" else run(args)


if __name__ == "__main__":
    main()
