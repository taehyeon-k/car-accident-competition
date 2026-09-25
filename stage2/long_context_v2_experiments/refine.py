"""Experiment G: native-frame candidate refinement.

A small local head reads the dense native-frame features in a +-W frame window around an anchor and predicts
ENTRY / COLLISION logits over the window.  Training windows are centred at GT + random offset (generator-independent,
fixed epochs, no validation use).  At inference the anchors are the generator's top-R candidate pairs; each pair is
refined and re-scored (generator pair log-prob + local peak log-prob), then entry <= collision is enforced.
Frame numbers / counts only; no FPS, duration or timestamps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import common as C
from .diagnostics import peaks

R = C.REPO / "stage2/long_context_v2_experiments/results"


class Block(nn.Module):
    def __init__(self, h, dilation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(h)
        self.dw = nn.Conv1d(h, h, 5, padding=2 * dilation, dilation=dilation, groups=h)
        self.pw = nn.Linear(h, h); self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):  # x [B,T,H]
        y = self.dw(self.norm(x).transpose(1, 2)).transpose(1, 2)
        return (x + self.drop(self.pw(F.gelu(y)))) * mask[..., None]


class LocalHead(nn.Module):
    def __init__(self, hidden=64, token_dim=8, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(384); self.token = nn.Linear(384, token_dim)
        self.frame = nn.Sequential(nn.Dropout(dropout), nn.Linear(70 * token_dim, hidden), nn.GELU())
        self.blocks = nn.ModuleList(Block(hidden, d, dropout) for d in (1, 2, 4))
        self.out = nn.Linear(hidden, 2)

    def forward(self, x, mask):  # x [B,T,70,384]
        h = self.frame(self.token(self.norm(x.float())).flatten(2)) * mask[..., None]
        for b in self.blocks: h = b(h, mask)
        return self.out(h).masked_fill(~mask[..., None], -1e4)  # [B,T,2]


def window(sample_id, anchor_idx, W):
    frames, feats = C.dense(sample_id)
    n = len(frames); lo, hi = anchor_idx - W, anchor_idx + W + 1
    x = np.zeros((2 * W + 1, 70, 384), np.float16); mask = np.zeros(2 * W + 1, bool)
    a, b = max(lo, 0), min(hi, n)
    x[a - lo:b - lo] = feats[a:b]; mask[a - lo:b - lo] = True
    return x, mask, lo


def train_head(rows, seed, W, epochs, device, per_clip=4, sigma=1.5):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = LocalHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    steps = epochs * ((len(rows) * per_clip + 63) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=.1)
    idx_of = {}
    for r in rows:
        frames, _ = C.dense(r["sample_id"])
        idx_of[r["sample_id"]] = (int(np.searchsorted(frames, r["entry_frame"])), int(np.searchsorted(frames, r["collision_frame"])))
    hist = []
    for ep in range(epochs):
        samples = []
        for r in rows:
            ei, ci = idx_of[r["sample_id"]]
            for _ in range(per_clip):
                # anchor between/around the two events so both targets are usually in the window
                ev = rng.integers(2)
                base = (ei, ci)[ev]
                spread = W * (0.8 if rng.random() < 0.7 else 0.25)
                samples.append((r["sample_id"], int(round(base + rng.uniform(-spread, spread))), ei, ci))
        rng.shuffle(samples); tot = 0.0
        model.train()
        for s in range(0, len(samples), 64):
            chunk = samples[s:s + 64]
            xs, ms, tg, tv = [], [], [], []
            pos = np.arange(2 * W + 1)
            for sid, anchor, ei, ci in chunk:
                x, m, lo = window(sid, anchor, W); xs.append(x); ms.append(m)
                t, v = [], []
                for g in (ei, ci):
                    k = g - lo; inside = 0 <= k <= 2 * W
                    w = np.exp(-0.5 * ((pos - k) / sigma) ** 2) * m if inside else np.zeros(2 * W + 1)
                    t.append(w / max(w.sum(), 1e-8)); v.append(inside and w.sum() > 0)
                tg.append(np.stack(t, -1)); tv.append(v)
            x = torch.from_numpy(np.stack(xs)).to(device); m = torch.from_numpy(np.stack(ms)).to(device)
            tgt = torch.from_numpy(np.stack(tg)).float().to(device); val = torch.tensor(tv, device=device)
            logits = model(x, m)
            loss = -(tgt * F.log_softmax(logits.float(), 1)).sum(1)  # [B,2]
            loss = (loss * val).sum() / val.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += float(loss.detach())
        hist.append({"epoch": ep + 1, "loss": tot / ((len(samples) + 63) // 64)})
    return model.eval(), hist


def pair_candidates(pred, R_):
    """Top-R constrained pairs from the generator logits (entry<=collision), with joint log-prob."""
    el, cl = np.asarray(pred["entry_logits"]), np.asarray(pred["collision_logits"])
    elp, clp = el - np.logaddexp.reduce(el), cl - np.logaddexp.reduce(cl)
    t = len(el); radius = max(1, round(0.015 * (t - 1)))
    ec, cc = peaks(el, radius, 8), peaks(cl, radius, 8)
    pairs = sorted(((elp[i] + clp[j], i, j) for i in ec for j in cc if i <= j), reverse=True)[:R_]
    return pairs or [(0.0, int(np.argmax(elp)), int(np.argmax(elp)))]


@torch.inference_mode()
def refine_preds(model, preds, W, R_, device, lam=1.0, min_frames=0):
    out = []
    for p in preds:
        frames_all, _ = C.dense(p["sample_id"])
        q = {k: v for k, v in p.items() if k not in ("entry_logits", "collision_logits", "frames", "phase_logits")}
        q["generator_pair"] = (p["entry_frame"], p["collision_frame"])
        if len(frames_all) <= min_frames: out.append(q); continue
        best = None
        for score, i, j in pair_candidates(p, R_):
            res = []
            for ev, f in ((0, p["frames"][i]), (1, p["frames"][j])):
                a = int(np.searchsorted(frames_all, f))
                x, m, lo = window(p["sample_id"], a, W)
                lg = model(torch.from_numpy(x)[None].to(device), torch.from_numpy(m)[None].to(device))[0, :, ev].float()
                lp = F.log_softmax(lg, 0); k = int(lp.argmax())
                res.append((int(frames_all[min(max(lo + k, 0), len(frames_all) - 1)]), float(lp[k])))
            (ef, el_), (cf, cl_) = res
            if ef > cf: ef = cf
            total = score + lam * (el_ + cl_)
            if best is None or total > best[0]: best = (total, ef, cf)
        q["entry_frame"], q["collision_frame"] = best[1], best[2]
        out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="results run id with predictions.json (logits), e.g. C0_avg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv-fold", type=int, default=None)
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out-name", default=None)
    a = ap.parse_args()
    torch.set_num_threads(4); device = torch.device("cuda")
    if a.cv_fold is None:
        train_rows = C.rows("train"); gdir = R / a.gen / f"seed{a.seed}"; sub = f"seed{a.seed}"
    else:
        train_rows = C.rows(str(C.REPO / f"stage2/long_context_v2_experiments/folds/fold{a.cv_fold}_train.jsonl"))
        gdir = R / a.gen / "cv" / f"fold{a.cv_fold}_seed{a.seed}"; sub = f"cv/fold{a.cv_fold}_seed{a.seed}"
    preds = json.loads((gdir / "predictions.json").read_text())
    model, hist = train_head(train_rows, a.seed, a.W, a.epochs, device)
    variants = {}
    for R_ in (1, 3):
        for mf in (0, 128):
            variants[f"R{R_}_min{mf}"] = refine_preds(model, preds, a.W, R_, device, min_frames=mf)
    gen = [{**{k: v for k, v in p.items() if k not in ("entry_logits", "collision_logits", "frames")}} for p in preds]
    res = {"generator": C.breakdown(gen), **{k: C.breakdown(v) for k, v in variants.items()}, "history": hist}
    name = a.out_name or f"G_refine_{a.gen}_W{a.W}"
    out = R / name / sub; out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "W": a.W}, out / "local_head.pt")
    C.dump(out / "refine_metrics.json", res)
    for k, v in variants.items(): C.dump(out / f"predictions_{k}.json", v)
    # summarize.py-compatible: main variant = R3 on all clips
    main_v = "R3_min0"
    C.dump(out / "predictions.json", variants[main_v])
    C.dump(out / "metrics.json", {"breakdown": res[main_v], "recall_at_k": json.loads((gdir / "metrics.json").read_text())["recall_at_k"],
                                  "best_epoch": a.epochs, "train_fpsblind_at_best": 0, "val_fpsblind_at_best": 0,
                                  "runtime": {"parameter_count": sum(x.numel() for x in model.parameters())}})
    print(f"REFINE {name} {sub}")
    for k in ("generator", *variants):
        print(f"   {k:10s} {C.short_table(res[k])}")


if __name__ == "__main__": main()
