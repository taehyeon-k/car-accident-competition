"""Experiment E: top-K ENTRY x COLLISION joint pair scorer on out-of-fold candidates.

Protocol (no validation labels touch the scorer; no val-based selection):
  * For an evaluation set V with training set T, the generator for V is the run trained on T
    (its predictions.json holds V logits).  Training candidates for T come from an inner 5-fold
    split of T (generators trained on 4/5 of T, predicting the held-out 1/5).
  * Candidates: top-K NMS peaks of ENTRY and COLLISION logits (K=8); pairs with entry <= collision.
  * Scorer: own small frame encoder on frozen DINO features + generator scores -> MLP.  Soft CE over
    the candidate pairs, target mass ~ exp(-normalized pair distance / tau).  Fixed epochs.

Usage:
  python -m stage2.long_context_v2_experiments.pair_scorer inner --gen C0_avg --seed 0 [--gen-args ...]
  python -m stage2.long_context_v2_experiments.pair_scorer fit   --gen C0_avg --seed 0 [--cv-fold k]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from stage2.spotting_experiments.train import stratified_folds
from . import common as C
from .diagnostics import peaks

D = C.REPO / "stage2/long_context_v2_experiments"
R = D / "results"
K = 8
WIN = 4  # positions for before-entry / after-collision pooling


# ---------------------------------------------------------------------------------- paths
def gen_dir(gen, seed, cv_fold=None):
    return R / gen / f"seed{seed}" if cv_fold is None else R / gen / "cv" / f"fold{cv_fold}_seed{seed}"


def inner_root(gen, seed, cv_fold=None):
    return R / gen / ("inner" if cv_fold is None else f"cv_inner/fold{cv_fold}") / f"seed{seed}"


def train_manifest(cv_fold):
    return "train" if cv_fold is None else str(D / f"folds/fold{cv_fold}_train.jsonl")


def eval_manifest(cv_fold):
    return "val" if cv_fold is None else str(D / f"folds/fold{cv_fold}_val.jsonl")


# ---------------------------------------------------------------------------------- inner OOF generators
def inner(gen, seed, cv_fold, gen_args, parallel):
    rows = C.rows(train_manifest(cv_fold))
    fold = stratified_folds(rows, 5, 1234 + seed)
    root = inner_root(gen, seed, cv_fold); root.mkdir(parents=True, exist_ok=True)
    cmds = []
    for k in range(5):
        for name, sel in (("train", lambda r: fold[r["sample_id"]] != k), ("val", lambda r: fold[r["sample_id"]] == k)):
            (root / f"inner{k}_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if sel(r)))
        if (root / f"inner{k}/predictions.json").exists(): continue
        cmds.append(f"OMP_NUM_THREADS=2 python -m stage2.long_context_v2_experiments.train --run-id {gen}_inner --seed {seed} "
                    f"--train-split {root}/inner{k}_train.jsonl --val-split {root}/inner{k}_val.jsonl --output {root}/inner{k} {gen_args}")
    procs = []
    for cmd in cmds:
        procs.append(subprocess.Popen(cmd, shell=True, cwd=C.REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if len(procs) >= parallel:
            procs.pop(0).wait()
    for p in procs: p.wait()
    missing = [k for k in range(5) if not (root / f"inner{k}/predictions.json").exists()]
    if missing: raise RuntimeError(f"inner folds failed: {missing}")


def load_preds(paths):
    out = {}
    for p in paths:
        for x in json.loads(Path(p).read_text()): out[x["sample_id"]] = x
    return out


# ---------------------------------------------------------------------------------- candidate pairs
def candidates(pred):
    t = len(pred["frames"]); radius = max(1, round(0.015 * (t - 1)))
    el, cl = np.asarray(pred["entry_logits"]), np.asarray(pred["collision_logits"])
    elp = el - np.logaddexp.reduce(el); clp = cl - np.logaddexp.reduce(cl)
    ec, cc = peaks(el, radius, K), peaks(cl, radius, K)
    pairs = [(i, j, ri, rj) for ri, i in enumerate(ec) for rj, j in enumerate(cc) if i <= j]
    if not pairs:  # degenerate: fall back to constrained argmax
        j = int(np.argmax(np.maximum.accumulate(el) + cl)); i = int(np.argmax(el[:j + 1])); pairs = [(i, j, 0, 0)]
    return pairs, elp, clp


def pair_scalars(pred, pairs, elp, clp):
    frames = np.asarray(pred["frames"], float); t = len(frames); span = max(frames[-1] - frames[0], 1)
    s = []
    for i, j, ri, rj in pairs:
        s.append([elp[i], clp[j], elp[i] + clp[j], np.log1p(ri), np.log1p(rj),
                  (frames[j] - frames[i]) / span, np.log1p(frames[j] - frames[i]), (frames[i] - frames[0]) / span,
                  (frames[j] - frames[0]) / span, float(np.argmax(elp) == i), float(np.argmax(clp) == j)])
    return np.asarray(s, np.float32)


def soft_target(pred, pairs, tau=0.01):
    frames = np.asarray(pred["frames"], float); span = max(frames[-1] - frames[0], 1)
    d = np.array([(abs(frames[i] - pred["entry_gt"]) + abs(frames[j] - pred["collision_gt"])) / span for i, j, *_ in pairs])
    w = np.exp(-(d - d.min()) / tau)
    return (w / w.sum()).astype(np.float32)


# ---------------------------------------------------------------------------------- model
class PairScorer(nn.Module):
    def __init__(self, n_scalar=11, hidden=64, token_dim=8, dropout=0.3, scalars_only=False):
        super().__init__()
        self.scalars_only = scalars_only
        self.norm = nn.LayerNorm(384); self.token = nn.Linear(384, token_dim)
        self.frame = nn.Sequential(nn.Dropout(dropout), nn.Linear(70 * token_dim, hidden), nn.GELU())
        self.local = nn.Conv1d(hidden, hidden, 5, padding=2, groups=hidden)
        self.mlp = nn.Sequential(nn.Linear((0 if scalars_only else 5 * hidden) + n_scalar, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.scalar_norm = nn.LayerNorm(n_scalar)

    def encode(self, x):  # x [T,70,384] -> [T,H]
        g = self.frame(self.token(self.norm(x.float())).flatten(1))
        return g + self.local(g.t()[None])[0].t()

    def forward(self, x, pairs, scalars):
        if self.scalars_only:
            return self.mlp(self.scalar_norm(scalars)).squeeze(-1)
        g = self.encode(x); t = len(g)
        cs = torch.cat([torch.zeros(1, g.shape[1], device=g.device), g.cumsum(0)])
        def mean(a, b):  # mean over [a, b)
            a = max(0, min(a, t - 1)); b = max(a + 1, min(b, t)); return (cs[b] - cs[a]) / (b - a)
        z = torch.stack([torch.cat([g[i], g[j], mean(i, j + 1), mean(i - WIN, i), mean(j + 1, j + 1 + WIN)]) for i, j, *_ in pairs])
        return self.mlp(torch.cat([z, self.scalar_norm(scalars)], -1)).squeeze(-1)


def build_examples(preds, row_by_id, device):
    ex = []
    for sid, p in preds.items():
        pairs, elp, clp = candidates(p)
        item = C.make_item(row_by_id[sid], "adaptive")
        assert item["frame_numbers"].tolist() == p["frames"], sid
        ex.append({"sid": sid, "pred": p, "pairs": pairs, "x": item["x"].to(device),
                   "scalars": torch.from_numpy(pair_scalars(p, pairs, elp, clp)).to(device),
                   "target": torch.from_numpy(soft_target(p, pairs)).to(device)})
    return ex


def fit(gen, seed, cv_fold, epochs, out_name, scalars_only=False):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda")
    row_by_id = {r["sample_id"]: r for r in C.rows("all")}
    iroot = inner_root(gen, seed, cv_fold)
    train_preds = load_preds([iroot / f"inner{k}/predictions.json" for k in range(5)])
    eval_preds = load_preds([gen_dir(gen, seed, cv_fold) / "predictions.json"])
    tr, ev = build_examples(train_preds, row_by_id, device), build_examples(eval_preds, row_by_id, device)
    model = PairScorer(scalars_only=scalars_only).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(tr))
    rng = np.random.default_rng(seed); hist = []
    for epoch in range(epochs):
        model.train(); tot = 0.0
        for idx in rng.permutation(len(tr)):
            e = tr[idx]
            if len(e["pairs"]) < 2: continue
            logits = model(e["x"], e["pairs"], e["scalars"])
            loss = -(e["target"] * F.log_softmax(logits, -1)).sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += float(loss.detach())
        hist.append({"epoch": epoch + 1, "train_loss": tot / len(tr)})
    model.eval()

    def decode(examples):
        out = []
        with torch.no_grad():
            for e in examples:
                s = model(e["x"], e["pairs"], e["scalars"]).cpu().numpy()
                i, j, *_ = e["pairs"][int(np.argmax(s))]
                p = dict(e["pred"]); f = p["frames"]
                base = (p["entry_frame"], p["collision_frame"])
                p["entry_frame"], p["collision_frame"] = int(f[i]), int(f[j])
                p["generator_pair"] = base
                # oracle: is any candidate pair a double hit / single-event hits?
                p["oracle_pair_hit"] = any(C.hit({**p, "entry_frame": f[a], "collision_frame": f[b]}, "entry") and
                                           C.hit({**p, "entry_frame": f[a], "collision_frame": f[b]}, "collision") for a, b, *_ in e["pairs"])
                for k2 in ("entry_logits", "collision_logits", "frames"): p.pop(k2, None)
                out.append(p)
        return out

    ev_out, tr_out = decode(ev), decode(tr)
    res = {"eval": C.breakdown(ev_out), "generator_eval": C.breakdown([{**p, "entry_frame": p["generator_pair"][0],
                                                                        "collision_frame": p["generator_pair"][1]} for p in ev_out]),
           "oracle_pair_hit_rate_eval": {s: float(np.mean([p["oracle_pair_hit"] for p in ev_out if s == "all" or C.source(p) == s]))
                                         for s in ("all", "NEXAR")},
           "train_oof": C.metrics(tr_out), "history": hist}
    out = (R / out_name / (f"seed{seed}" if cv_fold is None else f"cv/fold{cv_fold}_seed{seed}"))
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "generator": gen}, out / "pair_scorer.pt")
    C.dump(out / "predictions.json", ev_out); C.dump(out / "pair_metrics.json", res)
    # train.py-compatible metrics.json so summarize.py can aggregate
    C.dump(out / "metrics.json", {"breakdown": res["eval"], "recall_at_k": json.loads((gen_dir(gen, seed, cv_fold) / "metrics.json").read_text())["recall_at_k"],
                                  "best_epoch": epochs, "train_fpsblind_at_best": 0, "val_fpsblind_at_best": 0,
                                  "runtime": {"parameter_count": sum(x.numel() for x in model.parameters())}})
    g = res["generator_eval"]; e_ = res["eval"]
    print(f"PAIR {out_name} seed{seed} fold{cv_fold}: generator {C.short_table(g)}\n    -> scorer {C.short_table(e_)} | oracle pair hit {res['oracle_pair_hit_rate_eval']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["inner", "fit", "all"])
    ap.add_argument("--gen", required=True)
    ap.add_argument("--gen-args", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv-fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--parallel", type=int, default=5)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--scalars-only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.task in ("inner", "all"): inner(a.gen, a.seed, a.cv_fold, a.gen_args, a.parallel)
    if a.task in ("fit", "all"): fit(a.gen, a.seed, a.cv_fold, a.epochs, a.out_name or f"E_pair_{a.gen}", a.scalars_only)


if __name__ == "__main__": main()
