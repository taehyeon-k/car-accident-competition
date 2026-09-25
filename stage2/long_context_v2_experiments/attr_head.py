"""Event-conditioned side / evasion head (section 9), stacked on decoded events.

Window: native frames [e - d, c + d], d = max(c - e, 8) frames (frame counts only), 24 frames sampled uniformly.
Model: token proj -> frame MLP -> 2 residual DW-conv blocks -> attention pool -> side (2) + evasion (1).
Train on the training folds with GT events jittered by the decoder's typical error; evaluate on the held-out fold using the
events decoded by decode_motion (OOF). Fixed epochs, no validation use.
"""
from __future__ import annotations

import argparse, json

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import common as C
from .refine import Block

R = C.REPO / "stage2/long_context_v2_experiments/results"
T = 24


def window_item(sample_id, e_frame, c_frame):
    frames, feats = C.dense(sample_id); n = len(frames)
    e = int(np.clip(np.searchsorted(frames, e_frame), 0, n - 1)); c = int(np.clip(np.searchsorted(frames, c_frame), 0, n - 1))
    e, c = min(e, c), max(e, c); d = max(c - e, 8)
    idx = np.clip(np.rint(np.linspace(e - d, c + d, T)).astype(int), 0, n - 1)
    return np.asarray(feats[idx], np.float16)


class AttrHead(nn.Module):
    def __init__(self, hidden=64, token_dim=8, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(384); self.token = nn.Linear(384, token_dim)
        self.frame = nn.Sequential(nn.Dropout(dropout), nn.Linear(70 * token_dim, hidden), nn.GELU())
        self.blocks = nn.ModuleList(Block(hidden, d, dropout) for d in (1, 2))
        self.attn = nn.Linear(hidden, 1); self.side = nn.Linear(hidden, 2); self.evasion = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.frame(self.token(self.norm(x.float())).flatten(2)); m = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        for b in self.blocks: h = b(h, m)
        w = self.attn(h).squeeze(-1).softmax(-1); p = self.drop(torch.einsum("bt,bth->bh", w, h))
        return self.side(p), self.evasion(p).squeeze(-1)


def train(rows, seed, epochs, device, jitter=0.25):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = AttrHead().to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    steps = epochs * ((len(rows) + 31) // 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=.1)
    for _ in range(epochs):
        model.train(); order = rng.permutation(len(rows))
        for s in range(0, len(order), 32):
            batch = [rows[i] for i in order[s:s + 32]]; xs = []
            for r in batch:
                e, c = int(r["entry_frame"]), int(r["collision_frame"]); gap = max(c - e, 8)
                xs.append(window_item(r["sample_id"], e + rng.normal(0, jitter) * gap, c + rng.normal(0, jitter) * gap))
            x = torch.from_numpy(np.stack(xs)).to(device)
            side = torch.tensor([int(r["entry_side"] == "RIGHT") for r in batch], device=device)
            eva = torch.tensor([float(r["evasion_space"]) for r in batch], device=device)
            ls, le = model(x)
            loss = F.cross_entropy(ls, side) + F.binary_cross_entropy_with_logits(le, eva)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoded", default="C0_avg+X_ema", help="run key in decode_motion.json")
    ap.add_argument("--config", default="prior=0 gate_long=0 snap=1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    dev = torch.device("cuda"); torch.set_num_threads(4)
    # decoded OOF events per clip: re-run the decode_motion chosen configuration
    from .decode_motion import decode as dm_decode
    from .decode_priors import fit, oof
    dm = json.loads((R / "decode_motion.json").read_text())[a.decoded][a.config]
    use_prior = a.config.startswith("prior=1"); gate = "gate_long=1" in a.config; snap = "snap=1" in a.config
    folds = oof(a.decoded); final = []
    for k, items in folds.items():
        pr = fit(C.rows(str(C.REPO / f"stage2/long_context_v2_experiments/folds/fold{k}_train.jsonl"))) if use_prior else None
        beta = dm["beta_per_fold"][str(k)] if str(k) in dm["beta_per_fold"] else dm["beta_per_fold"][k]
        dec = [dm_decode(p, pr, 0.5, 1.0 if use_prior else 0, beta, gate, snap) for p in items]
        train_rows = C.rows(str(C.REPO / f"stage2/long_context_v2_experiments/folds/fold{k}_train.jsonl"))
        probs_s, probs_e = np.zeros(len(dec)), np.zeros(len(dec))
        for seed in a.seeds:
            model = train(train_rows, seed + 100 * k, a.epochs, dev)
            with torch.no_grad():
                for i, q in enumerate(dec):
                    x = torch.from_numpy(window_item(q["sample_id"], q["entry_frame"], q["collision_frame"]))[None].to(dev)
                    ls, le = model(x); probs_s[i] += float(ls.softmax(-1)[0, 1]) / len(a.seeds); probs_e[i] += float(le.sigmoid()[0]) / len(a.seeds)
        for i, q in enumerate(dec):
            q2 = dict(q); q2["entry_side"] = int(probs_s[i] >= .5); q2["evasion_space"] = int(probs_e[i] >= .5)
            q2["global_head"] = (q["entry_side"], q["evasion_space"]); final.append(q2)
        print("fold", k, "done", flush=True)
    glob_ = [{**q, "entry_side": q["global_head"][0], "evasion_space": q["global_head"][1]} for q in final]
    for name, ps in (("global heads (ensemble)", glob_), ("event-conditioned head", final)):
        b = C.breakdown(ps); n = b["source:NEXAR"]
        print(f"{name:26s} all {b['overall']['score']:.3f} side {b['overall']['side_f1']:.3f} eva {b['overall']['evasion_f1']:.3f} | "
              f"NEXAR {n['score']:.3f} side {n['side_f1']:.3f} eva {n['evasion_f1']:.3f} E {n['entry_acc']:.2f} C {n['collision_acc']:.2f} | non-NEXAR {b['source:non-NEXAR']['score']:.3f}")
    C.dump(R / f"attr_head_{a.decoded}_{a.config.replace(' ', '_')}.json", {"global": C.breakdown(glob_), "event": C.breakdown(final)})


if __name__ == "__main__": main()
