"""Near-duplicate audit: expansion pool (cache_unl, the clips in cache_unl_labels) vs the 349 labelled clips, and labelled vs labelled.

Frame signature = mean over the 7x10 pooled geometry-DINO tokens, L2-normalised (same extractor for both caches, parity-checked).
Each clip is represented by up to 48 evenly spaced frames. For a pair (A, B): match(A, B) = fraction of A's frames whose best cosine
to any frame of B is >= THR. A duplicate (same footage, possibly trimmed/re-encoded) has match close to 1; unrelated clips from the
same source share road scenes but rarely exceed THR on most frames. Reports the top pairs and the score distribution.
Output results/dup_audit.json.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C

UNL = C.REPO / "stage2/generalization/cache_unl"
LAB = C.REPO / "stage2/generalization/cache_unl_labels/labels_fold0.json"
NF, THR = 48, 0.90


def sig(x):
    idx = np.unique(np.linspace(0, len(x) - 1, min(NF, len(x))).round().astype(int))
    f = torch.from_numpy(np.asarray(x[idx], np.float32)); f = (f - f.mean(-1, keepdim=True)).flatten(1)  # spatial layout kept
    return torch.nn.functional.normalize(f, dim=-1)


def main():
    dev = torch.device("cuda")
    lab = {r["sample_id"]: sig(C.dense(r["sample_id"])[1]) for r in C.rows("all")}
    unl = {sid: sig(np.load(UNL / f"{sid}.npy", mmap_mode="r")) for sid in json.loads(LAB.read_text())}
    src = {r["sample_id"]: C.source(r) for r in C.rows("all")}
    fold = {}
    for k in range(5):
        for r in C.rows(str(C.REPO / f"stage2/long_context_v2_experiments/folds/fold{k}_val.jsonl")): fold[r["sample_id"]] = k
    L = list(lab)

    def scores(a, pool, skip_self=False):
        out = []
        A = a.to(dev)
        for sid in pool:
            if skip_self and sid is skip_self: continue
            B = lab[sid].to(dev); s = (A @ B.T).max(1).values
            out.append(((s >= THR).float().mean().item(), s.mean().item(), sid))
        return out

    res = {"unl_vs_lab": [], "lab_vs_lab": []}
    for u, a in unl.items():
        best = max(scores(a, L)); res["unl_vs_lab"].append((u, best[2], best[0], best[1]))
    for i, x in enumerate(L):
        best = max((s for s in scores(lab[x], L) if s[2] != x)); res["lab_vs_lab"].append((x, best[2], best[0], best[1], fold[x], fold[best[2]]))
    for key in res:
        rows = sorted(res[key], key=lambda t: -t[2])
        v = np.array([t[2] for t in rows])
        print(f"{key}: n {len(v)}  match>=0.9 {int((v >= .9).sum())}  >=0.5 {int((v >= .5).sum())}  >=0.2 {int((v >= .2).sum())}  median {np.median(v):.3f}")
        for t in rows[:15]: print("   ", *[f"{z:.3f}" if isinstance(z, float) else z for z in t])
    C.dump(C.REPO / "stage2/generalization/results/dup_audit.json", res)


if __name__ == "__main__": main()
