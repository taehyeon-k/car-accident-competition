"""Positional-artifact test: held-out NEXAR clips (CV fold models, OOF) cropped to L frames with the ENTRY..COLLISION
midpoint placed at relative position r of the crop. Each model uses its own pre-declared decoder."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from . import common as C
from .ensemble import load

R = C.REPO / "stage2/long_context_v2_experiments/results"


def crop(n, e, c, L, r):
    if n <= L: return 0, n
    mid = (e + c) / 2; start = int(round(mid - r * (L - 1)))
    start = min(max(start, c - L + 1, 0), e, n - L)
    return start, start + L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--L", type=int, default=600)
    ap.add_argument("--force-direct", action="store_true")
    a = ap.parse_args(); dev = torch.device("cuda"); torch.set_num_threads(4); out = {}
    for run in a.runs:
        res = {}
        for r_ in ("full", 0.3, 0.5, 0.7):
            per_seed = {}
            for ck in sorted((R / run / "cv").glob("fold*_seed*/checkpoint.pt")):
                k = int(ck.parent.name[4]); seed = ck.parent.name.split("_seed")[1]; model = load(ck, dev)
                if a.force_direct: model.decoder = "direct"
                rows = [x for x in C.rows(str(C.REPO / f"stage2/long_context_v2_experiments/folds/fold{k}_val.jsonl")) if C.source(x) == "NEXAR"]
                items, pos = [], []
                for row in rows:
                    fr, _ = C.dense(row["sample_id"])
                    e = int(np.searchsorted(fr, row["entry_frame"])); c = int(np.searchsorted(fr, row["collision_frame"]))
                    s, t = (0, len(fr)) if r_ == "full" else crop(len(fr), e, c, a.L, r_)
                    items.append(C.make_item(row, "adaptive", s, t)); pos.append(((e + c) / 2 - s) / (t - s - 1))
                preds = C.run(model, items, dev)
                for p, q in zip(preds, pos): p["event_rel_pos"] = q
                per_seed.setdefault(seed, []).extend(preds)
            ms = [C.metrics(v) for v in per_seed.values()]
            rel = np.mean([p["event_rel_pos"] for v in per_seed.values() for p in v])
            res[str(r_)] = {k2: float(np.mean([m[k2] for m in ms])) for k2 in ("score", "entry_acc", "collision_acc", "entry_catastrophic", "collision_catastrophic")}
            res[str(r_)]["mean_event_rel_pos"] = float(rel)
        out[run] = res
        print(run, " | ".join(f"r={k}: S {v['score']:.3f} E {v['entry_acc']:.2f} C {v['collision_acc']:.2f} Ccat {v['collision_catastrophic']:.2f} (pos {v['mean_event_rel_pos']:.2f})" for k, v in res.items()), flush=True)
    C.dump(R / f"crop_shift_L{a.L}{'_direct' if a.force_direct else ''}_{'_'.join(a.runs)}.json", out)


if __name__ == "__main__": main()
