"""ENTRY shortcut diagnostics (FPS used only offline, for the 0.3 s metric and gap in seconds).
For each run (CV OOF over all seeds, or LOSO held-out predictions): ENTRY hit rate by GT ENTRY->COLLISION gap bin, median predicted vs
GT gap, and P(ENTRY hit | COLLISION miss) - a model that anchors ENTRY on the collision fails ENTRY when COLLISION fails.
Usage: python -m stage2.generalization.entry_gap RUN [RUN ...] [--loso]
"""
from __future__ import annotations

import argparse, json

import numpy as np

from stage2.long_context_v2_experiments import common as C

ROOTS = [C.REPO / "stage2/generalization/results", C.REPO / "stage2/aux_signal_experiments/results", C.REPO / "stage2/phase_study/results"]
BINS = [(0, 0.5, "<0.5s"), (0.5, 1.0, "0.5-1"), (1.0, 1.5, "1-1.5"), (1.5, 99, ">1.5")]


def preds(run, loso):
    root = next(r for r in ROOTS if (r / run).is_dir())
    pat = "*_seed*/predictions.json" if loso else "cv/fold*_seed*/predictions.json"
    return [p for f in sorted((root / run).glob(pat)) for p in json.loads(f.read_text())]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--loso", action="store_true"); a = ap.parse_args()
    fps = C.fps_table()
    print("| run | " + " | ".join(f"E hit {b[2]}" for b in BINS) + " | E hit all | pred/GT gap (s) | P(E hit given C miss) | long-gap (>1s) E hit NEXAR / AIHUB |")
    print("|---|" + "---|" * (len(BINS) + 4))
    for run in a.runs:
        ps = preds(run, a.loso); rec = []
        for p in ps:
            f = fps[p["sample_id"]]
            rec.append((C.source(p), (p["collision_gt"] - p["entry_gt"]) / f, abs(p["entry_frame"] - p["entry_gt"]) / f <= .300001,
                        abs(p["collision_frame"] - p["collision_gt"]) / f <= .300001, (p["collision_frame"] - p["entry_frame"]) / f))
        cells = []
        for lo, hi, _ in BINS:
            sub = [r for r in rec if lo <= r[1] < hi]; cells.append(f"{np.mean([r[2] for r in sub]):.3f}" if sub else "—")
        gt = np.median([r[1] for r in rec]); pg = np.median([r[4] for r in rec])
        miss = [r[2] for r in rec if not r[3]]
        lg = {s: np.mean([r[2] for r in rec if r[0] == s and r[1] >= 1.0]) for s in ("NEXAR", "AIHUB")}
        print(f"| {run} | " + " | ".join(cells) + f" | {np.mean([r[2] for r in rec]):.3f} | {pg:.2f}/{gt:.2f} | {np.mean(miss):.3f} | "
              f"{lg['NEXAR']:.3f} / {lg['AIHUB']:.3f} |")


if __name__ == "__main__": main()
