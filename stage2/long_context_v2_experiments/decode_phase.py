"""Offline decoder comparison on saved logits (same checkpoints; decoders never used for selection except the
pre-declared 'structured phase+transition (1/1)' for phase-loss runs).

Decoders: direct (constrained argmax of direct heads) | transition (constrained argmax of s_E, s_C) |
structured phase-only | structured phase+transition [pre-declared] | structured phase+transition+direct (1/1/1).
Reports fixed split (mean±std over seeds) and 5-fold CV OOF (mean±std over CV seeds), plus catastrophic-flip analysis.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

from . import common as C
from .phase_loss import decode_direct, decode_structured, decode_transition

R = C.REPO / "stage2/long_context_v2_experiments/results"
DECODERS = {
    "direct": lambda p: decode_direct(p["el"], p["cl"]),
    "transition": lambda p: decode_transition(p["ph"]),
    "struct phase": lambda p: decode_structured(p["ph"], w_phase=1, w_tr=0),
    "struct phase+tr*": lambda p: decode_structured(p["ph"], w_phase=1, w_tr=1),
    "struct phase+tr+direct": lambda p: decode_structured(p["ph"], p["el"], p["cl"], w_phase=1, w_tr=1, w_direct=1),
}
KEYS = [("overall", "all"), ("source:NEXAR", "NEXAR"), ("bin:>1000", ">1000"), ("source:non-NEXAR", "nonNEX")]


def redecode(preds, name):
    out = []
    for p in preds:
        t = {"el": torch.tensor(p["entry_logits"]), "cl": torch.tensor(p["collision_logits"])}
        if "phase_logits" in p: t["ph"] = torch.tensor(p["phase_logits"])
        elif name != "direct": return None
        i, j = DECODERS[name](t)
        q = {k: v for k, v in p.items() if k not in ("entry_logits", "collision_logits", "phase_logits", "frames")}
        q["entry_frame"], q["collision_frame"] = int(p["frames"][i]), int(p["frames"][j])
        out.append(q)
    return out


def groups(run):
    fixed = [json.loads(p.read_text()) for p in sorted((R / run).glob("seed*/predictions.json"))]
    cv = {}
    for p in sorted((R / run / "cv").glob("fold*_seed*/predictions.json")):
        cv.setdefault(p.parent.name.split("_seed")[1], []).extend(json.loads(p.read_text()))
    return fixed, [v for v in cv.values() if len(v) == 349]


def agg(list_of_preds):
    bds = [C.breakdown(ps) for ps in list_of_preds]
    res = {}
    for key, short in KEYS:
        for f in ("score", "entry_acc", "collision_acc", "entry_norm_mae", "collision_norm_mae", "entry_catastrophic", "collision_catastrophic"):
            v = [b[key][f] for b in bds if b.get(key, {}).get("n")]
            res[f"{short}.{f}"] = (float(np.mean(v)), float(np.std(v))) if v else None
    return res


def catastrophic(p):
    n = max(p["num_available_frames"] - 1, 1)
    return any(abs(p[f"{e}_frame"] - p[f"{e}_gt"]) / n > C.CATASTROPHIC for e in ("entry", "collision"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); a = ap.parse_args()
    table, flips = {}, {}
    hdr = f"{'run / decoder':42s} | {'fixed all':11s} {'fixed NEXAR':11s} {'fixed >1000':11s} | {'CV all':11s} {'CV NEXAR':11s} {'CV >1000':11s} {'CV NEX E':11s} {'CV NEX C':11s} {'NEX Enmae':9s} {'NEX Cnmae':9s} {'NEX Ecat':9s} {'NEX Ccat':9s}"
    print(hdr)
    f_ = lambda t: "     -     " if t is None else f"{t[0]:.3f}±{t[1]:.3f}"
    g_ = lambda t: "    -    " if t is None else f"{t[0]:.3f}"
    for run in a.runs:
        fixed, cv = groups(run)
        for name in DECODERS:
            fd = [redecode(ps, name) for ps in fixed]; cd = [redecode(ps, name) for ps in cv]
            if not fd or fd[0] is None: continue
            fr, cr = agg(fd), (agg(cd) if cd else {})
            table[f"{run}|{name}"] = {"fixed": fr, "cv": cr, "n_fixed": len(fd), "n_cv": len(cd)}
            print(f"{run + ' / ' + name:42s} | {f_(fr['all.score'])} {f_(fr['NEXAR.score'])} {f_(fr['>1000.score'])} | "
                  f"{f_(cr.get('all.score'))} {f_(cr.get('NEXAR.score'))} {f_(cr.get('>1000.score'))} {f_(cr.get('NEXAR.entry_acc'))} "
                  f"{f_(cr.get('NEXAR.collision_acc'))} {g_(cr.get('NEXAR.entry_norm_mae'))} {g_(cr.get('NEXAR.collision_norm_mae'))} "
                  f"{g_(cr.get('NEXAR.entry_catastrophic'))} {g_(cr.get('NEXAR.collision_catastrophic'))}")
        # catastrophic flips direct -> structured (pre-declared), CV OOF, long clips
        if cv and "phase_logits" in cv[0][0]:
            rows = []
            for s, ps in enumerate(cv):
                d = {p["sample_id"]: p for p in redecode(ps, "direct")}; st = {p["sample_id"]: p for p in redecode(ps, "struct phase+tr*")}
                for sid in d:
                    if d[sid]["num_available_frames"] <= 1000: continue
                    cd_, cs_ = catastrophic(d[sid]), catastrophic(st[sid])
                    if cd_ != cs_:
                        rows.append({"cv_seed": s, "sample_id": sid, "fixed_by_structured": cd_ and not cs_,
                                     "direct": (d[sid]["entry_frame"], d[sid]["collision_frame"]), "structured": (st[sid]["entry_frame"], st[sid]["collision_frame"]),
                                     "gt": (d[sid]["entry_gt"], d[sid]["collision_gt"]), "n": d[sid]["num_available_frames"]})
            fixed_n = sum(r["fixed_by_structured"] for r in rows); broke_n = len(rows) - fixed_n
            n_long = sum(p["num_available_frames"] > 1000 for p in cv[0]) * len(cv)
            cat_d = sum(catastrophic(p) for ps in cv for p in redecode(ps, "direct") if p["num_available_frames"] > 1000)
            cat_s = sum(catastrophic(p) for ps in cv for p in redecode(ps, "struct phase+tr*") if p["num_available_frames"] > 1000)
            flips[run] = {"long_clip_evals": n_long, "catastrophic_direct": cat_d, "catastrophic_structured": cat_s,
                          "fixed_by_structured": fixed_n, "broken_by_structured": broke_n, "examples": rows}
            print(f"   >1000-frame CV OOF ({n_long} clip-evals): catastrophic direct {cat_d} -> structured {cat_s}; "
                  f"fixed {fixed_n}, broken {broke_n}")
            for r in rows[:8]:
                print(f"     {'FIXED ' if r['fixed_by_structured'] else 'BROKEN'} {r['sample_id']} seed{r['cv_seed']} gt {r['gt']} direct {r['direct']} structured {r['structured']}")
    C.dump(R / "phase_decoders.json", {"table": table, "flips": flips})


if __name__ == "__main__": main()
