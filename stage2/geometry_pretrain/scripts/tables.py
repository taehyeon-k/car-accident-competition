"""Collect evaluation JSONs into Markdown tables for the report.

    python -m stage2.geometry_pretrain.scripts.tables geometry --evals NAME=path.json ...
    python -m stage2.geometry_pretrain.scripts.tables probe --results path.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

GEOMETRY_ROWS = [
    ("bdd100k/drivable_miou", "BDD drivable mIoU (3-cls, human)", True),
    ("bdd100k/ego_lane_iou", "BDD ego/direct-lane IoU (human)", True),
    ("bdd100k/lane_bf1", "BDD lane boundary-F1 ±4px (human)", True),
    ("bdd100k/lane_iou", "BDD lane IoU (human)", True),
    ("bdd100k/curb_bf1", "BDD curb boundary-F1 (human)", True),
    ("tusimple/lane_bf1", "TuSimple lane boundary-F1 (human)", True),
    ("tusimple/ego_lane_recall", "TuSimple ego-lane recall (derived)", True),
    ("bdd100k/object_miou", "BDD object mIoU (SAM@human boxes)", True),
    ("bdd100k/vehicle_iou", "BDD vehicle IoU", True),
    ("bdd100k/contact_bf1", "BDD contact boundary-F1 (derived)", True),
    ("bdd100k/contact_loc_err_px", "BDD contact row error px (derived)", False),
    ("bdd100k/depth_delta1", "BDD depth δ<1.25 vs teacher", True),
    ("bdd100k/depth_absrel", "BDD depth AbsRel vs teacher", False),
    ("bdd100k/depth_ordinal", "BDD depth ordinal acc", True),
    ("bdd100k/vp_err_px", "BDD VP error px (derived)", False),
    ("bdd100k/horizon_err_px", "BDD horizon error px", False),
    ("tusimple/vp_err_px", "TuSimple VP error px", False),
    ("accident/drivable_miou", "Accident drivable mIoU vs teacher", True),
    ("accident/lane_bf1", "Accident lane bF1 vs teacher", True),
    ("accident/object_miou", "Accident object mIoU vs teacher", True),
    ("accident/contact_bf1", "Accident contact bF1 (derived)", True),
    ("accident/depth_delta1", "Accident depth δ1 vs teacher", True),
    ("baton/drivable_miou", "BATON drivable mIoU vs teacher", True),
    ("baton/depth_delta1", "BATON depth δ1 vs teacher", True),
    ("tusimple/flow_epe", "TuSimple flow EPE px @448", False),
    ("tusimple/flow_epe_dt_l", "TuSimple flow EPE (400 ms)", False),
    ("accident/flow_epe", "Accident flow EPE px", False),
    ("accident/flow_acc3", "Accident flow <3px acc", True),
    ("baton/flow_epe", "BATON flow EPE px", False),
    ("geometry_score", "Geometry selection score", True),
]


def geometry(args):
    evals = {}
    for item in args.evals:
        name, path = item.split("=", 1)
        evals[name] = json.load(open(path))
    names = list(evals)
    print("| Metric | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for key, label, higher in GEOMETRY_ROWS:
        vals = [evals[n].get(key) for n in names]
        if all(v is None for v in vals):
            continue
        present = [v for v in vals if v is not None]
        best = max(present) if higher else min(present)
        cells = []
        for v in vals:
            if v is None:
                cells.append("–")
            else:
                s = f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
                cells.append(f"**{s}**" if len(present) > 1 and v == best else s)
        print(f"| {label} {'↑' if higher else '↓'} | " + " | ".join(cells) + " |")


def probe(args):
    r = json.load(open(args.results))
    keys = [("competition_score_native", "Score"), ("acc_entry_0.3s_native", "Entry acc@0.3s"),
            ("acc_collision_0.3s_native", "Collision acc@0.3s"), ("f1_entry_side_macro", "Side macro-F1"),
            ("f1_evasion_space_macro", "Evasion macro-F1")]
    for block in ("fixed_split", "cv"):
        if not r.get(block):
            continue
        print(f"\n**{block}** ({'mean ± std over seeds' if block == 'fixed_split' else 'mean ± std over folds×seeds'})\n")
        print("| Backbone | " + " | ".join(k[1] for k in keys) + " |")
        print("|---|" + "---|" * len(keys))
        for name, runs in r[block].items():
            print(f"| {name} | " + " | ".join(f"{np.mean([m[k] for m in runs]):.3f} ± {np.std([m[k] for m in runs]):.3f}" for k, _ in keys) + " |")
        if block == "cv" and len(r[block]) >= 2:
            names = list(r[block])
            a, b = r[block][names[0]], r[block][names[1]]
            d = np.array([mb["competition_score_native"] - ma["competition_score_native"] for ma, mb in zip(a, b)])
            print(f"\nPaired difference ({names[1]} − {names[0]}) over {len(d)} fold×seed runs: "
                  f"{d.mean():+.4f} ± {d.std():.4f} (wins {int((d > 0).sum())}/{len(d)})")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geometry")
    g.add_argument("--evals", nargs="+", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--results", required=True)
    args = parser.parse_args()
    geometry(args) if args.cmd == "geometry" else probe(args)


if __name__ == "__main__":
    main()
