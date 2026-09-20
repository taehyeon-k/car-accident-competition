"""Collect every finished run in results/ into one comparison table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ORDER = ["B0_pooled_baseline", "E1_dense", "E2a_dilated", "E2b_maxer", "E2c_attention",
         "E3_multirate", "E4_soft", "E5_state_aux", "E6_augmentation", "E7_consistency",
         "E8_event_conditioned"]

COLUMNS = ["run", "id", "entry_0.3s", "collision_0.3s", "side_f1", "evasion_f1", "score",
           "entry_norm_p50", "entry_norm_p90", "collision_norm_p50", "collision_norm_p90",
           "params", "wall_s"]


def rows_from(results: Path):
    found = []
    for name in ORDER + sorted({p.parent.name for p in results.glob("*/metrics.json")} - set(ORDER)):
        path = results / name / "metrics.json"
        if not path.exists() or any(r["run"] == name for r in found):
            continue
        m = json.loads(path.read_text())
        official, blind, runtime = m["official_offline"], m["fpsblind"], m["runtime"]
        cfg = json.loads((results / name / "config.json").read_text())["config"]
        found.append({
            "run": name, "id": cfg.get("id", name),
            "entry_0.3s": round(official["acc_entry_0.3s"], 4),
            "collision_0.3s": round(official["acc_collision_0.3s"], 4),
            "side_f1": round(official["f1_entry_side_macro"], 4),
            "evasion_f1": round(official["f1_evasion_space_macro"], 4),
            "score": round(official["competition_score"], 4),
            "entry_norm_p50": round(blind["entry_normalized_p50"], 4),
            "entry_norm_p90": round(blind["entry_normalized_p90"], 4),
            "collision_norm_p50": round(blind["collision_normalized_p50"], 4),
            "collision_norm_p90": round(blind["collision_normalized_p90"], 4),
            "params": runtime["parameter_count"],
            "wall_s": round(runtime["wall_seconds"], 1),
        })
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="stage2/experiments_v2/results")
    args = parser.parse_args()
    results = Path(args.results)
    rows = rows_from(results)
    if not rows:
        raise SystemExit(f"no finished runs under {results}")

    with (results / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    width = max(len(r["id"]) for r in rows)
    header = f"{'experiment':<{width}}  {'ENTRY':>6} {'COLL':>6} {'side':>6} {'evas':>6} {'SCORE':>7} {'params':>9}"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: -x["score"]):
        print(f"{r['id']:<{width}}  {r['entry_0.3s']:>6.2f} {r['collision_0.3s']:>6.2f} "
              f"{r['side_f1']:>6.3f} {r['evasion_f1']:>6.3f} {r['score']:>7.4f} {r['params']:>9,}")
    print(f"\nwrote {results / 'summary.csv'}")


if __name__ == "__main__":
    main()
