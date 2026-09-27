"""COLLISION labels in the organisers' apparent convention (hypothesis): the public source annotation of first contact.

Evidence: the five DACON Stage 2 sample videos are CCD clips 000001-000005 and their t_collision equals CCD's
first-accident frame (0-based) exactly; our manual labels on the three we labelled are +1 frame. This builds fold manifests where
COLLISION = CCD first-accident frame / MM-AU t_co / NEXAR round(time_of_event * fps) when available (AIHUB and clips without
metadata keep the manual label). ENTRY, side and evasion are unchanged. FPS is used only to convert NEXAR seconds to a frame index
(label construction). Clips where ENTRY would fall after the official COLLISION keep the manual COLLISION.
Output stage2/generalization/folds_official/{fold{k}_train,fold{k}_val,all}.jsonl
"""
from __future__ import annotations

import csv, io, json, subprocess
from pathlib import Path

from stage2.long_context_v2_experiments import common as C

OUT = C.REPO / "stage2/generalization/folds_official"
FOLDS = C.REPO / "stage2/long_context_v2_experiments/folds"
META = Path("/workspace/data/stage2/meta")


def official_map():
    ccd = {r["video_id"]: int(r["ccd_first_accident_frame_0based"]) for r in csv.DictReader(open(META / "ccd_candidates.csv"))
           if r["ccd_first_accident_frame_0based"]}
    mm = {m["source_id"]: int(m["metadata_collision_frame_candidate"]) for m in map(json.loads, open(META / "normalized.jsonl"))}
    nx = {r["video_id"]: float(r["time_of_event"]) for r in csv.DictReader(io.StringIO(subprocess.run(
        ["rclone", "cat", "r2:car-accident-dataset/stage2/manifests/nexar_candidates.csv"], capture_output=True, text=True).stdout))
          if r.get("time_of_event")}
    return ccd, mm, nx


def convert(row, maps):
    ccd, mm, nx = maps; s = C.source(row); key = row["sample_id"].split("_", 1)[1]; c = None
    if s == "CCD": c = ccd.get(key)
    elif s == "MMAU": c = mm.get(key)
    elif s == "NEXAR" and key in nx: c = int(round(nx[key] * float(row["native_fps"])))
    out = dict(row)
    if c is not None and int(row["entry_frame"]) <= c < int(row["num_frames"]): out["collision_frame"] = c
    return out


def main():
    maps = official_map(); OUT.mkdir(exist_ok=True); changed = {}
    for name in [f"fold{k}_{p}" for k in range(5) for p in ("train", "val")]:
        rows = [convert(r, maps) for r in C.rows(str(FOLDS / f"{name}.jsonl"))]
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rows = [convert(r, maps) for r in C.rows("all")]
    for r0, r in zip(C.rows("all"), rows):
        if r0["collision_frame"] != r["collision_frame"]: changed.setdefault(C.source(r), []).append(int(r["collision_frame"]) - int(r0["collision_frame"]))
    (OUT / "all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for s, d in changed.items(): print(s, "changed", len(d), "mean shift %.2f frames" % (sum(d) / len(d)))


if __name__ == "__main__": main()
