"""Metadata-anchored labels for the unlabelled AIHUB / CCD / MMAU pool (cache_unl), one label set per CV fold (+ 'all' for refits).

  MMAU  COLLISION = MM-AU t_co + offset (frames; offset = median over the labelled MMAU clips of the split's TRAIN part; = -1 in practice,
                    99 % of labelled clips within 0.3 s). ENTRY = teacher argmax within [t_ai + e_off - W, COLLISION] where e_off is the
                    split's median (GT ENTRY - t_ai) and W = 0.5 s (t_ai + offset alone is within 0.3 s for 69 %).
  CCD   COLLISION = CCD first-accident frame + offset (split median; 81 % within 0.3 s). ENTRY = teacher argmax within 2 s before COLLISION.
  AIHUB (no timing metadata) COLLISION and ENTRY from the teacher (optional, --aihub).
Teacher for split fold k = the E4_sa fold-k CV checkpoints, seeds 0-2 (never trained on fold k). FPS is used only here, for the
seconds-valued windows (label construction, like nexar_labels.py); models never see it.
Output cache_unl_labels/labels_<split>.json {sid: {collision_frame, entry_frame, source}}.
"""
from __future__ import annotations

import argparse, csv, io, json, subprocess
from pathlib import Path

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.aux_signal_experiments.model import load as load_model
from stage2.aux_signal_experiments.train import unl_view

UNL = C.REPO / "stage2/generalization/cache_unl"
OUT = C.REPO / "stage2/generalization/cache_unl_labels"
META = Path("/workspace/data/stage2/meta")
FOLDS = C.REPO / "stage2/long_context_v2_experiments/folds"
GEN = C.REPO / "stage2/generalization/results"


def metadata():
    mm = {m["source_id"]: m for m in map(json.loads, open(META / "normalized.jsonl"))}
    ccd = {r["video_id"]: r for r in csv.DictReader(open(META / "ccd_candidates.csv"))}
    unus = {r["sample_id"]: r for r in csv.DictReader(io.StringIO(subprocess.run(
        ["rclone", "cat", "r2:car-accident-dataset/stage2/unusable/unusable_only.csv"], capture_output=True, text=True).stdout))}
    return mm, ccd, unus


def anchors(sid, mm, ccd):
    """(source, metadata collision frame, metadata entry anchor or None)"""
    if sid.startswith("mmauu_"):
        m = mm[sid.split("_")[1]]; return "MMAU", int(m["metadata_collision_frame_candidate"]), int(m["metadata_accident_start"])
    if sid.startswith("ccd_"):
        v = ccd.get(sid.split("_")[1])
        if v and v["ccd_first_accident_frame_0based"]: return "CCD", int(v["ccd_first_accident_frame_0based"]), None
        return None
    if sid.startswith("aihub_"): return "AIHUB", None, None
    return None


def offsets(train_rows, mm, ccd):
    c_mm, e_mm, c_ccd = [], [], []
    for r in train_rows:
        s = C.source(r)
        if s == "MMAU":
            m = mm[r["sample_id"].split("_")[1]]
            c_mm.append(int(r["collision_frame"]) - int(m["metadata_collision_frame_candidate"]))
            e_mm.append(int(r["entry_frame"]) - int(m["metadata_accident_start"]))
        elif s == "CCD":
            v = ccd[r["sample_id"].split("_")[1]]; c_ccd.append(int(r["collision_frame"]) - int(v["ccd_first_accident_frame_0based"]))
    return int(np.median(c_mm)), int(np.median(e_mm)), int(np.median(c_ccd))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--aihub", action="store_true")
    ap.add_argument("--full", nargs="*", default=None, help="'all' split: teacher = these full-data runs (dirs with checkpoint.pt)")
    a = ap.parse_args()
    dev = torch.device("cuda"); mm, ccd, unus = metadata(); OUT.mkdir(exist_ok=True)
    sids = sorted(p.name[:-len(".motion.npy")] for p in UNL.glob("*.motion.npy"))
    splits = {f"fold{k}": (C.rows(str(FOLDS / f"fold{k}_train.jsonl")), sorted(GEN.glob(f"E4_sa/cv/fold{k}_seed[012]/checkpoint.pt"))) for k in range(5)}
    if a.full is not None: splits = {"all": (C.rows("all"), [Path(p) / "checkpoint.pt" for p in a.full])}
    for name, (train_rows, cks) in splits.items():
        assert cks, name
        c_mm, e_mm, c_ccd = offsets(train_rows, mm, ccd)
        models = [load_model(p, dev) for p in cks]; labels = {}
        for sid in sids:
            an = anchors(sid, mm, ccd)
            if an is None or (an[0] == "AIHUB" and not a.aihub): continue
            src, cmeta, emeta = an
            fps = 30.0 if src == "MMAU" else float(unus[sid]["fps"])
            it = unl_view(sid, 1); fr = it["frame_numbers"].numpy()
            x = torch.from_numpy(np.ascontiguousarray(np.load(it["x_path"], mmap_mode="r")[it["x_idx"]]))[None].to(dev)
            v = torch.ones(1, len(fr), dtype=torch.bool, device=dev); mot = it["motion"][None].to(dev)
            outs = [m(x, v, motion=mot) for m in models]
            pe = torch.stack([o["entry_logits"][0].float().softmax(-1) for o in outs]).mean(0).cpu().numpy()
            pc = torch.stack([o["collision_logits"][0].float().softmax(-1) for o in outs]).mean(0).cpu().numpy()
            if src == "MMAU": c = cmeta + c_mm
            elif src == "CCD": c = cmeta + c_ccd
            else: c = int(fr[pc.argmax()])
            c = int(min(max(c, fr[1]), fr[-1]))
            lo = (emeta + e_mm - 0.5 * fps) if src == "MMAU" else c - 2.0 * fps
            ok = (fr <= c) & (fr >= lo)
            e = int(fr[np.argmax(np.where(ok, pe, -1))]) if ok.any() else int(fr[np.argmax(np.where(fr <= c, pe, -1))])
            labels[sid] = {"collision_frame": c, "entry_frame": min(e, c), "source": src}
        C.dump(OUT / f"labels_{name}.json", labels)
        from collections import Counter
        print(name, f"offsets mmau C {c_mm} E {e_mm} ccd C {c_ccd}", "teachers", len(models), Counter(l["source"] for l in labels.values()), flush=True)


if __name__ == "__main__": main()
