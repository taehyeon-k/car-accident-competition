"""Frame-rate views (native / 1/2 / 1/3) for the v5 and v6 recipes, to test whether the ~10 fps validation view ranks them as the
leaderboard did (v5 0.5314 > v6 0.5277). CV proxies: 5-fold CV checkpoints, seeds 0-1 of every family. Plain decoding (probability
average + constrained anchors), same as robust_eval, so the numbers are comparable to the v7/v8 views.
  v5: C0_avg + X_ema + M_motion + PH_phase (LCPyramid, long_context_v2)
  v6: ENTRY from E2_NT_both_bnd2 + M1_both + PH_phase + E4_NT_both + M_motion; COLLISION from E2_NT_both_bnd2;
      attributes from all 7 families (C0_avg, X_ema, M_motion, PH_phase, M1_both, E4_NT_both, E2_NT_both_bnd2)
"""
from __future__ import annotations

import torch

from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments.ensemble import load as load_lc
from stage2.aux_signal_experiments.model import load as load_aux
from stage2.spotting_experiments.objective import constrained_anchors
from .robust_eval import item, FOLDS

LC = C.REPO / "stage2/long_context_v2_experiments/results"
AUX = C.REPO / "stage2/aux_signal_experiments/results"
RECIPES = {
    "v5": dict(entry=["C0_avg", "X_ema", "M_motion", "PH_phase"], coll=["C0_avg", "X_ema", "M_motion", "PH_phase"],
               attr=["C0_avg", "X_ema", "M_motion", "PH_phase"]),
    "v6": dict(entry=["E2_NT_both_bnd2", "M1_both", "PH_phase", "E4_NT_both", "M_motion"], coll=["E2_NT_both_bnd2"],
               attr=["C0_avg", "X_ema", "M_motion", "PH_phase", "M1_both", "E4_NT_both", "E2_NT_both_bnd2"]),
}


def load_family(fam, fold, dev, seeds=(0, 1)):
    out = []
    for s in seeds:
        if (LC / fam).is_dir():
            p = LC / fam / "cv" / f"fold{fold}_seed{s}" / "checkpoint.pt"; cfg = torch.load(p, map_location="cpu", weights_only=False)["config"]
            out.append((load_lc(p, dev), "global" if cfg.get("motion") else None))
        else:
            p = AUX / fam / "cv" / f"fold{fold}_seed{s}" / "checkpoint.pt"; cfg = torch.load(p, map_location="cpu", weights_only=False)["config"]
            out.append((load_aux(p, dev), {"both": "both", "global": "global"}.get(cfg.get("motion", "none"))))
    return out


@torch.inference_mode()
def run(recipe, k, dev):
    spec = RECIPES[recipe]; fams = sorted(set(spec["entry"] + spec["coll"] + spec["attr"])); preds = []
    for f in range(5):
        models = {fam: load_family(fam, f, dev) for fam in fams}
        for r in C.rows(str(FOLDS / f"fold{f}_val.jsonl")):
            it = item(r, k); v = torch.ones(1, len(it["frames"]), dtype=torch.bool, device=dev); x = it["x"][None].to(dev)
            outs = {fam: [(m(x, v, motion=it[kind][None].to(dev)) if kind else m(x, v)) for m, kind in ms] for fam, ms in models.items()}
            avg = lambda key, fs, fn: torch.stack([fn(o[key].float()) for fam in fs for o in outs[fam]]).mean(0)
            pe = avg("entry_logits", spec["entry"], lambda z: z.softmax(-1)).log()
            pc = avg("collision_logits", spec["coll"], lambda z: z.softmax(-1)).log()
            side = float(avg("side_logits", spec["attr"], lambda z: z.softmax(-1)[0, 1]))
            eva = float(avg("evasion_logits", spec["attr"], lambda z: z.sigmoid().reshape(-1)[0]))
            ei, ci = constrained_anchors(pe, pc); fr = it["frames"]
            preds.append({"sample_id": it["sid"], "source_id": r["source_id"], "entry_frame": int(fr[int(ei[0])]), "collision_frame": int(fr[int(ci[0])]),
                          "entry_side": int(side >= .5), "evasion_space": int(eva >= .5), "entry_gt": int(r["entry_frame"]),
                          "collision_gt": int(r["collision_frame"]), "entry_side_gt": int(r["entry_side"] == "RIGHT"),
                          "evasion_gt": int(r["evasion_space"]), "num_available_frames": it["n_native"]})
    return C.breakdown(preds)


def main():
    dev = torch.device("cuda"); res = {}
    for rec in ("v5", "v6"):
        for k in (1, 2, 3):
            b = run(rec, k, dev); res[f"{rec}_k{k}"] = b
            print(rec, f"k{k}", C.short_table(b), flush=True)
    C.dump(C.REPO / "stage2/generalization/results/robust_v5v6.json", res)


if __name__ == "__main__": main()
