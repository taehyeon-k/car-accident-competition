"""Frame-rate robustness: evaluate each run's 5-fold CV checkpoints on their held-out clips re-sampled at 1/k of the native frame
rate (native frames[::k]; motion recomputed on the retained frames by extract_stride_motion.py). Frame numbers stay native, so the
official 0.3 s metric is unchanged. Plain direct decoding (no fusion) to measure the model itself. FPS is read only by the metric.
Usage: python -m stage2.generalization.robust_eval RUN [RUN ...] --strides 1 2 3 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments.decode_motion import decode  # noqa: F401  (kept for parity with other evaluators)
from stage2.spotting_experiments.objective import constrained_anchors
from stage2.aux_signal_experiments.model import load as load_model
from stage2.aux_signal_experiments.train import segments
from stage2.aux_signal_experiments.nexar_labels import motion28

ROOTS = [C.REPO / "stage2/aux_signal_experiments/results", C.REPO / "stage2/phase_study/results", C.REPO / "stage2/generalization/results"]
STRIDE = C.REPO / "stage2/generalization/cache_stride"
FOLDS = C.REPO / "stage2/long_context_v2_experiments/folds"


def root(run): return next(r for r in ROOTS if (r / run.split("+")[0]).is_dir())


def item(row, k, crop=0.0):
    sid = row["sample_id"]; frames_all, feats = C.dense(sid)
    if k == 1:
        kept = np.arange(len(frames_all)); mot = np.load(C.DENSE / f"{sid}.motion.npy")
        res = np.load(C.REPO / f"stage2/aux_signal_experiments/cache_residual/{sid}.residual.npy")
    else:
        d = np.load(STRIDE / f"k{k}" / f"{sid}.npz"); kept, mot, res = d["kept"], d["motion"], d["residual"]
    if crop:  # random window of crop*N retained frames that still contains ENTRY..COLLISION (seeded per clip)
        fr_k = frames_all[kept]; n = len(fr_k); L = max(int(round(crop * n)), 8)
        e = int(np.abs(fr_k - int(row["entry_frame"])).argmin()); c = int(np.abs(fr_k - int(row["collision_frame"])).argmin())
        L = max(L, c - e + 1); rng = np.random.default_rng(abs(hash(sid)) % (2 ** 32))
        lo, hi = max(0, c - L + 1), min(e, n - L)
        start = int(rng.integers(lo, hi + 1)) if hi >= lo else max(0, min(e, n - L))
        sl = slice(start, start + L); kept, mot, res = kept[sl], mot[sl].copy(), res[sl].copy(); mot[0] = 0; res[0] = 0
    reduced = frames_all[kept]; pos = C.select_adaptive(reduced)
    frames = reduced[pos]; x = torch.from_numpy(np.ascontiguousarray(feats[kept[pos]]))
    g = segments(motion28(mot), pos); r = segments(res, pos)
    ob = np.load(C.REPO / f"stage2/generalization/cache_objmotion/k{k}/{sid}.npy") if (C.REPO / f"stage2/generalization/cache_objmotion/k{k}/{sid}.npy").exists() else None
    if ob is not None and crop: ob = ob[sl]
    both = np.concatenate([g, r], 1)
    return {"sid": sid, "frames": frames, "x": x, "global": torch.from_numpy(g), "both": torch.from_numpy(both),
            "both_obj": torch.from_numpy(np.concatenate([both, segments(ob, pos)], 1)) if ob is not None else None,
            "n": len(reduced), "row": row}


@torch.inference_mode()
def predict(models, it, dev):
    v = torch.ones(1, len(it["frames"]), dtype=torch.bool, device=dev); x = it["x"][None].to(dev)
    outs = []
    for m, kind in models:
        mo = it[kind][None].to(dev) if kind else None
        outs.append(m(x, v, motion=mo) if m.uses_motion else m(x, v))
    pe = torch.stack([o["entry_logits"].float().softmax(-1) for o in outs]).mean(0).log()
    pc = torch.stack([o["collision_logits"].float().softmax(-1) for o in outs]).mean(0).log()
    side = float(torch.stack([o["side_logits"].float().softmax(-1)[0, 1] for o in outs]).mean())
    eva = float(torch.stack([o["evasion_logits"].float().sigmoid()[0] for o in outs]).mean())
    ei, ci = constrained_anchors(pe, pc); r = it["row"]; fr = it["frames"]
    return {"sample_id": it["sid"], "source_id": r["source_id"], "entry_frame": int(fr[int(ei[0])]), "collision_frame": int(fr[int(ci[0])]),
            "entry_side": int(side >= .5), "evasion_space": int(eva >= .5), "entry_gt": int(r["entry_frame"]), "collision_gt": int(r["collision_frame"]),
            "entry_side_gt": int(r["entry_side"] == "RIGHT"), "evasion_gt": int(r["evasion_space"]), "num_available_frames": it["n"]}


def evaluate(run, k, seeds, dev, per_seed=False, crop=0.0):
    """run may be 'A+B+C': an equal-weight ensemble of several families (all their given seeds)."""
    preds = []
    for f in range(5):
        cks = [root(r) / r / "cv" / f"fold{f}_seed{s}" / "checkpoint.pt" for r in run.split("+") for s in seeds]
        cks = [c for c in cks if c.exists()]
        models = []
        for c in cks:
            m = load_model(c, dev); cfg = torch.load(c, map_location="cpu", weights_only=False)["config"]
            kind = {"both": "both", "global": "global"}.get(cfg.get("motion", "none"), None)
            if cfg.get("objmotion"): kind = "both_obj"
            models.append((m, kind))
        if per_seed: models = models[:1]
        for r in C.rows(str(FOLDS / f"fold{f}_val.jsonl")): preds.append(predict(models, item(r, k, crop), dev))
    return C.breakdown(preds)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--crop", type=float, nargs="+", default=[0.0], help="0 = full clip; e.g. 0.5 = random half-length window with both events")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--out", default=None); a = ap.parse_args()
    dev = torch.device("cuda"); res = {}
    for run in a.runs:
        res[run] = {}
        for k in a.strides:
          for cr in a.crop:
            b = evaluate(run, k, a.seeds, dev, crop=cr); res[run][f"k{k}_crop{cr}"] = b; o = b["overall"]
            print(f"{run:28s} stride {k} crop {cr}: all {o['score']:.4f} (E {o['entry_acc']:.3f} C {o['collision_acc']:.3f} side {o['side_f1']:.3f} "
                  f"eva {o['evasion_f1']:.3f}) NEXAR {b['source:NEXAR']['score']:.4f} >1000 {b['bin:>1000']['score']:.4f}", flush=True)
    if a.out: C.dump(a.out, res)


if __name__ == "__main__": main()
