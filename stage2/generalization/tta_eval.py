"""H16 — frame-rate test-time augmentation. For each held-out clip, run the (seed/family) ensemble on several frame-rate views of the
same video (native frames[::k], motion recomputed on the retained frames; no FPS used) and average the ENTRY / COLLISION distributions
after projecting every view onto the base view's positions by frame number (nearest retained frame, renormalised). Attributes: mean.
Compares base view alone vs TTA over views. CV folds, plain constrained decoding (as robust_eval).
Usage: python -m stage2.generalization.tta_eval RUN [RUN ...] --base 1 --views 1 2 --seeds 0 1 2
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.spotting_experiments.objective import constrained_anchors
from stage2.aux_signal_experiments.model import load as load_model
from stage2.generalization.robust_eval import item, root, FOLDS


@torch.inference_mode()
def view_probs(models, it, dev):
    v = torch.ones(1, len(it["frames"]), dtype=torch.bool, device=dev); x = it["x"][None].to(dev)
    outs = [m(x, v, motion=it[kind][None].to(dev) if kind else None) if m.uses_motion else m(x, v) for m, kind in models]
    pe = torch.stack([o["entry_logits"].float().softmax(-1)[0] for o in outs]).mean(0).cpu().numpy()
    pc = torch.stack([o["collision_logits"].float().softmax(-1)[0] for o in outs]).mean(0).cpu().numpy()
    side = float(torch.stack([o["side_logits"].float().softmax(-1)[0, 1] for o in outs]).mean())
    eva = float(torch.stack([o["evasion_logits"].float().sigmoid()[0] for o in outs]).mean())
    return pe, pc, side, eva


def project(p, fr_src, fr_dst):
    idx = np.clip(np.searchsorted(fr_src, fr_dst), 0, len(fr_src) - 1); prev = np.clip(idx - 1, 0, len(fr_src) - 1)
    near = np.where(np.abs(fr_src[prev] - fr_dst) <= np.abs(fr_src[idx] - fr_dst), prev, idx)
    q = p[near]; return q / q.sum()


def decode(it, pe, pc, side, eva):
    ei, ci = constrained_anchors(torch.from_numpy(np.log(pe + 1e-12))[None], torch.from_numpy(np.log(pc + 1e-12))[None])
    r, fr = it["row"], it["frames"]
    return {"sample_id": it["sid"], "source_id": r["source_id"], "entry_frame": int(fr[int(ei[0])]), "collision_frame": int(fr[int(ci[0])]),
            "entry_side": int(side >= .5), "evasion_space": int(eva >= .5), "entry_gt": int(r["entry_frame"]), "collision_gt": int(r["collision_frame"]),
            "entry_side_gt": int(r["entry_side"] == "RIGHT"), "evasion_gt": int(r["evasion_space"]), "num_available_frames": it["n"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--base", type=int, default=1)
    ap.add_argument("--views", type=int, nargs="+", default=[1, 2]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args(); dev = torch.device("cuda")
    for run in a.runs:
        base, tta = [], []
        for f in range(5):
            cks = [root(r) / r / "cv" / f"fold{f}_seed{s}" / "checkpoint.pt" for r in run.split("+") for s in a.seeds]
            models = []
            for c in [c for c in cks if c.exists()]:
                cfg = torch.load(c, map_location="cpu", weights_only=False)["config"]
                models.append((load_model(c, dev), {"both": "both", "global": "global"}.get(cfg.get("motion", "none"))))
            for r in C.rows(str(FOLDS / f"fold{f}_val.jsonl")):
                its = {k: item(r, k) for k in sorted(set(a.views) | {a.base})}; b = its[a.base]
                out = {k: view_probs(models, it, dev) for k, it in its.items()}
                base.append(decode(b, *out[a.base]))
                pe = np.mean([project(out[k][0], its[k]["frames"], b["frames"]) for k in a.views], 0)
                pc = np.mean([project(out[k][1], its[k]["frames"], b["frames"]) for k in a.views], 0)
                tta.append(decode(b, pe, pc, np.mean([out[k][2] for k in a.views]), np.mean([out[k][3] for k in a.views])))
        for name, preds in (("base", base), (f"tta{a.views}", tta)):
            bd = C.breakdown(preds); o = bd["overall"]
            print(f"{run:30s} base k{a.base} {name:12s}: all {o['score']:.4f} (E {o['entry_acc']:.3f} C {o['collision_acc']:.3f} side {o['side_f1']:.3f} "
                  f"eva {o['evasion_f1']:.3f}) " + " ".join(f"{s} {bd['source:' + s]['score']:.3f}" for s in ("AIHUB", "CCD", "MMAU", "NEXAR")), flush=True)


if __name__ == "__main__": main()
