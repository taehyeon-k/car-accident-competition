"""Frozen recipe on the fixed 279/70 split: 3-run seed ensemble + motion fusion (beta fixed) + native snap
+ event-conditioned attribute head. Nothing tuned on the 70 val clips."""
from __future__ import annotations

import argparse
import numpy as np
import torch

from . import common as C
from .attr_head import train as train_attr, window_item
from .decode_motion import decode
from .ensemble import load

R = C.REPO / "stage2/long_context_v2_experiments/results"


@torch.inference_mode()
def infer(models, rows, dev):
    out = []
    for r in rows:
        it = C.make_item(r, "adaptive"); b = C.collate_m([it]); x, v, m = b["x"].to(dev), b["time_valid"].to(dev), b["motion"].to(dev)
        o = C.ProbEnsemble(models)(x, v, motion=m)
        p = {k: it[k] for k in ("sample_id", "source_id", "num_available_frames")}
        p.update(frames=it["frame_numbers"].tolist(), entry_logp=o["entry_logits"][0].float().cpu().numpy(),
                 collision_logp=o["collision_logits"][0].float().cpu().numpy(),
                 entry_side=int(o["side_logits"][0].argmax()), evasion_space=int(o["evasion_logits"][0] >= 0),
                 entry_gt=int(r["entry_frame"]), collision_gt=int(r["collision_frame"]),
                 entry_side_gt=int(r["entry_side"] == "RIGHT"), evasion_gt=int(r["evasion_space"]))
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", default=["C0_avg", "X_ema", "M_motion"])
    ap.add_argument("--beta", type=float, default=1.0); a = ap.parse_args()
    dev = torch.device("cuda"); torch.set_num_threads(4)
    models = [load(p, dev) for r in a.runs for p in sorted((R / r).glob("seed*/checkpoint.pt"))]
    val = C.rows("val"); raw = infer(models, val, dev)
    plain = [decode(p, None, 0, 0, 0.0, False, False) for p in raw]
    fused = [decode(p, None, 0, 0, a.beta, False, True) for p in raw]
    heads = [train_attr(C.rows("train"), s, 15, dev) for s in (0, 1, 2)]
    ev = []
    with torch.no_grad():
        for q in fused:
            x = torch.from_numpy(window_item(q["sample_id"], q["entry_frame"], q["collision_frame"]))[None].to(dev)
            ps = np.mean([float(h(x)[0].softmax(-1)[0, 1]) for h in heads]); pe = np.mean([float(h(x)[1].sigmoid()[0]) for h in heads])
            ev.append({**q, "entry_side": int(ps >= .5), "evasion_space": int(pe >= .5)})
    res = {}
    for name, ps in (("ensemble, plain decode", plain), ("+ motion fusion + snap", fused), ("+ event-conditioned attributes", ev)):
        b = C.breakdown(ps); res[name] = b; n = b["source:NEXAR"]
        print(f"{name:32s} all {b['overall']['score']:.3f} | NEXAR {n['score']:.3f} E {n['entry_acc']:.2f} C {n['collision_acc']:.2f} "
              f"side {n['side_f1']:.2f} eva {n['evasion_f1']:.2f} | non-NEXAR {b['source:non-NEXAR']['score']:.3f} | >1000 {b['bin:>1000']['score']:.3f}")
    C.dump(R / f"fixed_eval_recipe_{'+'.join(a.runs)}.json", {"members": len(models), "runs": a.runs, "beta": a.beta, **res})


if __name__ == "__main__": main()
