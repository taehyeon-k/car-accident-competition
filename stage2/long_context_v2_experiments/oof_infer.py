"""Re-run every CV checkpoint of a run on its held-out fold, saving event logits AND side/evasion probabilities."""
from __future__ import annotations

import argparse, json

import torch

from . import common as C
from .ensemble import load

R = C.REPO / "stage2/long_context_v2_experiments/results"


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); a = ap.parse_args()
    dev = torch.device("cuda"); torch.set_num_threads(4)
    for run in a.runs:
        for ck in sorted((R / run / "cv").glob("fold*_seed*/checkpoint.pt")):
            out = ck.parent / "oof_full.json"
            if out.exists(): continue
            k = int(ck.parent.name[4]); model = load(ck, dev)
            rows = C.rows(str(C.REPO / f"stage2/long_context_v2_experiments/folds/fold{k}_val.jsonl"))
            res = []
            for r in rows:
                it = C.make_item(r, "adaptive"); b = C.collate_m([it])
                x, v = b["x"].to(dev), b["time_valid"].to(dev)
                o = model(x, v, motion=b["motion"].to(dev)) if getattr(model, "uses_motion", False) else model(x, v)
                res.append({"sample_id": r["sample_id"], "frames": it["frame_numbers"].tolist(),
                            "entry_logits": o["entry_logits"][0].float().cpu().tolist(), "collision_logits": o["collision_logits"][0].float().cpu().tolist(),
                            "side_prob_right": float(o["side_logits"][0].float().softmax(-1)[1]), "evasion_prob": float(o["evasion_logits"][0].float().sigmoid())})
            C.dump(out, res); print(run, ck.parent.name, flush=True)


if __name__ == "__main__": main()
