"""Parity + held-out check of the packaged v6 Stage 2 runtime on the fixed 70-clip val split, from raw frames.

1. residual motion computed by the runtime == stage2/aux_signal_experiments/cache_residual (training input)
2. per-member ENTRY / COLLISION logits == each member's training-side val predictions.json (same fixed-split checkpoint)
3. end-to-end package predictions scored with the offline competition metric (the val split was not used for training)
Usage: python submission_tools/v6_stage2/check_v6_parity.py <submission dir> [--limit N]
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from stage2.long_context_v2_experiments import common as C

ap = argparse.ArgumentParser(); ap.add_argument("submission"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=str(Path(__file__).with_name("parity_results.json"))); a = ap.parse_args()
s2 = Path(a.submission) / "model" / "stage2"
spec = importlib.util.spec_from_file_location("v6_runtime", s2 / "runtime.py"); R = importlib.util.module_from_spec(spec)
sys.modules["v6_runtime"] = R; spec.loader.exec_module(R)

device = torch.device("cuda")
config, backbone, head = R.load_models(s2, device)
rows = C.rows("val")
if a.limit: rows = rows[:a.limit]
ref = {}
for m in config["members"]:
    src = Path(REPO / m["source"]).parent / "predictions.json"
    ref[m["file"]] = {p["sample_id"]: p for p in json.loads(src.read_text())}
gt = ref[next(m["file"] for m in config["members"] if m["run"] == "E4_NT_both")]
cache_res = REPO / "stage2/aux_signal_experiments/cache_residual"

res_diff, logit_diff, argmax_agree, frames_ok, preds, t0 = [], [], [], [], [], time.time()
with torch.inference_mode():
    for r in rows:
        folder = Path(r["frames_dir"]); pairs = R.indexed_frames(folder)
        frame_numbers, chosen, _, x, valid, inputs, shift = R.prepare(pairs, config, backbone, head, device)
        # 1. residual input parity (raw per-frame, before segments)
        _, residual = R.motion_features([p for _, p in pairs], True, 4)
        res_diff.append(float(np.abs(residual - np.load(cache_res / f"{r['sample_id']}.residual.npy")).max()))
        # 2. per-member logits
        parts = head.members(x, valid, inputs)
        for m, o in zip(config["members"], parts):
            p = ref[m["file"]][r["sample_id"]]
            frames_ok.append(p["frames"] == frame_numbers[chosen].tolist())
            for e in ("entry", "collision"):
                mine = o[f"{e}_logits"][0].float().cpu().numpy(); theirs = np.asarray(p[f"{e}_logits"], np.float32)
                logit_diff.append(float(np.abs(mine - theirs).max())); argmax_agree.append(int(mine.argmax() == theirs.argmax()))
        # 3. end-to-end prediction through the packaged decoder
        row, _, _ = R.predict_folder(folder, config, backbone, head, device)
        g = gt[r["sample_id"]]
        preds.append({**{k: g[k] for k in ("sample_id", "source_id", "entry_gt", "collision_gt", "entry_side_gt", "evasion_gt",
                                           "num_available_frames")},
                      "entry_frame": row["entry_frame"], "collision_frame": row["collision_frame"],
                      "entry_side": int(row["entry_side"] == "RIGHT"), "evasion_space": row["evasion_space"]})
        print(f"{len(preds):3d} {r['sample_id']:45s} n={len(pairs):5d} res {res_diff[-1]:.2e} "
              f"logit {max(logit_diff[-2 * len(parts):]):.2e}", flush=True)

b = C.breakdown(preds)
out = {"clips": len(preds), "seconds": time.time() - t0,
       "residual_max_abs_diff": max(res_diff), "residual_exact_clips": int(sum(d == 0 for d in res_diff)),
       "member_logit_max_abs_diff": max(logit_diff), "member_logit_median_max_abs_diff": float(np.median(logit_diff)),
       "member_argmax_agreement": float(np.mean(argmax_agree)), "sampled_frames_identical": bool(all(frames_ok)),
       "val_overall": b["overall"], "val_NEXAR": b.get("source:NEXAR", {}), "val_>1000": b.get("bin:>1000", {})}
Path(a.out).write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if not k.startswith("val_")}, indent=1))
for k in ("val_overall", "val_NEXAR", "val_>1000"):
    print(k, {m: round(out[k][m], 4) for m in ("score", "entry_acc", "collision_acc", "side_f1", "evasion_f1") if m in out[k]})
