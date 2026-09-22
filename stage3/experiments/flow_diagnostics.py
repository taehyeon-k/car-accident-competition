"""Compare flow backends on cached motion/physics (Stage 3 v2, Experiment H).

For each validation clip in two manifests (same clip ids, different flow caches)
compute per-GT-class statistics: dequantized derotated flow magnitude (masked by
the static-validity channel), flow confidence, rho(k=2/4) physics estimates,
FOE temporal variability, rotation residual. E_static = mean |flow| on frames
with GT speed <= 0.15 m/s; Var(rho) on frames with |a| < 0.1 m/s^2.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from stage3.data.adapters.base import Signals
from stage3.data.cache import dequantize_motion
from stage3.data.targets import make_targets
from stage3.utils.checkpoint import load_artifact
from stage3.utils.config import load_config, read_jsonl

CFG = None


def clip_stats(row):
    cache = load_artifact(row["cache_path"])
    signals = Signals(**{k: None if v is None else v.numpy() for k, v in cache["signals"].items()})
    t = make_targets(signals, cache["actual_times"].numpy(), CFG["targets"])
    motion = dequantize_motion(cache).numpy()
    valid = motion[:, 7] > 0.5
    magnitude = np.hypot(motion[:, 0], motion[:, 1])
    frame_mag = np.array([m[v].mean() if v.any() else np.nan for m, v in zip(magnitude, valid)])
    frame_conf = motion[:, 6].mean((1, 2))
    physics = cache["physics"].numpy()
    a = 0.5 * (t["a_long_s1"] + t["a_long_s2"])
    speed = t["speed"]
    label = np.where(t["stopped"] > 0.5, "STOPPED", np.where(a > 0.25, "ACCELERATING", np.where(a < -0.25, "DECELERATING", "CONSTANT")))
    ok = cache["time_valid"].numpy().astype(bool) & np.isfinite(speed) & (np.isfinite(a) | (t["stopped"] > 0.5))
    foe = np.asarray(cache["metadata"].get("foe", np.zeros((len(ok), 2))), np.float32)
    return dict(label=label[ok], mag=frame_mag[ok], conf=frame_conf[ok], rho2=physics[ok, 5], rho4=physics[ok, 6],
                rot_res=physics[ok, 4], speed=speed[ok], a=a[ok], physics=physics[ok],
                foe_step=np.r_[np.nan, np.linalg.norm(np.diff(foe, axis=0), axis=1)][ok])


def summarize(parts):
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out = {}
    for cls in ["STOPPED", "CONSTANT", "ACCELERATING", "DECELERATING"]:
        m = cat["label"] == cls
        out[cls] = {"frames": int(m.sum()), "flow_mag_mean": float(np.nanmean(cat["mag"][m])), "flow_mag_var": float(np.nanvar(cat["mag"][m])),
                    "confidence_mean": float(np.nanmean(cat["conf"][m])), "rotation_residual_mean": float(np.nanmean(cat["rot_res"][m])),
                    "foe_step_px_median": float(np.nanmedian(cat["foe_step"][m]))}
    static = cat["speed"] <= 0.15
    steady = np.abs(cat["a"]) < 0.1
    out["E_static"] = float(np.nanmean(cat["mag"][static]))
    out["var_rho_lag2_steady"] = float(np.nanvar(cat["rho2"][steady & ~static]))
    out["var_rho_lag4_steady"] = float(np.nanvar(cat["rho4"][steady & ~static]))
    out["physics_feature_std"] = np.nanstd(cat["physics"], 0).round(4).tolist()
    out["corr_rho2_vs_a_over_v"] = float(np.corrcoef(cat["rho2"][~static & np.isfinite(cat["a"])],
                                                     (cat["a"] / np.maximum(cat["speed"], 0.5))[~static & np.isfinite(cat["a"])])[0, 1])
    return out


def main():
    global CFG
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="stage3/configs/baseline_v1_2.workspace.yaml")
    p.add_argument("--manifest", action="append", required=True, help="name=path")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    CFG = load_config(args.config)
    result = {}
    for item in args.manifest:
        name, path = item.split("=", 1)
        with Pool(12) as pool:
            result[name] = summarize(pool.map(clip_stats, read_jsonl(path), chunksize=4))
        print(name, json.dumps({k: v for k, v in result[name].items() if k != "physics_feature_std"}, indent=1))
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
