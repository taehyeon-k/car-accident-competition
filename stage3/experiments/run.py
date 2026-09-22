"""Run one Stage 3 v2 ablation through the standard trainer, then evaluate best.pt.

Example:
  python -m stage3.experiments.run --name A1_refinement \
      --set model.refinement.enabled=true
Writes runs/stage3_v2/<name>/{config.yaml,history.jsonl,best.pt,last.pt,metrics.json,
predictions.npz,environment.json}.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from sklearn.metrics import confusion_matrix

from stage3.trainer.decoder import ACCEL_LABELS, STEER_LABELS
from stage3.trainer.trainer import Trainer
from stage3.utils.config import load_config, seed_everything

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "stage3/configs/baseline_v1_2.workspace.yaml"


def set_path(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def git_commit() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "--", "stage3"], cwd=REPO).returncode != 0
        return commit + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--set", action="append", default=[], help="dotted.key=yaml_value")
    parser.add_argument("--root", default=str(REPO / "runs/stage3_v2"))
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.base).read_text())
    for item in args.set:
        key, _, value = item.partition("=")
        set_path(raw, key, yaml.safe_load(value))
    out = Path(args.root) / args.name
    raw["output_dir"] = str(out)
    raw.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists() and not args.eval_only:
        raise SystemExit(f"{out} already has best.pt; refusing to overwrite")
    (out / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    cfg = load_config(out / "config.yaml")
    seed_everything(cfg["seed"])
    (out / "environment.json").write_text(json.dumps({
        "git_commit": git_commit(), "seed": cfg["seed"], "overrides": args.set,
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_manifest": cfg["data"]["manifest"], "val_manifest": cfg["data"]["val_manifest"],
    }, indent=2))

    opt = cfg["optimization"]
    accelerator = Accelerator(gradient_accumulation_steps=opt["accumulation_steps"], mixed_precision=opt["mixed_precision"])
    trainer = Trainer(accelerator, cfg)
    trainer.build()
    model = accelerator.unwrap_model(trainer.model)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    if not args.eval_only:
        trainer.train_loop(None)
    train_seconds = time.time() - start
    train_peak = torch.cuda.max_memory_allocated() / 2**20

    # Final evaluation of the selected (best validation) EMA checkpoint.
    eval_trainer = Trainer(Accelerator(mixed_precision=opt["mixed_precision"]), copy.deepcopy(cfg))
    eval_trainer.build()
    eval_trainer.resume(str(out / "best.pt"))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    metrics = eval_trainer.validate()
    torch.cuda.synchronize()
    eval_seconds = time.time() - t0
    pa, ta, ps, ts, am, sm = map(np.asarray, eval_trainer.last_validation_predictions)
    sm = sm & (ta != "STOPPED")
    # Same full-clip EMA protocol on the training split (train/val gap without crop/augmentation bias).
    train_cfg = copy.deepcopy(cfg)
    train_cfg["data"]["val_manifest"] = cfg["data"]["manifest"]
    train_eval = Trainer(Accelerator(mixed_precision=opt["mixed_precision"]), train_cfg)
    train_eval.build()
    train_eval.resume(str(out / "best.pt"))
    train_full = train_eval.validate()
    history = [json.loads(x) for x in (out / "history.jsonl").read_text().splitlines() if x.strip()]
    best_epoch = max(history, key=lambda h: h.get("val/competition_score", -1))
    result = {
        "name": args.name, "overrides": args.set, "git_commit": git_commit(), "seed": cfg["seed"],
        "params": params, "trainable_params": trainable,
        "train_seconds": train_seconds, "train_peak_vram_mb": train_peak,
        "eval_seconds": eval_seconds, "eval_peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
        "val_clips": len(eval_trainer.val_set), "val_frames": int(len(ta)),
        "eval_ms_per_frame": 1000 * eval_seconds / max(len(ta), 1),
        "best_epoch": best_epoch["epoch"], "epochs_run": len(history),
        "train_at_best": {k[6:]: v for k, v in best_epoch.items() if k.startswith("train/")},
        "val": metrics,
        "train_full": train_full,
        "confusion_acceleration": {"labels": ACCEL_LABELS.tolist(),
                                   "matrix": confusion_matrix(ta[am], pa[am], labels=ACCEL_LABELS).tolist()},
        "confusion_steering": {"labels": STEER_LABELS.tolist(),
                               "matrix": confusion_matrix(ts[sm], ps[sm], labels=STEER_LABELS).tolist()},
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(out / "predictions.npz", pred_accel=pa, true_accel=ta, pred_steer=ps, true_steer=ts,
                        accel_valid=am, steer_valid=sm)
    tr = result["train_at_best"]
    print(json.dumps({"name": args.name, "train_crop_score": tr.get("competition_score"), "train_full_score": train_full["competition_score"], "val_score": metrics["competition_score"],
                      "val_acc": metrics["acceleration_macro_f1"], "val_steer": metrics["steering_macro_f1"],
                      "params": trainable, "best_epoch": result["best_epoch"]}))


if __name__ == "__main__":
    main()
