"""Nested outer-fold screen of four full refits versus four fold-trained heads.

No outer-fold label is used for training, selection, or stopping. This is a
deployment-method experiment, not an existing-checkpoint ensemble screen.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.generalization import robust_eval as R
from stage2.generalization.clean_eval import EXCL
from stage2.aux_signal_experiments.model import load

ROOT = C.REPO / "stage2/robust_goal_20260927/crossfit_deploy"
FOLDS = C.REPO / "stage2/robust_goal_20260927/group_folds"


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    outer = C.rows(str(FOLDS / "fold0_train.jsonl"))
    ids = {row["sample_id"] for row in outer}
    assert len(ids) == len(outer) == 276
    seen = set()
    for inner in range(1, 5):
        val = C.rows(str(FOLDS / f"fold{inner}_val.jsonl"))
        val_ids = {row["sample_id"] for row in val}
        assert val_ids <= ids and not (seen & val_ids)
        seen |= val_ids
        train = [row for row in outer if row["sample_id"] not in val_ids]
        for tag, rows in (("train", train), ("val", val)):
            path = ROOT / f"inner{inner}_{tag}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert seen == ids
    (ROOT / "manifest.json").write_text(json.dumps({
        "hypothesis": "Fold-trained selected heads may transfer better than fixed-epoch full refits despite fewer training clips per head.",
        "outer_fold": 0, "outer_val_labels_used_for_training_or_selection": False,
        "inner_folds": [1, 2, 3, 4], "members_each": 4,
        "full_refit_epochs": 7, "stride_aug": "0.5,0.25,0.25",
        "control": "Four seed-matched full refits on the same 276 outer-train clips, each fixed at v8's epoch7.",
        "candidate": "Four inner-fold heads, each trained on three quarters of the 276 outer-train clips and selected only on its inner holdout.",
        "promotion_gate": "Duplicate-clean outer score gain >=0.01 at both native and third rate; no source score loss >0.02. Confirm on all outer folds before a package.",
    }, indent=2) + "\n")


def occupied():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return True


def train():
    prepare()
    jobs = []
    for kind in ("full", "inner"):
        for seed in range(4):
            split = seed + 1
            output = ROOT / kind / f"seed{seed}"
            if (output / "checkpoint.pt").exists():
                continue
            output.mkdir(parents=True, exist_ok=True)
            args = [sys.executable, "-m", "stage2.aux_signal_experiments.train", "--run-id", "CROSSFIT_" + kind,
                    "--seed", str(seed), "--motion", "both", "--stride-aug", "0.5,0.25,0.25",
                    "--output", str(output)]
            if kind == "full":
                args += ["--train-split", str(FOLDS / "fold0_train.jsonl"),
                         "--val-split", str(FOLDS / "fold0_val.jsonl"), "--stop-epoch", "7"]
            else:
                args += ["--train-split", str(ROOT / f"inner{split}_train.jsonl"),
                         "--val-split", str(ROOT / f"inner{split}_val.jsonl")]
            jobs.append((kind, seed, output, args))
    for kind, seed, output, args in jobs:
        while occupied():
            print("GPU busy; waiting without starting a new Stage2 job", flush=True)
            time.sleep(15)
        print("TRAIN", kind, seed, flush=True)
        with (output / "train.log").open("w") as log:
            result = subprocess.run(args, cwd=C.REPO, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
        if result.returncode:
            raise RuntimeError(f"Training {kind} seed{seed} failed; see {output / 'train.log'}")


@torch.inference_mode()
def evaluate():
    models = {}
    for kind in ("full", "inner"):
        paths = [ROOT / kind / f"seed{s}" / "checkpoint.pt" for s in range(4)]
        if any(not path.exists() for path in paths):
            raise FileNotFoundError(f"Missing {kind} checkpoints")
        models[kind] = [(load(path, "cuda"), "both") for path in paths]
    rows = [r for r in C.rows(str(FOLDS / "fold0_val.jsonl")) if r["sample_id"] not in EXCL]
    assert not ({r["sample_id"] for r in rows} & {r["sample_id"] for r in C.rows(str(FOLDS / "fold0_train.jsonl"))})
    fps = C.fps_table()
    result = {kind: {} for kind in models}
    for stride in (1, 2, 3):
        predictions = {kind: [] for kind in models}
        for row in rows:
            sample = R.item(row, stride)
            for kind, members in models.items():
                predictions[kind].append(R.predict(members, sample, torch.device("cuda")))
        for kind, preds in predictions.items():
            metrics = C.metrics(preds)
            by_source = {}
            for source in ("AIHUB", "CCD", "MMAU", "NEXAR"):
                subset = [p for p in preds if C.source(p) == source]
                by_source[source] = C.metrics(subset)["score"] if subset else None
            result[kind][f"k{stride}"] = {name: metrics[name] for name in ("score", "entry_acc", "collision_acc", "side_f1", "evasion_f1")}
            result[kind][f"k{stride}"]["n"] = len(preds)
            result[kind][f"k{stride}"]["source_score"] = by_source
            (ROOT / f"{kind}_k{stride}_predictions.json").write_text(json.dumps(preds) + "\n")
    (ROOT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "evaluate"))
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate}[args.action]()
