"""Reproduce the submitted leaderboard baseline: geometry-DINOv3-S + simple probe.

This is the architecture and protocol behind ``submission_tools/train_fps_stage2.py``
(real leaderboard ~0.41, held-out selection ~0.61). It is reproduced here rather
than invoked so the run cannot overwrite the recorded submission artifacts in
``submission_tools/fps_stage2/``. The model and loss are imported from their
original modules, not copied, so this stays a faithful reference point.

Only the held-out selection phase is run. The submitted script then refits on all
251 clips for submission; that refit trains on the validation clips and so has no
independent score to compare against the v2 matrix.

Unlike ``experiments_v2`` proper, this baseline is deliberately NOT FPS-blind:
the cache stride is ``round(native_fps/10)`` and the Gaussian loss widths are in
seconds, exactly as submitted. That is the property the v2 work is testing.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submission_tools/fps_stage2"))

from probe import TemporalProbe  # noqa: E402  (original submitted architecture)
from sampling import normalized_indices  # noqa: E402

from stage2.utils.joint_losses import constrained_decode, joint_loss  # noqa: E402
from stage2.utils.joint_metrics import JointMetricAccumulator, joint_metric_packet  # noqa: E402

FEATURES = "/workspace/cache/geometry_pretrain/stage2_probe_features/adapted_noanchor"


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


class Features(Dataset):
    """Exactly the submitted dataset: ~10 Hz stride cache, seconds-valued targets."""

    def __init__(self, rows, train=False, feature_dir=FEATURES):
        self.rows, self.train = rows, train
        self.arrays = []
        for row in rows:
            array = np.load(f"{feature_dir}/{row['sample_id']}.npy", mmap_mode="r")
            index = np.arange(0, row["num_frames"], max(1, round(row["native_fps"] / 10)))
            if len(array) != len(index):
                raise ValueError(f"cached features do not match frame count: {row['sample_id']}")
            self.arrays.append((array, index))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        array, index = self.arrays[i]
        selected = normalized_indices(index, 128, np.random if self.train else None)
        chosen = index[selected]
        return {
            "x": torch.from_numpy(np.array(array[selected], copy=True)),
            "time_valid": torch.ones(128, dtype=torch.bool),
            "frame_seconds": torch.tensor(chosen / row["native_fps"], dtype=torch.float32),
            "entry_index": int(np.abs(chosen - row["entry_frame"]).argmin()),
            "collision_index": int(np.abs(chosen - row["collision_frame"]).argmin()),
            "entry_side": int(row["entry_side"] == "RIGHT"),
            "evasion": int(row["evasion_space"]),
            "entry_s": row["entry_frame"] / row["native_fps"],
            "collision_s": row["collision_frame"] / row["native_fps"],
        }


def to_device(batch):
    return {k: v.cuda() for k, v in batch.items()}


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    metrics = JointMetricAccumulator()
    hits_entry = hits_collision = n = 0
    for batch in loader:
        batch = to_device(batch)
        out = model(batch["x"], batch["time_valid"])
        metrics.update(joint_metric_packet(out, batch))
        entry, collision = constrained_decode(out["entry_logits"], out["collision_logits"])
        entry_s = batch["frame_seconds"].gather(1, entry[:, None])[:, 0]
        collision_s = batch["frame_seconds"].gather(1, collision[:, None])[:, 0]
        hits_entry += int(((entry_s - batch["entry_s"]).abs() <= 0.300001).sum())
        hits_collision += int(((collision_s - batch["collision_s"]).abs() <= 0.300001).sum())
        n += len(entry)
    m = metrics.compute()
    m["acc_entry_native"] = hits_entry / n
    m["acc_collision_native"] = hits_collision / n
    m["competition_score_native"] = (0.35 * (hits_entry + hits_collision) / n
                                     + 0.15 * (m["f1_entry_side_macro"] + m["f1_evasion_space_macro"]))
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_rows = read("/workspace/data/stage2/manifests/train.jsonl")
    val_rows = read("/workspace/data/stage2/manifests/val.jsonl")

    model = TemporalProbe(dropout=0.3).cuda()
    loader = DataLoader(Features(train_rows, True), batch_size=8, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(Features(val_rows), batch_size=8, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    total_steps=args.epochs * len(loader),
                                                    pct_start=0.1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    history, best = [], None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            batch = to_device(batch)
            prediction = model(batch["x"], batch["time_valid"])
            loss, _ = joint_loss(prediction, batch, entry_sigma_seconds=0.15,
                                 collision_sigma_seconds=0.10)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        record = {"epoch": epoch, "loss": total / len(loader), **evaluate(model, val_loader)}
        if best is None or record["competition_score_native"] > best["competition_score_native"]:
            best = record
        history.append(record)
        print(json.dumps(record), flush=True)

    report = {
        "id": "BASELINE-submitted-fps_stage2",
        "seed": args.seed,
        "epochs": args.epochs,
        "selection_train_clips": len(train_rows),
        "selection_val_clips": len(val_rows),
        "best_epoch": best["epoch"],
        "held_out_selection_metrics": best,
        "seconds": time.perf_counter() - started,
        "peak_gpu_MiB": torch.cuda.max_memory_allocated() / 2 ** 20,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "note": "Selection phase only. FPS is not a model input, but the cache stride "
                "and the Gaussian loss widths are FPS-derived, as submitted.",
    }
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
