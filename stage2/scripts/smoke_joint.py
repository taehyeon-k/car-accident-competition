"""Real frozen-backbone joint smoke: local cache, update, validation, resume, inference."""

import argparse
import json
from pathlib import Path

import torch
from accelerate import Accelerator

from stage2.data.cache_joint_features import cache_manifest
from stage2.joint_test import cache_item, predict
from stage2.trainer.trainer import Trainer
from stage2.utils.utils import load_config, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="stage2/configs/joint.workspace.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output)
    config["data"].update(
        feature_dir=str(output / "local_cache"), batch_size=1, num_workers=0
    )
    config["model"]["dino_batch_size"] = 8
    config["optimization"].update(
        epochs=1,
        accumulation_steps=1,
        mixed_precision="bf16" if args.device == "cuda" else "no",
    )
    config["logging"].update(val_every=1, checkpoint_metric="competition_score")
    config["logging"]["wandb"]["enabled"] = False
    for key in ("manifest", "val_manifest"):
        row = read_manifest(config["data"][key])[0]
        path = output / f"{key}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        config["data"][key] = str(path)
        cache_manifest(config, str(path), args.device)
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        gradient_accumulation_steps=1,
        mixed_precision=config["optimization"]["mixed_precision"],
    )
    logs = []
    accelerator.log = lambda values, step: logs.append({"step": step, **values})
    trainer = Trainer(accelerator, config)
    trainer.build()
    system = accelerator.unwrap_model(trainer.model)
    before = system.head.global_projection[0].weight.detach().clone()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trainer.train_loop()
    assert trainer.step == 1
    assert not torch.equal(before, system.head.global_projection[0].weight)
    assert all(
        p.grad is None and not p.requires_grad
        for p in system.global_visual.parameters()
    )
    assert not system.global_visual.encoder.training
    assert trainer.load(str(output / "last.pt")) == 1
    row = read_manifest(config["data"]["val_manifest"])[0]
    for key in (
        "native_fps",
        "entry_frame",
        "collision_frame",
        "entry_side",
        "evasion_space",
    ):
        row.pop(key, None)
    prediction = predict(
        system.eval(), cache_item(row, Path(config["data"]["feature_dir"])), args.device
    )
    assert prediction["entry_frame"] <= prediction["collision_frame"]
    report = {
        "updates": trainer.step,
        "prediction_without_fps_or_labels": prediction,
        "validation": trainer.last_validation_metrics,
        "logs": logs,
        "peak_cuda_bytes": (
            torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        ),
        "frozen_vjepa": True,
        "projection_updated": True,
        "resume_epoch": 1,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    accelerator.end_training()


if __name__ == "__main__":
    main()
