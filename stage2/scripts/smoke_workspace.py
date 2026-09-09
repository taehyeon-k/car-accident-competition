"""One real-data Accelerate epoch, validation, gradient checks and checkpoint resume on CUDA."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator

from stage2.data.dataset import NativeDataset
from stage2.data.geometry_stats import fit_statistics
from stage2.trainer.trainer import Trainer
from stage2.utils.utils import atomic_save, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["coarse", "fine"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / f"{args.stage}.workspace.yaml"
    )
    output = Path("/workspace/runs/smoke") / args.stage
    output.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output)
    config["data"].update(
        manifest="/workspace/data/stage2/manifests/smoke_train.jsonl",
        val_manifest="/workspace/data/stage2/manifests/smoke_val.jsonl",
        geometry_stats=str(output / "geometry_stats.pt"),
        num_workers=0,
    )
    config["optimization"].update(epochs=1)
    config["logging"]["log_every"] = 1
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    dataset = NativeDataset(
        config["data"]["manifest"],
        args.stage,
        config["tracking"],
        True,
        geometry_only=True,
    )
    statistics = fit_statistics((dataset[i] for i in range(len(dataset))), args.stage)
    statistics["source_ids"] = sorted({r["source_id"] for r in dataset.rows})
    atomic_save(statistics, config["data"]["geometry_stats"])
    accelerator = Accelerator(
        gradient_accumulation_steps=config["optimization"]["accumulation_steps"],
        mixed_precision="bf16",
    )
    trainer = Trainer(accelerator, config)
    trainer.build()
    model = accelerator.unwrap_model(trainer.model)
    lora = {n: p for n, p in model.named_parameters() if n.endswith(".B")}
    before = {n: p.detach().clone() for n, p in lora.items()}
    gradient_checks = {}

    def check(name):
        def hook(gradient):
            if not torch.isfinite(gradient).all():
                raise RuntimeError(f"Nonfinite gradient: {name}")
            gradient_checks[name] = True
            return gradient

        return hook

    handles = [
        p.register_hook(check(n))
        for n, p in model.named_parameters()
        if p.requires_grad
    ]
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train_loop()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    updated = [n for n, p in lora.items() if not torch.equal(before[n], p)]
    assert updated and trainer.step > 0
    assert np.isfinite(trainer.best_validation_loss)
    assert all(n in gradient_checks for n in lora)
    resume_epoch = trainer.load(str(output / "last.pt"))
    assert resume_epoch == 1
    report = dict(
        stage=args.stage,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        mixed_precision="bf16",
        optimizer_steps=trainer.step,
        validation_loss=trainer.best_validation_loss,
        lora_B_tensors_updated=len(updated),
        lora_B_tensors_total=len(lora),
        finite_gradient_tensors=len(gradient_checks),
        elapsed_seconds=elapsed,
        peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
        geometry_observations=statistics["observations"],
        checkpoint_resume_epoch=resume_epoch,
        passed=True,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    for handle in handles:
        handle.remove()
    accelerator.end_training()


if __name__ == "__main__":
    main()
