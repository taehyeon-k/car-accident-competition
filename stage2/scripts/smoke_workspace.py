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
    parser.add_argument("--output-root", default="/workspace/runs/smoke")
    parser.add_argument("--overfit-steps", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / f"{args.stage}.workspace.yaml"
    )
    output = Path(args.output_root) / args.stage
    output.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output)
    config["data"].update(
        manifest="/workspace/data/stage2/manifests/smoke_train.jsonl",
        val_manifest="/workspace/data/stage2/manifests/smoke_val.jsonl",
        geometry_stats=str(output / "geometry_stats.pt"),
        num_workers=0,
    )
    # Repeat the smoke sample to exercise full batches and an accumulation boundary.
    batch_size = config["data"]["batch_size"]
    accumulation = config["optimization"]["accumulation_steps"]
    for key, count in (
        ("manifest", batch_size * accumulation),
        ("val_manifest", batch_size),
    ):
        row = Path(config["data"][key]).read_text().splitlines()[0]
        manifest = output / f"{key}.jsonl"
        manifest.write_text((row + "\n") * count)
        config["data"][key] = str(manifest)
    config["optimization"].update(epochs=1)
    config["logging"]["wandb"]["enabled"] = False
    config["logging"]["log_every"] = 1
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    dataset = NativeDataset(
        config["data"]["manifest"],
        args.stage,
        config["tracking"],
        True,
        geometry_only=True,
        coarse_t_max=config["model"].get("T_max", 32),
    )
    statistics = fit_statistics((dataset[i] for i in range(len(dataset))), args.stage)
    statistics["coarse_t_max"] = config["model"].get("T_max", 32)
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
    base = {
        n: p
        for n, p in model.named_parameters()
        if n.startswith("visual.encoder.")
        and p.requires_grad
        and not n.endswith((".A", ".B"))
    }
    base_before = {n: p.detach().cpu().clone() for n, p in base.items()}
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
    base_updated = [
        n for n, p in base.items() if not torch.equal(base_before[n], p.detach().cpu())
    ]
    assert updated and trainer.step > 0
    if config["model"].get("unfreeze_last_blocks", 0):
        assert base_updated, "Unfrozen backbone weights did not update"
        assert all(
            n in gradient_checks for n in base
        ), "Missing unfrozen backbone gradients"
    del base_before
    assert np.isfinite(trainer.best_validation_loss)
    assert all(n in gradient_checks for n in lora)
    resume_epoch = trainer.load(str(output / "last.pt"))
    assert resume_epoch == 1
    if args.overfit_steps:
        # Fix RGB/crops while retaining train-mode activation checkpointing.
        batches = list(trainer.train_loader)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        for i, group in enumerate(trainer.optimizer.param_groups):
            group["weight_decay"] = 0.0
            group["lr"] = config["optimization"]["lora_lr" if i == 0 else "new_lr"]

        def measure():
            values = []
            with torch.no_grad(), accelerator.autocast():
                for batch in batches:
                    loss, metrics = trainer._forward(batch)
                    values.append(
                        {
                            "loss": loss.item(),
                            **{k: v.float().mean().item() for k, v in metrics.items()},
                        }
                    )
            return {k: float(np.mean([v[k] for v in values])) for k in values[0]}

        history = [{"step": 0, **measure()}]
        print(json.dumps({"overfit": history[-1]}), flush=True)
        for step in range(1, args.overfit_steps + 1):
            trainer.optimizer.zero_grad(set_to_none=True)
            for batch in batches:
                with accelerator.autocast():
                    loss, _ = trainer._forward(batch)
                assert torch.isfinite(loss), "Nonfinite overfit loss"
                accelerator.backward(
                    loss * config["optimization"]["accumulation_steps"] / len(batches)
                )
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            trainer.optimizer.step()
            if step % 5 == 0 or step == args.overfit_steps:
                history.append({"step": step, **measure()})
                (output / "overfit.json").write_text(
                    json.dumps(history, indent=2) + "\n"
                )
                print(json.dumps({"overfit": history[-1]}), flush=True)
        assert (
            history[-1]["loss"] < history[0]["loss"]
        ), "Fixed-sample loss did not decrease"
    report = dict(
        stage=args.stage,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        mixed_precision="bf16",
        optimizer_steps=trainer.step,
        batch_size=batch_size,
        T_max=config["model"].get("T_max", 32 if args.stage == "coarse" else 64),
        accumulation_steps=accumulation,
        unfreeze_last_blocks=config["model"].get("unfreeze_last_blocks", 0),
        unfrozen_base_tensors_updated=len(base_updated),
        unfrozen_base_tensors_total=len(base),
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
    if args.overfit_steps:
        report["overfit"] = {
            "initial": history[0],
            "final": history[-1],
            "batches": len(batches),
            "steps": args.overfit_steps,
        }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    for handle in handles:
        handle.remove()
    accelerator.end_training()


if __name__ == "__main__":
    main()
