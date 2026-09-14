"""Accelerate training loop for the joint Stage 2 model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from stage2.data.data import get_data
from stage2.model.joint_system import JointSystem
from stage2.utils.joint_metrics import JointMetricAccumulator, joint_metric_packet
from stage2.utils.joint_losses import joint_loss
from stage2.utils.checkpoint import rank_rng_states, restore_rng, save_checkpoint
from stage2.utils.utils import source_name, validate_config
from stage2.utils.tracking import TrainingMetrics
from stage2.utils.early_stopping import EarlyStopping


class _MetricBucket:
    """Loss sums and joint metrics over one slice of the validation samples."""

    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.count = 0
        self.joint = JointMetricAccumulator()

    def update(self, gathered: dict, mask: torch.Tensor | None = None) -> None:
        selected = {
            name: values if mask is None else values[mask]
            for name, values in gathered.items()
        }
        self.joint.update({k: v for k, v in selected.items() if k.startswith("_")})
        scalars = {k: v for k, v in selected.items() if not k.startswith("_")}
        self.count += scalars["loss"].numel()
        for name, values in scalars.items():
            self.sums[name] = self.sums.get(name, 0) + values.sum()

    def compute(self) -> dict[str, float]:
        if not self.count:
            return {}
        return {
            **{name: value.item() / self.count for name, value in self.sums.items()},
            **self.joint.compute(),
        }


class Trainer:
    """Owns model lifecycle, optimization, validation, and resumable checkpoints."""

    def __init__(
        self,
        accelerator: Any,
        config: dict[str, Any],
    ) -> None:
        self.accelerator = accelerator
        self.config = config
        self.stage = config["stage"]
        self.step = 0
        self.best_validation_loss = float("inf")
        self.best_competition_score = -float("inf")
        self.joint_train_metrics = JointMetricAccumulator()
        self.last_validation_metrics = {}
        self.validation_sources: list[str] = []
        self.early_stopping = EarlyStopping(
            config.get("early_stopping"),
            config.get("logging", {}).get("checkpoint_metric", "loss"),
        )

    def build(self) -> None:
        """Build loaders, choose live-visual or cached-feature training, and prepare state."""
        data_config = self.config["data"]
        validate_config(self.config)
        self.train_dataset, self.train_loader = get_data(
            data_config["manifest"],
            data_config["batch_size"],
            data_config["num_workers"],
            shuffle=True,
            config=self.config,
        )
        self.validation_dataset, self.validation_loader = get_data(
            data_config["val_manifest"],
            data_config["batch_size"],
            data_config["num_workers"],
            shuffle=False,
            config=self.config,
        )
        # Fixed, sorted order gives stable per-source metric keys across runs.
        self.validation_sources = sorted(
            {
                source_name(row["source_id"])
                for row in self.validation_dataset.rows
                if row.get("source_id")
            }
        )

        self.model = JointSystem(self.config["model"])

        # Compute scheduler lengths after distributed loader sharding. Keep the
        # scheduler unwrapped and advance it exactly once per successful update.
        self.train_loader, self.validation_loader = self.accelerator.prepare(
            self.train_loader,
            self.validation_loader,
        )

        self.optimizer, self.scheduler = self._build_optimizer()
        self.model, self.optimizer = self.accelerator.prepare(
            self.model,
            self.optimizer,
        )

    def _build_optimizer(self) -> tuple[AdamW, LambdaLR]:
        """Each backbone's adapters get their own LR; the joint head uses new_lr."""
        optimization = self.config["optimization"]
        groups = self.model.parameter_groups()
        # Group order is fixed: the scheduler and the LR logs index into it.
        parameter_groups = [
            {"params": groups["dino_lora"], "lr": optimization["dino_lora_lr"]},
            {"params": groups["vjepa_lora"], "lr": optimization["vjepa_lora_lr"]},
            {"params": groups["new"], "lr": optimization["new_lr"]},
        ]
        optimizer = AdamW(
            parameter_groups,
            weight_decay=optimization["weight_decay"],
        )

        total_updates = (
            math.ceil(len(self.train_loader) / optimization["accumulation_steps"])
            * optimization["epochs"]
        )
        warmup_updates = max(
            1,
            int(total_updates * optimization["warmup_ratio"]),
        )
        # LambdaLR.state_dict() excludes the closure, so a resume only restores
        # last_epoch. Record the curve's geometry here; load() rejects a
        # checkpoint whose schedule was built from different settings rather
        # than replaying a restored step count against a different curve.
        self.schedule_geometry = {
            "total_updates": total_updates,
            "warmup_updates": warmup_updates,
        }

        def cosine_with_warmup(step: int) -> float:
            if step < warmup_updates:
                return (step + 1) / warmup_updates
            progress = min(
                1.0,
                (step - warmup_updates)
                / max(
                    1,
                    total_updates - warmup_updates,
                ),
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return optimizer, LambdaLR(
            optimizer,
            cosine_with_warmup,
        )

    def _forward(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str = "mean",
    ):
        """Run the joint model and pair its losses with per-sample metric counts."""
        outputs = self.model(batch)
        loss, components = joint_loss(
            outputs,
            batch,
            reduction=reduction,
            **self.config.get("loss", {}),
        )
        return loss, {**components, **joint_metric_packet(outputs, batch)}

    def validate(self) -> float:
        """Return globally averaged validation loss without updating model state."""
        self.model.eval()
        overall = _MetricBucket()
        # Per-source buckets view exactly the same gathered, duplicate-trimmed
        # samples as the overall bucket, so the slices reconcile with the total.
        by_source = {name: _MetricBucket() for name in self.validation_sources}

        with torch.no_grad():
            for batch in self.validation_loader:
                with self.accelerator.autocast():
                    losses, metrics = self._forward(
                        batch,
                        reduction="none",
                    )
                packet = {"loss": losses, **metrics}
                if self.validation_sources:
                    packet["source_index"] = torch.tensor(
                        [
                            self.validation_sources.index(source_name(value))
                            for value in batch["source_id"]
                        ],
                        device=self.accelerator.device,
                    )
                gathered = self.accelerator.gather_for_metrics(packet)
                origin = gathered.pop("source_index", None)
                overall.update(gathered)
                if origin is not None:
                    for position, name in enumerate(self.validation_sources):
                        by_source[name].update(gathered, origin == position)

        if overall.count == 0:
            raise ValueError("Validation loader is empty")
        self.last_validation_metrics = overall.compute()
        self.accelerator.log(
            {
                **{
                    f"val/{name}": value
                    for name, value in self.last_validation_metrics.items()
                },
                # The ordering loss is a batch-level regulariser, not a per-source
                # quality signal, so it is excluded from the source breakdown.
                **{
                    f"val/{source}/{name}": value
                    for source, bucket in by_source.items()
                    for name, value in bucket.compute().items()
                    if name != "loss_invalid_order"
                },
            },
            step=self.step,
        )
        global_mean = self.last_validation_metrics["loss"]
        self.model.train()
        return global_mean

    def save(
        self,
        epoch: int,
        filename: str,
    ) -> None:
        """Save model, optimizer, scheduler, configuration, and progress together."""
        rng_states = rank_rng_states()
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        checkpoint = {
            "epoch": epoch,
            "step": self.step,
            "model": self.accelerator.unwrap_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "schedule_geometry": self.schedule_geometry,
            "config": self.config,
            "best_validation_loss": self.best_validation_loss,
            "best_competition_score": self.best_competition_score,
            "joint_train_metrics": self.joint_train_metrics.state_dict(),
            "early_stopping": self.early_stopping.state_dict(),
            "validation_metrics": self.last_validation_metrics,
            "rng_states": rng_states,
            "scaler": (
                self.accelerator.scaler.state_dict()
                if self.accelerator.scaler is not None
                else None
            ),
        }
        save_checkpoint(
            checkpoint,
            output_dir / filename,
        )

    def load(
        self,
        checkpoint_path: str,
    ) -> int:
        """Restore all training state and return the next epoch index."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("format_version") != 2:
            raise ValueError(
                "Checkpoint predates corrected module names; explicit migration is required"
            )
        if checkpoint["config"]["stage"] != self.stage:
            raise ValueError(
                "Checkpoint stage differs from the requested training stage"
            )
        saved_geometry = checkpoint.get("schedule_geometry")
        if saved_geometry is None:
            raise ValueError(
                "Checkpoint predates schedule-geometry tracking; its learning-rate "
                "curve cannot be verified. Start a new run instead of resuming."
            )
        if saved_geometry != self.schedule_geometry:
            raise ValueError(
                "Learning-rate schedule differs from the checkpoint's: "
                f"{saved_geometry} vs {self.schedule_geometry}. The restored step "
                "count would replay against a different curve. Restore the original "
                "epochs/accumulation_steps/batch_size, or start a new run."
            )
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.step = checkpoint["step"]
        self.best_validation_loss = checkpoint["best_validation_loss"]
        self.best_competition_score = checkpoint.get(
            "best_competition_score", -float("inf")
        )
        if "joint_train_metrics" in checkpoint:
            self.joint_train_metrics.load_state_dict(
                checkpoint["joint_train_metrics"], self.accelerator.device
            )
        self.early_stopping.load_state_dict(checkpoint.get("early_stopping", {}))
        self.last_validation_metrics = checkpoint.get("validation_metrics", {})
        if len(checkpoint["rng_states"]) != self.accelerator.num_processes:
            raise ValueError("Exact resume requires the same process count")
        restore_rng(checkpoint["rng_states"][self.accelerator.process_index])
        if checkpoint["scaler"] is not None and self.accelerator.scaler is not None:
            self.accelerator.scaler.load_state_dict(checkpoint["scaler"])
        return checkpoint["epoch"] + 1

    def train_loop(
        self,
        resume_path: str | None = None,
    ) -> None:
        """Train all epochs, validating and writing both best and latest checkpoints."""
        start_epoch = self.load(resume_path) if resume_path else 0
        optimization = self.config["optimization"]
        logging = self.config["logging"]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        training_metrics = TrainingMetrics()
        for epoch in range(
            start_epoch,
            optimization["epochs"],
        ):
            if hasattr(
                self.train_dataset,
                "set_epoch",
            ):
                self.train_dataset.set_epoch(epoch)
            if hasattr(
                self.train_loader,
                "set_epoch",
            ):
                self.train_loader.set_epoch(epoch)
            stop_training = False
            accumulated_samples = 0
            if self.stage != "joint":
                training_metrics = TrainingMetrics()
            for batch in self.train_loader:
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        sample_losses, components = self._forward(
                            batch,
                            reduction="none",
                        )
                    packet = {k: v for k, v in components.items() if k.startswith("_")}
                    self.joint_train_metrics.update(
                        self.accelerator.gather_for_metrics(packet)
                    )
                    training_metrics.update(
                        sample_losses,
                        {k: v for k, v in components.items() if not k.startswith("_")},
                    )
                    accumulated_samples += sample_losses.numel()

                    # Accelerate divides by the accumulation count. Accumulate
                    # sums here and normalize by actual samples at the boundary,
                    # including an incomplete final accumulation group.
                    self.accelerator.backward(
                        sample_losses.sum() * optimization["accumulation_steps"]
                    )
                    if self.accelerator.sync_gradients:
                        samples = torch.tensor(
                            float(accumulated_samples),
                            device=sample_losses.device,
                        )
                        samples = (
                            self.accelerator.reduce(
                                samples,
                                reduction="sum",
                            )
                            / self.accelerator.num_processes
                        )
                        for parameter in self.model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.div_(samples)
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            optimization["grad_clip_norm"],
                        )
                        self.optimizer.step()
                        if not self.accelerator.optimizer_step_was_skipped:
                            self.scheduler.step()
                            self.step += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        accumulated_samples = 0

                if (
                    self.accelerator.sync_gradients
                    and not self.accelerator.optimizer_step_was_skipped
                    and self.step % logging["log_every"] == 0
                ):
                    self.accelerator.log(
                        training_metrics.flush(self.accelerator),
                        step=self.step,
                    )

            if (epoch + 1) % logging["val_every"] == 0:
                validation_loss = self.validate()
                self.accelerator.log(
                    {
                        **{
                            f"train/{k}": v
                            for k, v in self.joint_train_metrics.compute().items()
                        },
                        "train_config/epoch": epoch + 1,
                        "train_config/lr_dino_lora": self.scheduler.get_last_lr()[0],
                        "train_config/lr_vjepa_lora": self.scheduler.get_last_lr()[1],
                        "train_config/lr_new_parameters": self.scheduler.get_last_lr()[
                            -1
                        ],
                    },
                    step=self.step,
                )
                self.joint_train_metrics = JointMetricAccumulator()
                score = self.last_validation_metrics.get(
                    "competition_score", -float("inf")
                )
                select_score = (
                    logging.get("checkpoint_metric", "loss") == "competition_score"
                )
                improved = (
                    score > self.best_competition_score
                    if select_score
                    else validation_loss < self.best_validation_loss
                )
                self.best_validation_loss = min(
                    self.best_validation_loss, validation_loss
                )
                self.best_competition_score = max(self.best_competition_score, score)
                stop_training = self.early_stopping.update(
                    {"loss": validation_loss, **self.last_validation_metrics}
                )
                if self.early_stopping.enabled:
                    self.accelerator.log(
                        {
                            "train_config/early_stopping_bad_validations": self.early_stopping.bad_validations,
                            "train_config/early_stopping_triggered": int(stop_training),
                        },
                        step=self.step,
                    )
                if improved:
                    self.save(epoch, "best.pt")

            self.save(
                epoch,
                "last.pt",
            )
            if stop_training:
                if self.accelerator.is_main_process:
                    print(
                        f"Early stopping at epoch {epoch + 1}: {self.early_stopping.monitor} "
                        f"did not improve for {self.early_stopping.patience} validations. "
                        "Use best.pt for inference."
                    )
                break
