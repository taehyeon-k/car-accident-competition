"""Accelerate training loop shared by coarse and fine Stage 2 models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from stage2.data.data import get_data
from stage2.model.model import CoarseModel, FineModel
from stage2.model.pipeline import CoarseSystem, FineSystem
from stage2.utils.losses import coarse_loss, fine_loss
from stage2.utils.metrics import batch_metrics
from stage2.utils.checkpoint import rank_rng_states, restore_rng, save_checkpoint
from stage2.utils.utils import validate_config


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

        model_config = self.config["model"]
        live_visual = (
            model_config.get(
                "training_mode",
                "lora",
            )
            == "lora"
        )
        train_sources = {
            str(row["source_id"])
            for row in self.train_dataset.rows
            if "source_id" in row
        }
        val_sources = {
            str(row["source_id"])
            for row in self.validation_dataset.rows
            if "source_id" in row
        }
        if train_sources & val_sources:
            raise ValueError("Training and validation share original source IDs")

        if live_visual:
            self.model = (
                CoarseSystem(model_config)
                if self.stage == "coarse"
                else FineSystem(model_config)
            )
        else:
            self.model = CoarseModel() if self.stage == "coarse" else FineModel()

        if live_visual:
            statistics = torch.load(
                data_config["geometry_stats"],
                map_location="cpu",
                weights_only=True,
            )
            if statistics["stage"] != self.stage or statistics["split"] != "train":
                raise ValueError(
                    "Geometry statistics must come from this stage's training split"
                )
            if set(statistics["source_ids"]) != train_sources:
                raise ValueError(
                    "Geometry statistics do not match the training source split"
                )
            self.model.head.geometry_embedding.set_statistics(statistics)

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
        """Use the architecture's separate LoRA and newly initialized LR groups."""
        optimization = self.config["optimization"]
        lora_parameters = []
        new_parameters = []

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "visual.encoder" in name:
                lora_parameters.append(parameter)
            else:
                new_parameters.append(parameter)

        parameter_groups = [
            {"params": lora_parameters, "lr": optimization["lora_lr"]},
            {"params": new_parameters, "lr": optimization["new_lr"]},
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
        """Select the correct feature source and loss for the active training stage."""
        has_live_images = "coarse_rgb" in batch or "fine_rgb" in batch

        if has_live_images:
            outputs = self.model(batch)
        elif self.stage == "coarse":
            outputs = self.model(
                batch["dense"],
                batch["boxes_grid"],
                batch["geometry"],
                batch["object_valid"].bool(),
                batch["bin_valid"].bool(),
            )
        else:
            outputs = self.model(
                batch["global_tokens"],
                batch["dense"],
                batch["boxes_grid"],
                batch["geometry"],
                batch["object_valid"].bool(),
                batch["time_valid"].bool(),
            )

        objective = coarse_loss if self.stage == "coarse" else fine_loss
        loss, components = objective(
            outputs,
            batch,
            reduction=reduction,
        )
        return loss, {
            **components,
            **batch_metrics(
                outputs,
                batch,
                self.stage,
            ),
        }

    def validate(self) -> float:
        """Return globally averaged validation loss without updating model state."""
        self.model.eval()
        total = torch.zeros(
            (),
            device=self.accelerator.device,
        )
        count = 0
        metrics_sum = {}

        with torch.no_grad():
            for batch in self.validation_loader:
                with self.accelerator.autocast():
                    losses, metrics = self._forward(
                        batch,
                        reduction="none",
                    )
                gathered = self.accelerator.gather_for_metrics(
                    {"loss": losses, **metrics}
                )
                total += gathered["loss"].sum()
                count += gathered["loss"].numel()
                for name, values in gathered.items():
                    metrics_sum[name] = (
                        metrics_sum.get(
                            name,
                            0,
                        )
                        + values.sum()
                    )

        if count == 0:
            raise ValueError("Validation loader is empty")
        self.accelerator.log(
            {
                f"val/{name}": value.item() / count
                for name, value in metrics_sum.items()
            },
            step=self.step,
        )
        global_mean = (total / count).item()
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
            "config": self.config,
            "best_validation_loss": self.best_validation_loss,
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
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.step = checkpoint["step"]
        self.best_validation_loss = checkpoint["best_validation_loss"]
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
            accumulated_samples = 0
            for batch in self.train_loader:
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        sample_losses, _ = self._forward(
                            batch,
                            reduction="none",
                        )
                        loss = sample_losses.mean()
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
                    and self.step % logging["log_every"] == 0
                ):
                    self.accelerator.log(
                        {
                            "train/loss": loss.item(),
                            "lr": self.scheduler.get_last_lr()[-1],
                        },
                        step=self.step,
                    )

            if (epoch + 1) % logging["val_every"] == 0:
                validation_loss = self.validate()
                self.accelerator.log(
                    {"val/loss": validation_loss},
                    step=self.step,
                )
                if validation_loss < self.best_validation_loss:
                    self.best_validation_loss = validation_loss
                    self.save(
                        epoch,
                        "best.pt",
                    )

            self.save(
                epoch,
                "last.pt",
            )
        self.accelerator.end_training()
