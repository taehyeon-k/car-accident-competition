"""Accelerate training loop shared by coarse and fine Stage 2 models."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data.data import get_data
from model.model import CoarseModel, FineModel
from model.pipeline import CoarseSystem, FineSystem
from utils.losses import coarse_loss, fine_loss


class Trainer:
    """Owns model lifecycle, optimization, validation, and resumable checkpoints."""

    def __init__(self, accelerator: Any, config: dict[str, Any]) -> None:
        self.accelerator = accelerator
        self.config = config
        self.stage = config["stage"]
        self.step = 0
        self.best_validation_loss = float("inf")

    def build(self) -> None:
        """Build loaders, choose live-visual or cached-feature training, and prepare state."""
        data_config = self.config["data"]
        self.train_dataset, self.train_loader = get_data(
            data_config["manifest"],
            data_config["batch_size"],
            data_config["num_workers"],
            shuffle=True,
        )
        self.validation_dataset, self.validation_loader = get_data(
            data_config["val_manifest"],
            data_config["batch_size"],
            data_config["num_workers"],
            shuffle=False,
        )

        model_config = self.config["model"]
        live_visual = bool(
            model_config.get("vjepa_factory")
            if self.stage == "coarse"
            else model_config.get("dino_factory")
        )

        if live_visual:
            self.model = CoarseSystem(model_config) if self.stage == "coarse" else FineSystem(model_config)
        else:
            self.model = CoarseModel() if self.stage == "coarse" else FineModel()

        self.optimizer, self.scheduler = self._build_optimizer()
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.validation_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.validation_loader,
            self.scheduler,
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
        optimizer = AdamW(parameter_groups, weight_decay=optimization["weight_decay"])

        total_updates = math.ceil(
            len(self.train_loader)
            * optimization["epochs"]
            / optimization["accumulation_steps"]
        )
        warmup_updates = max(1, int(total_updates * optimization["warmup_ratio"]))

        def cosine_with_warmup(step: int) -> float:
            if step < warmup_updates:
                return (step + 1) / warmup_updates
            progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return optimizer, LambdaLR(optimizer, cosine_with_warmup)

    def _forward(self, batch: dict[str, torch.Tensor]):
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

        return coarse_loss(outputs, batch) if self.stage == "coarse" else fine_loss(outputs, batch)

    def validate(self) -> float:
        """Return globally averaged validation loss without updating model state."""
        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch in self.validation_loader:
                loss, _ = self._forward(batch)
                losses.append(loss.detach())

        local_mean = torch.stack(losses).mean()
        global_mean = self.accelerator.gather_for_metrics(local_mean).mean().item()
        self.model.train()
        return global_mean

    def save(self, epoch: int, filename: str) -> None:
        """Save model, optimizer, scheduler, configuration, and progress together."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "step": self.step,
            "model": self.accelerator.unwrap_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, output_dir / filename)

    def load(self, checkpoint_path: str) -> int:
        """Restore all training state and return the next epoch index."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.step = checkpoint["step"]
        return checkpoint["epoch"] + 1

    def train_loop(self, resume_path: str | None = None) -> None:
        """Train all epochs, validating and writing both best and latest checkpoints."""
        start_epoch = self.load(resume_path) if resume_path else 0
        optimization = self.config["optimization"]
        logging = self.config["logging"]
        self.model.train()

        for epoch in range(start_epoch, optimization["epochs"]):
            for batch in self.train_loader:
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        loss, _ = self._forward(batch)

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), optimization["grad_clip_norm"]
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.step += 1

                if self.step % logging["log_every"] == 0:
                    self.accelerator.log(
                        {"train/loss": loss.item(), "lr": self.scheduler.get_last_lr()[-1]},
                        step=self.step,
                    )

            if (epoch + 1) % logging["val_every"] == 0:
                validation_loss = self.validate()
                self.accelerator.log({"val/loss": validation_loss}, step=self.step)
                if validation_loss < self.best_validation_loss:
                    self.best_validation_loss = validation_loss
                    self.save(epoch, "best.pt")

            self.save(epoch, "last.pt")
