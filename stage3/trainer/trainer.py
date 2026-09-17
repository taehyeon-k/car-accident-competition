from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from stage3.data.dataset import CachedMotionDataset, motion_collate
from stage3.model import Stage3MotionModel
from stage3.trainer.decoder import decode_predictions
from stage3.trainer.ema import EMA
from stage3.trainer.losses import stage3_loss
from stage3.trainer.metrics import classification_metrics
from stage3.utils.checkpoint import load_checkpoint, save_checkpoint


class Trainer:
    def __init__(self, accelerator, config: dict):
        self.accelerator = accelerator
        self.cfg = config

    def build(self) -> None:
        data = self.cfg["data"]
        self.train_set = CachedMotionDataset(
            data["manifest"], data["crop_frames"], True, self.cfg["seed"],
            data.get("flip_probability", 0.5), self.cfg["targets"],
            data.get("event_fraction", 0.5), data.get("event_position_margin", 8),
        )
        self.val_set = CachedMotionDataset(
            data["val_manifest"], data["crop_frames"], False, self.cfg["seed"],
            0, self.cfg["targets"], 0, data.get("event_position_margin", 8),
        )
        loader_args = dict(batch_size=data["batch_size"], num_workers=data["num_workers"], collate_fn=motion_collate, pin_memory=True)
        self.train_loader = DataLoader(self.train_set, shuffle=True, **loader_args)
        self.val_loader = DataLoader(self.val_set, shuffle=False, **loader_args)
        stats = torch.load(data["statistics"], map_location="cpu", weights_only=True)
        self.physics_center = stats["center"].float()
        self.physics_scale = stats["scale"].float().clamp_min(1e-6)
        self.model = Stage3MotionModel(self.cfg["model"])
        self.ema = EMA(self.model, self.cfg["optimization"].get("ema_decay", 0.999))
        opt = self.cfg["optimization"]
        self.optimizer = AdamW(self.model.parameters(), lr=float(opt["learning_rate"]), weight_decay=float(opt["weight_decay"]))
        updates = max(1, math.ceil(len(self.train_loader) / opt["accumulation_steps"]) * opt["epochs"])
        warmup = round(updates * opt.get("warmup_ratio", 0.05))
        def schedule(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(warmup, 1)
            progress = (step - warmup) / max(updates - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
        self.scheduler = LambdaLR(self.optimizer, schedule)
        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
        )
        self.ema.model.to(self.accelerator.device)
        self.start_epoch, self.global_step, self.best, self.bad_validations = 0, 0, -float("inf"), 0

    def _normalize(self, batch: dict) -> dict:
        center = self.physics_center.to(batch["physics"].device)
        scale = self.physics_scale.to(batch["physics"].device)
        batch["physics"] = (batch["physics"] - center) / scale
        return batch

    def resume(self, path: str | None) -> None:
        if not path:
            return
        state = load_checkpoint(path, "cpu")
        self.accelerator.unwrap_model(self.model).load_state_dict(state["model"])
        self.ema.model.load_state_dict(state["ema_model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.start_epoch = int(state["epoch"]) + 1
        self.global_step, self.best = int(state["global_step"]), float(state["best_metric"])
        self.bad_validations = int(state.get("bad_validations", 0))

    def _save(self, name: str, epoch: int, metrics: dict) -> None:
        if not self.accelerator.is_main_process:
            return
        save_checkpoint(
            Path(self.cfg["output_dir"]) / name, config=self.cfg,
            model=self.accelerator.unwrap_model(self.model).state_dict(), ema_model=self.ema.state_dict(),
            optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
            epoch=epoch, global_step=self.global_step, best_metric=self.best,
            bad_validations=self.bad_validations,
            validation_metrics=metrics, physics_center=self.physics_center, physics_scale=self.physics_scale,
        )

    def train_loop(self, resume: str | None = None) -> None:
        self.resume(resume)
        opt = self.cfg["optimization"]
        for epoch in range(self.start_epoch, opt["epochs"]):
            self.train_set.set_epoch(epoch)
            self.model.train()
            totals: dict[str, float] = {}
            samples = 0
            for batch in self.train_loader:
                batch = self._normalize(batch)
                with self.accelerator.accumulate(self.model):
                    output = self.model(batch["motion"], batch["physics"])
                    loss, parts = stage3_loss(output, batch, self.cfg["loss"])
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), opt["grad_clip_norm"])
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.accelerator.sync_gradients:
                        self.ema.update(self.accelerator.unwrap_model(self.model))
                        self.global_step += 1
                size = batch["motion"].shape[0]
                samples += size
                for name, value in {"loss": loss, **parts}.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
            metrics = self.validate()
            log = {f"train/{key}": value / max(samples, 1) for key, value in totals.items()}
            log.update({f"val/{key}": value for key, value in metrics.items()})
            self.accelerator.log(log, step=self.global_step)
            score = metrics["acceleration_macro_f1_threshold_grid_mean"]
            if score > self.best:
                self.best = score
                self.bad_validations = 0
                self._save("best.pt", epoch, metrics)
            else:
                self.bad_validations += 1
            self._save("last.pt", epoch, metrics)
            stopping = self.cfg.get("early_stopping", {})
            if stopping.get("enabled", False) and self.bad_validations >= int(stopping.get("patience", 4)):
                break

    @torch.inference_mode()
    def validate(self) -> dict[str, float]:
        model = self.ema.model.to(self.accelerator.device).eval()
        predicted_accel, true_accel, predicted_steer, true_steer = [], [], [], []
        continuous_accel, stop_probability, target_direct, target_stopped, target_speed = [], [], [], [], []
        boundary = {"0.5": [], "1.0": []}
        mae = {"acceleration": [], "speed": [], "steering_angle": [], "yaw": []}
        for batch in self.val_loader:
            batch = self._normalize(batch)
            output = model(batch["motion"], batch["physics"])
            for i, valid in enumerate(batch["time_valid"]):
                n = int(valid.sum())
                sample_output = {key: value[i, :n].cpu() for key, value in output.items()}
                pa, ps = decode_predictions(sample_output, self.cfg["decoder"])
                stopped = batch["stopped"][i, :n].cpu().numpy() > 0.5
                direct = 0.5 * (batch["a_long_s1"][i, :n] + batch["a_long_s2"][i, :n]).cpu().numpy()
                acfg = self.cfg["decoder"]["acceleration"]
                ta = np.where(stopped, "STOPPED", np.where(direct > acfg["accelerating_above"], "ACCELERATING", np.where(direct < acfg["decelerating_below"], "DECELERATING", "CONSTANT")))
                angle = batch["steering_angle"][i, :n].cpu().numpy()
                threshold = self.cfg["decoder"]["steering"]["threshold_deg"]
                ts = np.where(angle > threshold, "LEFT", np.where(angle < -threshold, "RIGHT", "STRAIGHT"))
                predicted_accel.extend(pa); true_accel.extend(ta); predicted_steer.extend(ps); true_steer.extend(ts)
                predicted_continuous = sample_output["acceleration"].mean(-1).numpy()
                continuous_accel.extend(predicted_continuous)
                stop_probability.extend(torch.sigmoid(sample_output["stop_logit"]).numpy())
                target_direct.extend(direct); target_stopped.extend(stopped)
                speed_values = batch["speed"][i, :n].cpu().numpy()
                target_speed.extend(speed_values)
                mae["acceleration"].extend(np.abs(predicted_continuous - direct))
                mae["speed"].extend(np.abs(sample_output["speed"].numpy() - batch["speed"][i, :n].cpu().numpy()))
                mae["steering_angle"].extend(np.abs(sample_output["steering_angle"].numpy() - angle))
                if "yaw_rate_aux" in sample_output:
                    yaw_target = batch["yaw_rate_aux"][i, :n].cpu().numpy()
                    yaw_valid = batch["valid_yaw"][i, :n].cpu().numpy().astype(bool)
                    mae["yaw"].extend(np.abs(sample_output["yaw_rate_aux"].numpy()[yaw_valid] - yaw_target[yaw_valid]))
                from stage3.trainer.metrics import boundary_f1
                for seconds in (0.5, 1.0):
                    boundary[f"{seconds:.1f}"].append(boundary_f1(pa, ta, round(seconds * 10)))
        metrics = classification_metrics(np.asarray(predicted_accel), np.asarray(true_accel), np.asarray(predicted_steer), np.asarray(true_steer))
        metrics.update({f"{key}_mae": float(np.nanmean(value)) for key, value in mae.items() if value})
        metrics["direct_acceleration_mae"] = metrics["acceleration_mae"]
        for seconds, values in boundary.items():
            metrics[f"boundary_f1_{seconds}s"] = float(np.mean([value[0] for value in values]))
            metrics[f"boundary_delay_{seconds}s"] = float(np.mean([value[1] for value in values]))
        continuous_accel, stop_probability = np.asarray(continuous_accel), np.asarray(stop_probability)
        target_direct, target_stopped, target_speed = map(np.asarray, (target_direct, target_stopped, target_speed))
        grid = self.cfg.get("validation", {}).get("acceleration_threshold_grid", [-0.30, -0.25, -0.20])
        grid_scores = []
        for decelerating in grid:
            for accelerating in [-value for value in reversed(grid)]:
                prediction = np.where(stop_probability >= 0.5, "STOPPED", np.where(continuous_accel > accelerating, "ACCELERATING", np.where(continuous_accel < decelerating, "DECELERATING", "CONSTANT")))
                grid_scores.append(f1_score(true_accel, prediction, labels=["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"], average="macro", zero_division=0))
        metrics["acceleration_macro_f1_threshold_grid_mean"] = float(np.mean(grid_scores))
        predicted_accel_array, true_accel_array = np.asarray(predicted_accel), np.asarray(true_accel)
        for name, low, high in (("stopped", 0, 0.15), ("low", 0.15, 2.0), ("medium", 2.0, 8.0), ("high", 8.0, np.inf)):
            selected = (target_speed >= low) & (target_speed < high)
            if selected.any():
                metrics[f"acceleration_macro_f1_speed_{name}"] = float(f1_score(true_accel_array[selected], predicted_accel_array[selected], labels=["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"], average="macro", zero_division=0))
        return metrics
