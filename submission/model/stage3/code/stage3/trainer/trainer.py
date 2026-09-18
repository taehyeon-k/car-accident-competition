from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from stage3.data.cache import cache_key
from stage3.data.dataset import CachedMotionDataset, motion_collate
from stage3.model import Stage3MotionModel
from stage3.trainer.decoder import decode_predictions
from stage3.trainer.ema import EMA
from stage3.trainer.losses import stage3_loss
from stage3.trainer.metrics import (
    TrainingCompetitionMetrics, classification_targets, classification_metrics, boundary_f1,
)
from stage3.utils.checkpoint import load_checkpoint, save_checkpoint


class Trainer:
    BEST_METRIC = "competition_score"

    def __init__(self, accelerator, config: dict):
        self.accelerator = accelerator
        self.cfg = config

    def build(self) -> None:
        data = self.cfg["data"]
        self.train_set = CachedMotionDataset(
            data["manifest"], data["crop_frames"], True, self.cfg["seed"],
            data.get("flip_probability", 0.5), self.cfg["targets"],
            data.get("event_fraction", 0.5), data.get("event_position_margin", 8), cache_key(self.cfg),
        )
        self.val_set = CachedMotionDataset(
            data["val_manifest"], data["crop_frames"], False, self.cfg["seed"],
            0, self.cfg["targets"], 0, data.get("event_position_margin", 8), cache_key(self.cfg),
        )
        loader_args = dict(batch_size=data["batch_size"], num_workers=data["num_workers"], collate_fn=motion_collate, pin_memory=True)
        self.train_loader = DataLoader(self.train_set, shuffle=True, **loader_args)
        self.val_loader = DataLoader(self.val_set, shuffle=False, **loader_args)
        stats = torch.load(data["statistics"], map_location="cpu", weights_only=True)
        if stats.get("cache_key") != cache_key(self.cfg):
            raise ValueError("Stale physics statistics; rebuild motion caches and recompute statistics")
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
        if state["feature_cache_key"] != cache_key(self.cfg):
            raise ValueError("Resume configuration has different motion features from the checkpoint")
        self.physics_center = state["physics_center"].float()
        self.physics_scale = state["physics_scale"].float().clamp_min(1e-6)
        self.accelerator.unwrap_model(self.model).load_state_dict(state["model"])
        self.ema.model.load_state_dict(state["ema_model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.start_epoch = int(state["epoch"]) + 1
        self.global_step = int(state["global_step"])
        if state.get("best_metric_name") == self.BEST_METRIC:
            self.best = float(state["best_metric"])
            self.bad_validations = int(state.get("bad_validations", 0))
        else:
            # Older checkpoints selected on acceleration threshold-grid F1.
            # Do not compare that historical number with the weighted score.
            self.best, self.bad_validations = -float("inf"), 0
            self.accelerator.print("Reset best-score/early-stopping history: selection now uses validation competition_score.")

    def _save(self, name: str, epoch: int, metrics: dict) -> None:
        if not self.accelerator.is_main_process:
            return
        save_checkpoint(
            Path(self.cfg["output_dir"]) / name, config=self.cfg,
            model=self.accelerator.unwrap_model(self.model).state_dict(), ema_model=self.ema.state_dict(),
            optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
            epoch=epoch, global_step=self.global_step, best_metric=self.best,
            best_metric_name=self.BEST_METRIC,
            bad_validations=self.bad_validations,
            validation_metrics=metrics, physics_center=self.physics_center, physics_scale=self.physics_scale,
        )

    def train_loop(self, resume: str | None = None) -> None:
        self.resume(resume)
        opt = self.cfg["optimization"]
        val_every = int(self.cfg.get("logging", {}).get("val_every", 1))
        if val_every < 1:
            raise ValueError("logging.val_every must be positive")
        for epoch in range(self.start_epoch, opt["epochs"]):
            self.train_set.set_epoch(epoch)
            self.model.train()
            totals: dict[str, float] = {}
            samples = 0
            evaluate = (epoch + 1) % val_every == 0 or epoch + 1 == opt["epochs"]
            train_metrics = TrainingCompetitionMetrics() if evaluate else None
            for batch in self.train_loader:
                batch = self._normalize(batch)
                with self.accelerator.accumulate(self.model):
                    output = self.model(batch["motion"], batch["physics"], batch["lengths"])
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
                if train_metrics is not None:
                    train_metrics.update(output, batch, self.cfg["decoder"], self.accelerator)
                size = batch["motion"].shape[0]
                samples += size
                for name, value in {"loss": loss, **parts}.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
            metrics = self.validate() if evaluate else {}
            log = {f"train/{key}": value / max(samples, 1) for key, value in totals.items()}
            if train_metrics is not None:
                log.update({f"train/{key}": value for key, value in train_metrics.compute().items()})
            log.update({f"val/{key}": value for key, value in metrics.items()})
            log["epoch"] = epoch + 1
            # Explicit W&B steps otherwise keep the row pending until the next
            # log call, delaying epoch metrics by an entire epoch.
            self.accelerator.log(log, step=self.global_step, log_kwargs={"wandb": {"commit": True}})
            if evaluate:
                score = metrics[self.BEST_METRIC]
                if score > self.best:
                    self.best = score
                    self.bad_validations = 0
                    self._save("best.pt", epoch, metrics)
                else:
                    self.bad_validations += 1
            self._save("last.pt", epoch, metrics)
            stopping = self.cfg.get("early_stopping", {})
            if evaluate and stopping.get("enabled", False) and self.bad_validations >= int(stopping.get("patience", 4)):
                break

    @torch.inference_mode()
    def validate(self) -> dict[str, float]:
        model = self.ema.model.to(self.accelerator.device).eval()
        predicted_accel, true_accel, predicted_steer, true_steer = [], [], [], []
        acceleration_masks, steering_masks = [], []
        target_speed = []
        boundary = {"0.5": [], "1.0": []}
        mae = {"acceleration": [], "speed": [], "steering_angle": [], "yaw": []}
        grid_scores = []
        grid = self.cfg.get("validation", {}).get("acceleration_threshold_grid", [-0.30, -0.25, -0.20])
        grid_predictions = {(d, a): [] for d in grid for a in [-v for v in reversed(grid)]}
        for batch in self.val_loader:
            batch = self._normalize(batch)
            if isinstance(model, Stage3MotionModel):
                output = model(batch["motion"], batch["physics"], batch["lengths"],
                               chunk_frames=int(self.cfg.get("inference", {}).get("cnn_chunk_frames", 32)))
            else:
                output = model(batch["motion"], batch["physics"], batch["lengths"])
            samples = []
            for i, length in enumerate(batch["lengths"]):
                n = int(length)
                samples.append({
                    "targets": {key: value[i, :n].detach().cpu().numpy() for key, value in batch.items()
                                if isinstance(value, torch.Tensor) and key not in {"motion", "physics", "lengths"}},
                    "output": {key: value[i, :n].float().cpu() for key, value in output.items()},
                })
            if getattr(self.accelerator, "num_processes", 1) > 1:
                samples = self.accelerator.gather_for_metrics(samples, use_gather_object=True)
            for sample in samples:
                def values(name):
                    return sample["targets"][name]
                sample_output = sample["output"]
                pa, ps = decode_predictions(sample_output, self.cfg["decoder"])
                time = values("time_valid").astype(bool)
                direct = 0.5 * (values("a_long_s1") + values("a_long_s2"))
                angle, speed = values("steering_angle"), values("speed")
                speed_valid = time & values("valid_speed").astype(bool) & np.isfinite(speed)
                direct_valid = time & values("valid_accel").astype(bool) & np.isfinite(direct)
                ta, ts, accel_valid, steer_valid = classification_targets(sample["targets"], self.cfg["decoder"])
                acfg = self.cfg["decoder"]["acceleration"]
                predicted_accel.extend(pa); true_accel.extend(ta)
                predicted_steer.extend(ps); true_steer.extend(ts)
                acceleration_masks.extend(accel_valid); steering_masks.extend(steer_valid)
                target_speed.extend(speed)
                predicted_continuous = sample_output["acceleration"].mean(-1).numpy()
                mae["acceleration"].extend(np.abs(predicted_continuous[direct_valid] - direct[direct_valid]))
                mae["speed"].extend(np.abs(sample_output["speed"].numpy()[speed_valid] - speed[speed_valid]))
                mae["steering_angle"].extend(np.abs(sample_output["steering_angle"].numpy()[steer_valid] - angle[steer_valid]))
                if "yaw_rate_aux" in sample_output:
                    yaw_target = values("yaw_rate_aux")
                    yaw_valid = time & values("valid_yaw").astype(bool) & np.isfinite(yaw_target)
                    mae["yaw"].extend(np.abs(sample_output["yaw_rate_aux"].numpy()[yaw_valid] - yaw_target[yaw_valid]))
                # Score contiguous valid runs separately: never create a boundary
                # across a missing frame or between clips.
                edges = np.diff(np.r_[False, accel_valid, False].astype(int))
                for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    for seconds in (0.5, 1.0):
                        boundary[f"{seconds:.1f}"].append(boundary_f1(pa[start:stop], ta[start:stop], round(seconds * 10)))
                for (decelerating, accelerating), predictions in grid_predictions.items():
                    decoder = {**self.cfg["decoder"], "acceleration": {**acfg, "decelerating_below": decelerating, "accelerating_above": accelerating}}
                    predictions.extend(decode_predictions(sample_output, decoder)[0])
        metrics = classification_metrics(np.asarray(predicted_accel), np.asarray(true_accel), np.asarray(predicted_steer), np.asarray(true_steer), np.asarray(acceleration_masks), np.asarray(steering_masks))
        metrics.update({f"{key}_mae": float(np.mean(value)) if value else 0.0 for key, value in mae.items()})
        metrics["direct_acceleration_mae"] = metrics["acceleration_mae"]
        for seconds, values in boundary.items():
            metrics[f"boundary_f1_{seconds}s"] = float(np.mean([value[0] for value in values])) if values else 0.0
            metrics[f"boundary_delay_{seconds}s"] = float(np.mean([value[1] for value in values])) if values else 0.0
        selected = np.asarray(acceleration_masks, bool)
        true_accel_array = np.asarray(true_accel)
        for predictions in grid_predictions.values() if selected.any() else ():
            grid_scores.append(f1_score(true_accel_array[selected], np.asarray(predictions)[selected], labels=["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"], average="macro", zero_division=0))
        metrics["acceleration_macro_f1_threshold_grid_mean"] = float(np.mean(grid_scores)) if grid_scores else 0.0
        predicted_accel_array, target_speed = np.asarray(predicted_accel), np.asarray(target_speed)
        for name, low, high in (("stopped", 0, 0.15), ("low", 0.15, 2.0), ("medium", 2.0, 8.0), ("high", 8.0, np.inf)):
            speed_selected = selected & (target_speed >= low) & (target_speed < high)
            if speed_selected.any():
                metrics[f"acceleration_macro_f1_speed_{name}"] = float(f1_score(true_accel_array[speed_selected], predicted_accel_array[speed_selected], labels=["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"], average="macro", zero_division=0))
        return metrics
