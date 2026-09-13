"""Validation-interval early stopping with resumable state."""

import math


class EarlyStopping:
    def __init__(self, settings=None, default_monitor="loss"):
        settings = settings or {}
        self.enabled = settings.get("enabled", False)
        self.monitor = settings.get("monitor", default_monitor)
        self.mode = settings.get("mode", "min" if self.monitor == "loss" else "max")
        self.patience = settings.get("patience", 4)
        self.min_delta = float(settings.get("min_delta", 0.002))
        if self.monitor not in {"loss", "competition_score"} or self.mode not in {
            "min",
            "max",
        }:
            raise ValueError(
                "Early stopping requires loss/competition_score and min/max mode"
            )
        if not isinstance(self.patience, int) or self.patience < 1:
            raise ValueError(
                "Early-stopping patience must be a positive number of validations"
            )
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("Early-stopping min_delta must be finite and nonnegative")
        self.best = None
        self.bad_validations = 0

    def update(self, metrics):
        if not self.enabled:
            return False
        value = float(metrics[self.monitor])
        improvement = self.best is None or (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if math.isfinite(value) and improvement:
            self.best = value
            self.bad_validations = 0
        else:
            self.bad_validations += 1
        return self.bad_validations >= self.patience

    def state_dict(self):
        return {
            k: getattr(self, k)
            for k in (
                "monitor",
                "mode",
                "patience",
                "min_delta",
                "best",
                "bad_validations",
            )
        }

    def load_state_dict(self, state):
        # A changed stopping policy starts a fresh counter, never reuses stale patience.
        if all(
            state.get(k) == getattr(self, k)
            for k in ("monitor", "mode", "patience", "min_delta")
        ):
            self.best = state["best"]
            self.bad_validations = state["bad_validations"]
