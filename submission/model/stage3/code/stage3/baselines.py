from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class MajorityBaseline:
    def fit(self, acceleration: np.ndarray, steering: np.ndarray):
        self.acceleration = Counter(acceleration).most_common(1)[0][0]
        self.steering = Counter(steering).most_common(1)[0][0]
        return self

    def predict(self, physics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.full(len(physics), self.acceleration), np.full(len(physics), self.steering)


class PhysicsFormulaBaseline:
    """F1: threshold pooled rho and rotational yaw proxy without learned video features."""

    def __init__(self, decelerating: float = -0.25, accelerating: float = 0.25, steer_rate: float = 0.02):
        self.decelerating, self.accelerating, self.steer_rate = decelerating, accelerating, steer_rate

    def predict(self, physics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rho = np.nanmean(physics[:, 5:7], axis=1)
        acceleration = np.where(rho > self.accelerating, "ACCELERATING", np.where(rho < self.decelerating, "DECELERATING", "CONSTANT"))
        yaw = physics[:, 2]
        steering = np.where(yaw > self.steer_rate, "LEFT", np.where(yaw < -self.steer_rate, "RIGHT", "STRAIGHT"))
        return acceleration, steering


class PhysicsGradientBoostingBaseline:
    """F2: independent histogram gradient boosting heads on the 20-D vector."""

    def __init__(self, random_state: int = 42):
        self.acceleration = HistGradientBoostingClassifier(random_state=random_state)
        self.steering = HistGradientBoostingClassifier(random_state=random_state)

    def fit(self, physics: np.ndarray, acceleration: np.ndarray, steering: np.ndarray):
        self.acceleration.fit(physics, acceleration)
        self.steering.fit(physics, steering)
        return self

    def predict(self, physics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.acceleration.predict(physics), self.steering.predict(physics)
