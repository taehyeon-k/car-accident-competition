"""Validation-only weighted voting across the completed C0/T1/T2 screens.

This deliberately uses only already-decoded predictions from the fixed
0.4 m/s^2 and 7 degree protocol. Weights are selected on the same validation
set, so the result is exploratory and must not be presented as an unbiased
generalization estimate.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from stage3.trainer.decoder import ACCEL_LABELS, STEER_LABELS
from stage3.trainer.metrics import competition_score


ROOT = Path('/workspace/runs/stage3_tcp_transfer')
RUNS = ('C0_control', 'T1_frozen_tcp', 'T2_tcp_projection')
OUTPUT = ROOT / 'E1_weighted_vote'


def vote(predictions: np.ndarray, labels: np.ndarray, weights: tuple[int, ...]) -> np.ndarray:
    scores = np.zeros((predictions.shape[1], len(labels)), dtype=np.float32)
    for prediction, weight in zip(predictions, weights):
        for index, label in enumerate(labels):
            scores[:, index] += weight * (prediction == label)
    # Resolve an exact vote tie with C0's prediction rather than label order.
    for index, label in enumerate(labels):
        scores[:, index] += 1e-4 * (predictions[0] == label)
    return labels[scores.argmax(1)]


def optimize(predictions: np.ndarray, target: np.ndarray, mask: np.ndarray,
             labels: np.ndarray) -> tuple[dict, np.ndarray]:
    results = []
    for weights in itertools.product(range(5), repeat=len(RUNS)):
        if not any(weights):
            continue
        predicted = vote(predictions, labels, weights)
        macro = float(f1_score(target[mask], predicted[mask], labels=labels,
                               average='macro', zero_division=0))
        results.append((macro, weights, predicted))
    # Prefer less total weight and then C0 on an exact score tie.
    macro, weights, predicted = max(results, key=lambda x: (x[0], -sum(x[1]), x[1][0]))
    matrix = confusion_matrix(target[mask], predicted[mask], labels=labels)
    per_class = f1_score(target[mask], predicted[mask], labels=labels,
                         average=None, zero_division=0)
    return ({'macro_f1': macro, 'weights': dict(zip(RUNS, weights)),
             'labels': labels.tolist(), 'confusion_matrix': matrix.tolist(),
             'per_class_f1': dict(zip(labels.tolist(), map(float, per_class)))}, predicted)


def main() -> None:
    loaded = [np.load(ROOT / name / 'predictions.npz') for name in RUNS]
    reference = loaded[0]
    for other in loaded[1:]:
        for key in ('true_accel', 'true_steer', 'accel_valid', 'steer_valid'):
            if not np.array_equal(reference[key], other[key]):
                raise ValueError(f'Prediction alignment differs for {key}')
    acceleration, pa = optimize(
        np.stack([item['pred_accel'] for item in loaded]), reference['true_accel'],
        reference['accel_valid'].astype(bool), ACCEL_LABELS)
    steering, ps = optimize(
        np.stack([item['pred_steer'] for item in loaded]), reference['true_steer'],
        reference['steer_valid'].astype(bool), STEER_LABELS)
    score = competition_score(acceleration['macro_f1'], steering['macro_f1'])
    result = {
        'name': 'E1_weighted_vote',
        'selection_note': ('Integer voting weights were selected on this validation set. '
                           'This is exploratory and is not an unbiased generalization estimate.'),
        'decoder': {'acceleration_threshold_mps2': 0.4, 'steering_threshold_deg': 7.0},
        'runs': list(RUNS),
        'val': {'competition_score': score,
                'acceleration_macro_f1': acceleration['macro_f1'],
                'steering_macro_f1': steering['macro_f1']},
        'acceleration': acceleration,
        'steering': steering,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(OUTPUT / 'predictions.npz', pred_accel=pa,
                        true_accel=reference['true_accel'], pred_steer=ps,
                        true_steer=reference['true_steer'],
                        accel_valid=reference['accel_valid'],
                        steer_valid=reference['steer_valid'])
    print(json.dumps(result['val']), flush=True)


if __name__ == '__main__':
    main()
