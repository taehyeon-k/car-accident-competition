"""Quality probe for the v0.11.1 two-camera openpilot teacher on BATON video.

The single BATON view is duplicated for both openpilot views.  This is a
feasibility probe, not a production teacher preprocessing contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

from stage3.data.timing import decode_external_training_video

ROOT = Path('/workspace/pretrained/openpilot_v0.11.1')


def pack(rgb: np.ndarray, letterbox: bool = False) -> np.ndarray:
    if letterbox:
        width = round(rgb.shape[1] * 256 / rgb.shape[0])
        fitted = cv2.resize(rgb, (width, 256), interpolation=cv2.INTER_LINEAR)
        left = (512 - width) // 2
        rgb = cv2.copyMakeBorder(fitted, 0, 0, left, 512 - width - left, cv2.BORDER_REPLICATE)
    else:
        rgb = cv2.resize(rgb, (512, 256), interpolation=cv2.INTER_LINEAR)
    packed = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
    height, width = 256, 512
    return np.concatenate((
        packed[:height:2, ::2], packed[1:height:2, ::2],
        packed[:height:2, 1::2], packed[1:height:2, 1::2],
        packed[height:height + height // 4].reshape(height // 2, width // 2),
        packed[height + height // 4:height + height // 2].reshape(height // 2, width // 2),
    ), axis=0).reshape(6, height // 2, width // 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--routes', nargs='+', default=['route_1', 'route_8'])
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--letterbox', action='store_true')
    parser.add_argument('--output', default='/workspace/runs/stage3_tcp_transfer/openpilot_probe.json')
    args = parser.parse_args()
    vision = ort.InferenceSession(str(ROOT / 'driving_vision.onnx'), providers=['CPUExecutionProvider'])
    policy = ort.InferenceSession(str(ROOT / 'driving_policy.onnx'), providers=['CPUExecutionProvider'])
    results = {}
    for route in args.routes:
        folder = Path('/workspace/data/stage3/BATON-Sample') / route
        decoded = decode_external_training_video(folder / 'qcamera.mp4', 10.0, .04, .2,
                                                 start_time=0, end_time=args.seconds)
        can = pd.read_csv(folder / 'vehicle_dynamics.csv', usecols=['time_s', 'vEgo', 'aEgo'])
        time = can['time_s'].to_numpy()
        actual_speed = np.interp(decoded.actual_times, time, can['vEgo'].to_numpy())
        actual_accel = np.interp(decoded.actual_times, time, can['aEgo'].to_numpy())
        packed = [pack(frame, args.letterbox) for frame in decoded.frames]
        features = []
        records = []
        for index in range(0, len(packed), 2):
            previous = packed[max(0, index - 2)]
            image = np.concatenate((previous, packed[index]), axis=0)[None]
            vision_out = vision.run(None, {'img': image, 'big_img': image})[0]
            features.append(vision_out[0, 1064:1576])
            history = np.zeros((1, 25, 512), dtype=np.float16)
            history[0, -min(len(features), 25):] = features[-25:]
            policy_out = policy.run(None, {
                'features_buffer': history,
                'desire_pulse': np.zeros((1, 25, 8), np.float16),
                'traffic_convention': np.array([[1, 0]], np.float16),
            })[0]
            plan = policy_out[0, :990].astype(np.float32).reshape(33, 30)[:, :15]
            records.append((float(decoded.actual_times[index]), float(actual_speed[index]),
                            float(actual_accel[index]), float(plan[0, 3]), float(plan[0, 6]),
                            float(plan[16, 3]), float(plan[0, 14])))
        values = np.asarray(records)
        warm = values[:, 0] >= 5.0
        valid = warm & np.isfinite(values).all(1)
        if valid.sum() < 5:
            raise ValueError('Too few finite teacher samples after warmup')
        speed_corr = np.corrcoef(values[valid, 1], values[valid, 3])[0, 1]
        accel_corr = np.corrcoef(values[valid, 2], values[valid, 4])[0, 1]
        results[route] = {
            'samples': len(records), 'post_warmup_samples': int(valid.sum()),
            'actual_speed_median': float(np.median(values[valid, 1])),
            'teacher_speed_median': float(np.median(values[valid, 3])),
            'speed_correlation': float(speed_corr) if np.isfinite(speed_corr) else None,
            'speed_mae': float(np.mean(np.abs(values[valid, 1] - values[valid, 3]))),
            'acceleration_correlation': float(accel_corr) if np.isfinite(accel_corr) else None,
            'acceleration_mae': float(np.mean(np.abs(values[valid, 2] - values[valid, 4]))),
            'teacher_future_speed_range': [float(values[valid, 5].min()), float(values[valid, 5].max())],
            'example_rows': values[valid][::max(1, valid.sum() // 8)].tolist(),
        }
        print(route, results[route], flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'adaptation': 'BATON single RGB duplicated to both openpilot cameras; '
                                 + ('aspect-preserving 512x256 letterbox' if args.letterbox else 'naive resize to 512x256'),
                                  'policy_inputs': 'zero desire pulses, right-hand traffic; 25-step 5 Hz feature history',
                                  'routes': results}, indent=2))


if __name__ == '__main__':
    main()
