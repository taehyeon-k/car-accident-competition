"""Reproducible synchronized CPU/CUDA geometry and CNN memory benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from stage3.data.timing import decode_external_training_video
from stage3.flow import build_flow_estimator
from stage3.flow.sea_raft import resize_frames
from stage3.geometry.pipeline import build_motion_features
from stage3.model import Stage3MotionModel
from stage3.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='stage3/configs/baseline_v1_2.workspace.yaml')
    parser.add_argument('--video', required=True)
    parser.add_argument('--frames', type=int, default=96)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this benchmark')
    torch.manual_seed(42)
    cfg = load_config(args.config)
    decoded = decode_external_training_video(args.video, max_frames=args.frames)
    frames = resize_frames(decoded.frames, tuple(cfg['flow']['working_size']))
    flow = build_flow_estimator(cfg['flow'], 'cuda')
    started = perf_counter()
    flows, confidence = flow.estimate_sequence(frames)
    torch.cuda.synchronize()
    flow_seconds = perf_counter() - started
    del flow
    inputs = (flows, confidence, decoded.actual_times, cfg['calibration'], tuple(cfg['geometry']['canonical_size']), frames[0])
    # Warm both implementations; timings include transfers and feature assembly.
    reference = build_motion_features(*inputs, tracking_device='cpu')
    actual = build_motion_features(*inputs, tracking_device='cuda')
    deltas = {name: {'max_abs': float(np.max(np.abs(a - b))),
                     'mean_abs': float(np.mean(np.abs(a - b))),
                     'p99_abs': float(np.quantile(np.abs(a - b), .99))}
              for name, a, b in [('motion', actual[0], reference[0]), ('physics', actual[1], reference[1])]}
    # Require close robust aggregate features; report dense-map differences as
    # well because validity thresholds can amplify tiny solver differences.
    np.testing.assert_allclose(actual[0], reference[0], atol=.002, rtol=.002)
    np.testing.assert_allclose(actual[1], reference[1], atol=.02, rtol=.02)
    timings = {}
    for name, device, backend in [('cpu', 'cpu', 'numpy'), ('cpu_fit_cuda_tracks', 'cuda', 'numpy'), ('cuda', 'cuda', 'auto')]:
        elapsed = []
        for _ in range(args.repeats):
            torch.cuda.synchronize(); started = perf_counter()
            result = build_motion_features(*inputs, tracking_device=device, geometry_backend=backend)
            torch.cuda.synchronize(); elapsed.append(perf_counter() - started)
        timings[name] = {'median_seconds': float(np.median(elapsed)), 'runs_seconds': elapsed,
                           'sections_seconds': {k.removeprefix('timing_'): float(v[0]) for k, v in result[2].items() if k.startswith('timing_')}}
    model = Stage3MotionModel(cfg['model']).cuda().eval()
    motion = torch.from_numpy(actual[0])[None]
    physics = torch.from_numpy(actual[1])[None].cuda()
    cnn = {}
    outputs = {}
    with torch.inference_mode(), torch.backends.cudnn.flags(allow_tf32=False):
        model.encode_motion(motion[:, :min(32, len(frames))], 32)
        for name, chunk in [('full', None), ('chunked', 32)]:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            started = perf_counter()
            model_input = motion.cuda() if chunk is None else motion
            outputs[name] = {k: v.cpu() for k, v in model(model_input, physics, chunk_frames=chunk).items()}
            torch.cuda.synchronize()
            cnn[name] = {'seconds': perf_counter() - started,
                         'extra_peak_MiB': (torch.cuda.max_memory_allocated() - baseline) / 2**20}
            del model_input
    for name in outputs['full']:
        torch.testing.assert_close(outputs['full'][name], outputs['chunked'][name], atol=2e-4, rtol=2e-4)
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'frames': len(frames),
              'flow_seconds': flow_seconds, 'geometry': timings, 'geometry_difference': deltas,
              'geometry_speedup_vs_cpu': timings['cpu']['median_seconds'] / timings['cuda']['median_seconds'],
              'geometry_speedup_vs_previous_cuda': timings['cpu_fit_cuda_tracks']['median_seconds'] / timings['cuda']['median_seconds'],
              'model': cnn, 'model_outputs_match': True, 'cnn_tf32': False}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
