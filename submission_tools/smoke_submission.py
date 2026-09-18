"""Offline CUDA smoke test, run outside the repository to catch missing assets.

Uses a temporary copy of the submission plus a snapshot of Stage 3 best.pt.
Never installs an unfinished Stage 3 checkpoint into the deliverable.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time

import cv2
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', default='/workspace/car-accident/submission')
    parser.add_argument('--checkpoint', default='/workspace/car-accident/runs/stage3/baseline_v1_2/best.pt')
    parser.add_argument('--stages', nargs='+', default=['stage1','stage2','stage3'])
    parser.add_argument('--output', default='/workspace/car-accident/submission_tools/smoke_results.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    assert torch.cuda.is_available(), 'This smoke test requires CUDA'
    result = {'device': torch.cuda.get_device_name(), 'tests': {}}
    with tempfile.TemporaryDirectory(prefix='dacon-smoke-') as tmp:
        root = Path(tmp)
        package = root / 'submission'
        shutil.copytree(args.submission, package, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(args.checkpoint, package / 'model/stage3/best.pt')
        record = json.loads(Path('/workspace/data/stage2/manifests/val.jsonl').read_text().splitlines()[0])
        # One actual image-folder example, retaining native indices and FPS.
        folder = root / 'data/stage2/images/TEST_S2_001'
        folder.mkdir(parents=True)
        for p in sorted(Path(record['frames_dir']).glob('*.jpg')):
            shutil.copy2(p, folder / ('frame_' + p.name))
        (folder / 'metadata.json').write_text(json.dumps({'fps': record['native_fps']}))
        # Compact real-video excerpt for Stage 1/3, decoded completely by each entry point.
        video = root / 'clip.mp4'
        capture = cv2.VideoCapture(record['video_path'])
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 360))
        count = 0
        while count < 32:
            ok, frame = capture.read()
            if not ok: break
            writer.write(cv2.resize(frame, (640, 360))); count += 1
        capture.release(); writer.release()
        assert count == 32
        for stage in ('stage1', 'stage3'):
            dest = root / 'data' / stage / 'videos'; dest.mkdir(parents=True)
            shutil.copy2(video, dest / ('TEST_' + stage.upper() + '.mp4'))
        def blocked(*a, **kw):
            raise AssertionError('Inference attempted network access')
        socket.socket.connect = blocked
        socket.create_connection = blocked
        torch.hub.download_url_to_file = blocked
        spec = importlib.util.spec_from_file_location('submission_inference', package / 'inference.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for stage, columns, expected in (
            ('stage1', ['ID','answer'], 1),
            ('stage2', ['ID','collision_frame','entry_frame','evasion_space','entry_side'], 1),
            ('stage3', ['ID','sample_index','accel_label','steer_label'], 32)):
            if stage not in args.stages: continue
            torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
            df = getattr(module, 'predict_' + stage)(root/'data', package/'model')
            torch.cuda.synchronize()
            assert list(df.columns) == columns and len(df) == expected and not df.isna().any().any()
            if stage == 'stage1': assert set(df.answer) <= {'ORIGINAL','RERECORDED'}
            if stage == 'stage2':
                assert 0 <= df.entry_frame.iloc[0] <= df.collision_frame.iloc[0] < record['num_frames']
                assert set(df.entry_side) <= {'LEFT','RIGHT'} and set(df.evasion_space) <= {0,1}
            if stage == 'stage3':
                assert df.sample_index.tolist() == list(range(32))
                assert set(df.accel_label) <= {'STOPPED','CONSTANT','ACCELERATING','DECELERATING'}
                assert set(df.steer_label) <= {'LEFT','STRAIGHT','RIGHT'}
                # Resizing during decode must be pixel-identical to the old two-pass preparation.
                import av
                from stage3.data.timing import decode_dacon_stage3_video
                with av.open(str(video)) as container:
                    reference = [cv2.resize(f.to_ndarray(format='rgb24'), (336,192), interpolation=cv2.INTER_AREA)
                                 for f in container.decode(video=0)]
                actual = decode_dacon_stage3_video(video, resize_hw=(192,336)).frames
                assert all(np.array_equal(a,b) for a,b in zip(actual,reference))
                result['tests']['streamed_resize_exact'] = True
                # Ensure it really imported bundled code, not the workspace checkout.
                import stage3
                assert str(package) in stage3.__file__
            result['tests'][stage] = {'rows': len(df), 'seconds': time.perf_counter()-started,
                'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20, 'sample': df.head(2).to_dict('records')}
            print(stage, result['tests'][stage], flush=True)
        result['network_blocked'] = True
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
