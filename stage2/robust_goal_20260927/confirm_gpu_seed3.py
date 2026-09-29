"""Isolated, low-priority matched GPU replication; never touches the CPU campaign."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from .confirm_folds import OUT, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(5)), choices=range(5))
    args = parser.parse_args()
    manifest = json.loads((OUT / 'training_manifest.json').read_text())
    shared = manifest['shared_args']
    assert shared[:2] == ['--seed', '0']
    train_env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                     MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    eval_env = dict(train_env, CUDA_VISIBLE_DEVICES='')
    status = OUT / 'gpu_seed3_status.json'
    state = {'started_utc': datetime.now(timezone.utc).isoformat(), 'seed': 3,
             'folds': args.folds, 'phase': 'starting', 'jobs': {}}

    def save():
        temp = status.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(status)

    save()
    for fold in args.folds:
        for arm, original in [('gpu_base', 'base'), ('gpu_aux_sharp', 'aux_sharp')]:
            key = f'{arm}/fold{fold}_seed3'
            dest = OUT / 'training' / arm / 'cv' / f'fold{fold}_seed3'
            state['current'] = key
            state['jobs'][key] = {'status': 'training'}
            state['phase'] = 'training'
            save()
            if not (dest / 'predictions.json').exists():
                command = [sys.executable, '-m', 'stage2.robust_goal_20260927.train_gpu',
                           '--run-id', 'RG_' + arm, '--output', str(dest),
                           '--train-split', str(OUT / f'group_folds/fold{fold}_train.jsonl'),
                           '--val-split', str(OUT / f'group_folds/fold{fold}_val.jsonl'),
                           '--seed', '3'] + shared[2:] + manifest['arms'][original]
                with (OUT / f'train_{arm}_fold{fold}_seed3.log').open('a') as log:
                    ret = subprocess.run(command, cwd=ROOT, env=train_env,
                                         stdout=log, stderr=subprocess.STDOUT)
                if ret.returncode:
                    state['phase'] = 'training_failed'
                    state['jobs'][key]['status'] = 'failed'
                    save()
                    return ret.returncode
            state['phase'] = 'evaluating'
            state['jobs'][key]['status'] = 'evaluating'
            save()
            with (OUT / 'gpu_seed3.log').open('a') as log:
                ret = subprocess.run([sys.executable, '-m',
                                      'stage2.robust_goal_20260927.evaluate_training',
                                      '--fold', str(fold), '--seed', '3', '--arms', arm,
                                      '--allow-adjacent-gpu-parity'],
                                     cwd=ROOT, env=eval_env,
                                     stdout=log, stderr=subprocess.STDOUT)
            if ret.returncode:
                state['phase'] = 'evaluation_failed'
                state['jobs'][key]['status'] = 'evaluation_failed'
                save()
                return ret.returncode
            state['jobs'][key]['status'] = 'complete'
            save()
    state['phase'] = 'complete'
    save()
    return 0


if __name__ == '__main__':
    sys.exit(main())
