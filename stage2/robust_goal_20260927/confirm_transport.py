"""Sequential remaining-fold transport confirmation; reuse completed controls."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from .confirm_folds import OUT, ROOT, live_campaign_trainers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    assert json.loads((OUT/'confirmation_status.json').read_text())['phase'] == 'complete'
    assert not live_campaign_trainers(), 'Another campaign trainer is live; do not overlap.'
    for fold in range(5):
        data = json.loads((OUT/f'training/base/cv/fold{fold}_seed0/rate_eval.json').read_text())
        assert data['native_parity_mismatches'] == []
    shared = json.loads((OUT/'training_manifest.json').read_text())['shared_args']
    commands = []
    for fold in [1, 2, 3, 4]:
        dest = OUT/f'training/transport/cv/fold{fold}_seed0'
        command = [sys.executable, '-m', 'stage2.robust_goal_20260927.train_transport',
                   '--run-id', 'RG_transport', '--output', str(dest),
                   '--train-split', str(OUT/f'group_folds/fold{fold}_train.jsonl'),
                   '--val-split', str(OUT/f'group_folds/fold{fold}_val.jsonl')] + shared
        commands.append((fold, dest, command))
    if args.dry_run:
        print(json.dumps({'folds': [x[0] for x in commands], 'commands': [x[2] for x in commands]}, indent=2))
        return 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    state = {'started_utc': datetime.now(timezone.utc).isoformat(), 'phase': 'starting', 'jobs': {}}
    def save():
        temp = OUT/'transport_confirmation_status.tmp'
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(OUT/'transport_confirmation_status.json')
    save()
    for fold, dest, command in commands:
        assert not live_campaign_trainers(), 'Another campaign trainer appeared; refusing overlap.'
        key = f'fold{fold}_seed0'
        state['current'] = key
        state['jobs'][key] = {'status': 'checkpoint_available', 'command': command}
        if not (dest/'predictions.json').exists():
            state['phase'] = 'training'; state['jobs'][key] = {'status': 'running', 'command': command}; save()
            with (OUT/f'train_transport_fold{fold}.log').open('a') as log:
                ret = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if ret.returncode:
                state['phase'] = 'training_failed'; save(); return ret.returncode
        state['phase'] = 'evaluating'; save()
        with (OUT/'transport_confirmation.log').open('a') as log:
            ret = subprocess.run([sys.executable, '-m', 'stage2.robust_goal_20260927.evaluate_training',
                                  '--fold', str(fold), '--arms', 'transport'],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if ret.returncode:
            state['phase'] = 'evaluation_failed'; save(); return ret.returncode
        state['jobs'][key]['status'] = 'complete'; save()
    state['phase'] = 'complete'; save()
    return 0


if __name__ == '__main__':
    sys.exit(main())
