"""Sequential matched-seed CPU confirmation for one selected arm and baseline."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from .confirm_folds import OUT, ROOT, live_campaign_trainers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True, choices=['aux_sharp','aux_broad','transport'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    primary = json.loads((OUT/'confirmation_aggregate.json').read_text())
    assert primary['all_five_folds_complete']
    transport_status = json.loads((OUT/'transport_confirmation_status.json').read_text())
    if not args.dry_run:
        assert transport_status['phase'] == 'complete', 'Finish current sequential transport study first.'
        assert not live_campaign_trainers(), 'Another campaign trainer is live; refusing overlap.'
    manifest = json.loads((OUT/'training_manifest.json').read_text())
    shared = manifest['shared_args']
    assert shared[:2] == ['--seed','0']
    jobs=[]
    for seed in [1,2]:
        for fold in range(5):
            for arm in ('base', args.candidate):
                dest = OUT/f'training/{arm}/cv/fold{fold}_seed{seed}'
                module = 'train_transport' if arm == 'transport' else 'train_cpu'
                extra = [] if arm == 'transport' else manifest['arms'][arm]
                command=[sys.executable,'-m',f'stage2.robust_goal_20260927.{module}',
                         '--run-id','RG_'+arm,'--output',str(dest),
                         '--train-split',str(OUT/f'group_folds/fold{fold}_train.jsonl'),
                         '--val-split',str(OUT/f'group_folds/fold{fold}_val.jsonl')]
                command += ['--seed',str(seed)]+shared[2:]+extra
                jobs.append((seed,fold,arm,dest,command))
    if args.dry_run:
        print(json.dumps({'candidate':args.candidate,'jobs':len(jobs),'first':jobs[0][-1],'last':jobs[-1][-1]},indent=2))
        return 0
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    state={'started_utc':datetime.now(timezone.utc).isoformat(),'candidate':args.candidate,
           'phase':'starting','jobs':{}}
    status=OUT/f'extra_seeds_{args.candidate}_status.json'
    def save():
        temp=status.with_suffix('.tmp')
        temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(status)
    save()
    for seed,fold,arm,dest,command in jobs:
        assert not live_campaign_trainers(), 'Another campaign trainer appeared; refusing overlap.'
        key=f'{arm}/fold{fold}_seed{seed}'
        state['current']=key
        state['jobs'][key]={'status':'checkpoint_available' if (dest/'predictions.json').exists() else 'running',
                            'command':command}
        if not (dest/'predictions.json').exists():
            state['phase']='training';save()
            with (OUT/f'train_{arm}_fold{fold}_seed{seed}.log').open('a') as log:
                ret=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if ret.returncode:
                state['phase']='training_failed';state['jobs'][key]['status']='failed';save();return ret.returncode
        state['phase']='evaluating';save()
        with (OUT/f'extra_seeds_{args.candidate}.log').open('a') as log:
            ret=subprocess.run([sys.executable,'-m','stage2.robust_goal_20260927.evaluate_training',
                                '--fold',str(fold),'--seed',str(seed),'--arms',arm],
                               cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if ret.returncode:
            state['phase']='evaluation_failed';state['jobs'][key]['status']='evaluation_failed';save();return ret.returncode
        state['jobs'][key]['status']='complete';save()
    state['phase']='complete';save()
    return 0


if __name__=='__main__':
    sys.exit(main())
