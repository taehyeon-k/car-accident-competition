"""Managed sequential CPU confirmation. Never modifies other queues/processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]

def live_campaign_trainers():
    result=[]
    modules={b'stage2.robust_goal_20260927.train_cpu',b'stage2.robust_goal_20260927.train_transport'}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if any(m in args for m in modules):result.append(int(p.name))
    return result

def main():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    state={'started_utc':datetime.now(timezone.utc).isoformat(),'phase':'waiting_for_current_campaign_training','jobs':{}}
    status=OUT/'confirmation_status.json'
    def save():status.write_text(json.dumps(state,indent=2)+'\n')
    save()
    while live_campaign_trainers():
        state['waiting_for_pids']=live_campaign_trainers();save();time.sleep(20)
    transport=OUT/'training/transport/cv/fold0_seed0'
    if (transport/'predictions.json').exists():
        with (OUT/'confirmation.log').open('a') as log:
            ret=subprocess.run([sys.executable,'-m','stage2.robust_goal_20260927.evaluate_training','--arms','transport'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if ret.returncode:state['phase']='transport_evaluation_failed';save();return ret.returncode
    manifest=json.loads((OUT/'training_manifest.json').read_text())
    for fold in [1,2,3,4]:
        for arm,extra in manifest['arms'].items():
            key=f'{arm}/fold{fold}_seed0';dest=OUT/'training'/arm/f'cv/fold{fold}_seed0'
            if not (dest/'predictions.json').exists():
                while live_campaign_trainers():time.sleep(20)
                cmd=[sys.executable,'-m','stage2.robust_goal_20260927.train_cpu','--run-id','RG_'+arm,'--output',str(dest),
                    '--train-split',str(OUT/f'group_folds/fold{fold}_train.jsonl'),'--val-split',str(OUT/f'group_folds/fold{fold}_val.jsonl')]+manifest['shared_args']+extra
                state['phase']='training';state['current']=key;state['jobs'][key]={'status':'running','command':cmd};save()
                with (OUT/f'train_{arm}_fold{fold}.log').open('a') as log:
                    ret=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                if ret.returncode:state['jobs'][key]['status']='failed';state['phase']='failed';save();return ret.returncode
            state['phase']='evaluating';state['current']=key;save()
            with (OUT/'confirmation.log').open('a') as log:
                ret=subprocess.run([sys.executable,'-m','stage2.robust_goal_20260927.evaluate_training','--fold',str(fold),'--arms',arm],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['jobs'][key]={'status':'complete' if ret.returncode==0 else 'evaluation_failed'};save()
            if ret.returncode:return ret.returncode
    state['phase']='complete';save();return 0

if __name__=='__main__':sys.exit(main())
