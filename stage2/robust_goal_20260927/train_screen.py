"""Sequential CPU-only first fold screen, with persistent logs and status."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]

def main():
    manifest=json.loads((OUT/'training_manifest.json').read_text())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    state={'started_utc':datetime.now(timezone.utc).isoformat(),'arms':{}}
    for arm,extra in manifest['arms'].items():
        dest=OUT/'training'/arm/'cv/fold0_seed0'
        if (dest/'metrics.json').exists():state['arms'][arm]={'status':'complete_existing'};continue
        cmd=[sys.executable,'-m','stage2.robust_goal_20260927.train_cpu','--run-id','RG_'+arm,'--output',str(dest),
             '--train-split',str(OUT/'group_folds/fold0_train.jsonl'),'--val-split',str(OUT/'group_folds/fold0_val.jsonl')]+manifest['shared_args']+extra
        state['arms'][arm]={'status':'running','command':cmd};(OUT/'training_status.json').write_text(json.dumps(state,indent=2)+'\n')
        with (OUT/f'train_{arm}.log').open('a') as log:
            ret=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['arms'][arm]['status']='complete' if ret.returncode==0 else 'failed'
        state['arms'][arm]['exit_code']=ret.returncode
        (OUT/'training_status.json').write_text(json.dumps(state,indent=2)+'\n')
        print(arm,state['arms'][arm]['status'],flush=True)
        if ret.returncode:return ret.returncode
    return 0

if __name__=='__main__':sys.exit(main())
