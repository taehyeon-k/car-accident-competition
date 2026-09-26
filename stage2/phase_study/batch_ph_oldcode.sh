#!/bin/bash
# Old LC-v2 training code, unchanged (--loss phase), new CV seeds 2 and 3, sequential (old code holds ~4 GB of features/process).
cd /workspace/car-accident; source /venv/main/bin/activate >/dev/null
F=stage2/long_context_v2_experiments/folds; O=stage2/phase_study/results/PH_oldcode
for s in 2 3; do for k in 0 1 2 3 4; do
  OMP_NUM_THREADS=2 python -m stage2.long_context_v2_experiments.train --run-id PH_oldcode --loss phase --seed $s \
    --train-split $F/fold${k}_train.jsonl --val-split $F/fold${k}_val.jsonl --output $O/cv/fold${k}_seed$s 2>&1 | grep -E "RESULT|Error|Traceback"
done; done >> stage2/phase_study/logs/PH_oldcode.log
echo PH_OLDCODE_DONE >> stage2/phase_study/logs/batches.log
