#!/bin/bash
# usage: run.sh RUN_ID "FIXED_SEEDS" "CV_SEEDS" [train.py args...]
# fixed 279/70 split per seed in FIXED_SEEDS; 5-fold CV (LC-v2 folds, 349 clips) per seed in CV_SEEDS. P parallel jobs.
cd /workspace/car-accident; source /venv/main/bin/activate >/dev/null
ID=$1; SEEDS=$2; CVS=$3; shift 3
D=stage2/aux_signal_experiments; F=stage2/long_context_v2_experiments/folds; mkdir -p $D/logs
jobs=()
for s in $SEEDS; do jobs+=("--run-id $ID --seed $s $*"); done
for s in $CVS; do for k in 0 1 2 3 4; do
  jobs+=("--run-id $ID --seed $s --train-split $F/fold${k}_train.jsonl --val-split $F/fold${k}_val.jsonl --output $D/results/$ID/cv/fold${k}_seed$s $*")
done; done
printf '%s\n' "${jobs[@]}" | xargs -P ${P:-8} -I{} sh -c "OMP_NUM_THREADS=2 python -m stage2.aux_signal_experiments.train {} 2>&1 | grep -E 'RESULT|Error|Traceback|refusing|rror' " >> $D/logs/$ID.log
echo "DONE $ID" >> $D/logs/$ID.log
