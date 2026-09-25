#!/bin/bash
# usage: run_config.sh RUN_ID "SEEDS" "CV_SEEDS" [train.py args...]
# fixed 279/70 split for each seed in SEEDS; 5-fold CV over all 349 for each seed in CV_SEEDS. Parallel (P jobs).
cd /workspace/car-accident; source /venv/main/bin/activate >/dev/null
ID=$1; SEEDS=$2; CVS=$3; shift 3
D=stage2/long_context_v2_experiments; mkdir -p $D/logs
jobs=()
for s in $SEEDS; do jobs+=("--run-id $ID --seed $s $*"); done
for s in $CVS; do for k in 0 1 2 3 4; do
  jobs+=("--run-id $ID --seed $s --train-split $D/folds/fold${k}_train.jsonl --val-split $D/folds/fold${k}_val.jsonl --output $D/results/$ID/cv/fold${k}_seed$s $*")
done; done
printf '%s\n' "${jobs[@]}" | xargs -P ${P:-6} -I{} sh -c "OMP_NUM_THREADS=2 python -m stage2.long_context_v2_experiments.train {} 2>&1 | grep -E 'RESULT|Error|Traceback|refusing' " >> $D/logs/$ID.log
echo "DONE $ID" >> $D/logs/$ID.log
