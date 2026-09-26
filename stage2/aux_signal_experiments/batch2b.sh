#!/bin/bash
# Exp 3 rerun after the collate fix: M1 / M2 in full, plus the 2 M0 runs whose processes died (fold0_seed0, fold4_seed0).
export P=6; R=stage2/aux_signal_experiments/run.sh; F="0 1"; CV="0 1 2"; D=stage2/aux_signal_experiments
cd /workspace/car-accident; source /venv/main/bin/activate >/dev/null
for k in 0 4; do python -m stage2.aux_signal_experiments.train --run-id M0_global --seed 0 --base-loss p2 --motion global \
  --train-split stage2/long_context_v2_experiments/folds/fold${k}_train.jsonl --val-split stage2/long_context_v2_experiments/folds/fold${k}_val.jsonl \
  --output $D/results/M0_global/cv/fold${k}_seed0 2>&1 | grep -E "RESULT|rror" >> $D/logs/M0_global.log & done
$R M1_both     "$F" "$CV" --base-loss p2 --motion both
$R M2_residual "$F" "$CV" --base-loss p2 --motion residual
wait
echo BATCH2B_DONE >> $D/logs/batches.log
