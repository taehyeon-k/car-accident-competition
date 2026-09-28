#!/bin/bash
cd /workspace/car-accident; J=stage2/aux_signal_experiments/jobs.txt; Q=stage2/night/eval_queue.txt
until grep -q "^done" stage2/night/flowgrid_nexar.log; do sleep 30; done
for f in 0 1 2 3 4; do
  echo "FG_XN4|stage2/generalization/results/FG_XN4/cv/fold${f}_seed0|--seed 0 --train-split stage2/long_context_v2_experiments/folds/fold${f}_train.jsonl --val-split stage2/long_context_v2_experiments/folds/fold${f}_val.jsonl --motion both --stride-aug 0.5,0.25,0.25 --extra-nexar --objmotion --obj-cache stage2/night/cache_flowgrid" >> $J
done
printf 'FG_XN4|0|s0\nFG+FG_E2+FG_XN4|0|s0\n' >> $Q
