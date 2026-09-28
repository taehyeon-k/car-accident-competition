#!/bin/bash
cd /workspace/car-accident; J=stage2/aux_signal_experiments/jobs.txt; Q=stage2/night/eval_queue.txt
until grep -q "^done" stage2/night/flowgrid2.log; do sleep 30; done
FG2="--objmotion --obj-cache stage2/night/cache_flowgrid4x6"
for f in 0 1 2 3 4; do
  B="--seed 0 --train-split stage2/long_context_v2_experiments/folds/fold${f}_train.jsonl --val-split stage2/long_context_v2_experiments/folds/fold${f}_val.jsonl --motion both --stride-aug 0.5,0.25,0.25"
  echo "FG2|stage2/generalization/results/FG2/cv/fold${f}_seed0|$B $FG2" >> $J
  echo "FG2_E2|stage2/generalization/results/FG2_E2/cv/fold${f}_seed0|$B $FG2 --boundary bnd2 --w-bnd 1.0" >> $J
done
printf 'FG2|0|s0\nFG2_E2|0|s0\n' >> $Q
