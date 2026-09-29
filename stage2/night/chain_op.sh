#!/bin/bash
cd /workspace/car-accident; J=stage2/aux_signal_experiments/jobs.txt; Q=stage2/night/eval_queue.txt
until grep -q "^done" stage2/night/op_features.log; do sleep 30; done
for f in 0 1 2 3 4; do for s in 0 1 2; do
  B="--seed $s --train-split stage2/long_context_v2_experiments/folds/fold${f}_train.jsonl --val-split stage2/long_context_v2_experiments/folds/fold${f}_val.jsonl --motion both --objmotion --obj-cache stage2/night/cache_op"
  echo "OP|stage2/generalization/results/OP/cv/fold${f}_seed$s|$B --stride-aug 0.5,0.25,0.25" >> $J
  echo "OPt|stage2/generalization/results/OPt/cv/fold${f}_seed$s|$B --stride-aug 0.25,0.25,0.5" >> $J
done; done
printf 'OP|012|s012\nOPt|012|s012\nE4t+FG2+FGk+E4_sbB+E2_sbB+OP|012|s012\nE4t+FG2+FGk+E4_sbB+E2_sbB+OPt|012|s012\n' >> $Q
