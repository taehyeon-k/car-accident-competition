#!/bin/bash
cd /workspace/car-accident; J=stage2/aux_signal_experiments/jobs.txt; Q=stage2/night/eval_queue.txt
until grep -q "fold 4 done" stage2/night/kd3.log; do sleep 30; done
for f in 0 1 2 3 4; do for s in 0 1 2; do
  echo "KD3|stage2/generalization/results/KD3/cv/fold${f}_seed$s|--seed $s --train-split stage2/long_context_v2_experiments/folds/fold${f}_train.jsonl --val-split stage2/long_context_v2_experiments/folds/fold${f}_val.jsonl --motion both --stride-aug 0.5,0.25,0.25 --kd-dir stage2/night/cache_kd3 --w-kd 1.0" >> $J
done; done
printf 'KD3|012|s012\nKD3+E2_sa+XN4_sa|012|s012\n' >> $Q
