#!/bin/bash
# queue flow-grid arms when the flow cache is done and KD2 arms when the kd2 cache is done (seed 0 screens) + their evals
cd /workspace/car-accident; J=stage2/aux_signal_experiments/jobs.txt; Q=stage2/night/eval_queue.txt
F="--seed 0 --train-split stage2/long_context_v2_experiments/folds/fold{F}_train.jsonl --val-split stage2/long_context_v2_experiments/folds/fold{F}_val.jsonl --motion both --stride-aug 0.5,0.25,0.25"
add() { for f in 0 1 2 3 4; do echo "$1|stage2/generalization/results/$1/cv/fold${f}_seed0|${F//\{F\}/$f} $2" >> $J; done; echo "$1|0|s0" >> $Q; }
fg=0; k2=0
while [ $fg -eq 0 ] || [ $k2 -eq 0 ]; do
  if [ $fg -eq 0 ] && grep -q "^done" stage2/night/flowgrid.log; then add FG "--objmotion --obj-cache stage2/night/cache_flowgrid"; add FGk "--objmotion --obj-cache stage2/night/cache_flowgrid --kd-dir stage2/actor/cache_kd --w-kd 1.0"; fg=1; fi
  if [ $k2 -eq 0 ] && grep -q "fold 4 done" stage2/night/kd2.log; then add K2e "--kd-dir stage2/night/cache_kd2 --w-kd 1.0"; add K2ea "--kd-dir stage2/night/cache_kd2 --w-kd 1.0 --w-kd-attr 0.5"; k2=1; fi
  sleep 30
done
