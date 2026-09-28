#!/bin/bash
# wait for 5 seed-0 fold checkpoints of each arm, then ENTRY-suite (seed 0).  args: LOG then "ARM|OBJ_CACHE|extra entry keys"
cd /workspace/car-accident; LOG=$1; shift
for line in "$@"; do
  IFS='|' read -r arm cache keys <<< "$line"
  until [ $(ls stage2/generalization/results/$arm/cv/fold*_seed0/checkpoint.pt 2>/dev/null | wc -l) -ge 5 ]; do sleep 30; done
  stage2/actor/eval_arms.sh $LOG "$line"
done
