#!/bin/bash
# Stage2_experiments: ENTRY-suite evaluation (seed 0) of screening arms.  args: LOG then lines "ARM|OBJ_CACHE|extra entry keys"
cd /workspace/car-accident; LOG=$1; shift; P=/venv/main/bin/python
for line in "$@"; do
  IFS='|' read -r arm cache keys <<< "$line"
  export OBJ_CACHE=$cache
  $P -m stage2.generalization.entry_suite "$arm" --seeds 0 2>&1 | grep -v Warn | grep -v "^$" >> $LOG
  for k in $keys; do $P -m stage2.generalization.entry_suite "$arm" --seeds 0 --entry-key $k --tag ${k%_logits} 2>&1 | grep -v Warn | grep -v "^$" >> $LOG; done
done
echo "EVAL_DONE $*" >> $LOG
