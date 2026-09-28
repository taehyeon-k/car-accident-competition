#!/bin/bash
# Evaluate each ENTRY-campaign arm with the ENTRY suite as soon as its 15 CV checkpoints exist.  line = ARM|OBJ_CACHE|extra entry keys
cd /workspace/car-accident; LOG=stage2/generalization/results/entry_campaign.log; P=/venv/main/bin/python
while read -r line; do
  [ -z "$line" ] && continue
  IFS='|' read -r arm cache keys <<< "$line"; fams=${arm//+/ }; need=$(( 15 * $(echo $fams | wc -w) ))
  until [ $(for f in $fams; do find stage2/generalization/results/$f/cv -name checkpoint.pt 2>/dev/null; done | wc -l) -ge $need ]; do sleep 60; done
  export OBJ_CACHE=${cache:-stage2/objtrack/cache_objfeat}
  $P -m stage2.generalization.entry_suite "$arm" 2>&1 | grep -v Warn | grep -v "^$" >> $LOG
  for k in $keys; do $P -m stage2.generalization.entry_suite "$arm" --entry-key $k --tag ${k%_logits} 2>&1 | grep -v Warn | grep -v "^$" >> $LOG; done
done
