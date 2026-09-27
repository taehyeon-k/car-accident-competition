#!/bin/bash
# Evaluate each CV arm (seeds 0-2, 3-seed ensemble) at strides 1-4 on duplicate-clean clips as soon as its 15 checkpoints exist.
cd /workspace/car-accident; LOG=stage2/generalization/results/robust_campaign.log
for arm in "$@"; do
  fams=${arm//+/ }; need=$(( 15 * $(echo $fams | wc -w) ))
  until [ $(for f in $fams; do find stage2/generalization/results/$f/cv -name checkpoint.pt 2>/dev/null; done | wc -l) -ge $need ]; do sleep 60; done
  /venv/main/bin/python -m stage2.generalization.clean_eval robust_eval "$arm" --strides 1 2 3 4 --seeds 0 1 2 2>&1 | grep -v Warn | grep stride >> $LOG
done
