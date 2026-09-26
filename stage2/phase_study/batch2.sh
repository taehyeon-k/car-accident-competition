#!/bin/bash
# Batch 2 (Exp 2): phase representation. direct 1.0 + phase lambda 1 + attributes; no transition/mono. CAT reference = A2_d1_cat1.
until grep -q BATCH1_DONE stage2/phase_study/logs/batches.log 2>/dev/null; do sleep 15; done
export P=6; R=stage2/phase_study/run.sh; F="0 1"; CV="0 1 2"
$R R_ord1           "$F" "$CV" --w-direct 1 --phase-rep ord --w-phase 1
$R R_cat1_soft1     "$F" "$CV" --w-direct 1 --phase-rep cat --w-phase 1 --soft-width 1
$R R_cat1_soft2     "$F" "$CV" --w-direct 1 --phase-rep cat --w-phase 1 --soft-width 2
$R R_ord1_soft1     "$F" "$CV" --w-direct 1 --phase-rep ord --w-phase 1 --soft-width 1
$R R_ord1_soft2     "$F" "$CV" --w-direct 1 --phase-rep ord --w-phase 1 --soft-width 2
echo BATCH2_DONE >> stage2/phase_study/logs/batches.log
