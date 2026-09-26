#!/bin/bash
# Finalist ORD lambda=1 (R_ord1): CV seeds 3-7 (-> 8) and fixed seeds 2-3 (-> 4, same as other v5 families). After batch 4.
until grep -q BATCH4_DONE stage2/phase_study/logs/batches.log 2>/dev/null; do sleep 15; done
export P=4; R=stage2/phase_study/run.sh
$R R_ord1 "2 3" "3 4 5 6 7" --w-direct 1 --phase-rep ord --w-phase 1
echo FINAL_A_DONE >> stage2/phase_study/logs/batches.log
