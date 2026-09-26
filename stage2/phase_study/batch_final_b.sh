#!/bin/bash
# Second finalist ORD lambda=1 + transition 0.75: CV seeds 3-7 (-> 8), fixed seeds 2-3 (-> 4). After batch 3 frees its slots.
until grep -q BATCH3_DONE stage2/phase_study/logs/batches.log 2>/dev/null; do sleep 15; done
export P=6; R=stage2/phase_study/run.sh
$R T_ord1_tr0.75 "2 3" "3 4 5 6 7" --w-direct 1 --phase-rep ord --w-phase 1 --w-tr 0.75 --grad-cos 8
echo FINAL_B_DONE >> stage2/phase_study/logs/batches.log
