#!/bin/bash
# Extra CV seeds 3-7 for the matched no-phase control (A0) and the CAT phase reference (A2); runs alongside batch 2.
export P=4; R=stage2/phase_study/run.sh
$R A0_d1      "" "3 4 5 6 7" --w-direct 1
$R A2_d1_cat1 "" "3 4 5 6 7" --w-direct 1 --phase-rep cat --w-phase 1
echo SEEDS_A_DONE >> stage2/phase_study/logs/batches.log
