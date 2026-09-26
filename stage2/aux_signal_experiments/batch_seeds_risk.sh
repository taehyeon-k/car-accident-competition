#!/bin/bash
# Stage C: risk finalists (R3 adaptive monotonic, R2 future consistency) to 8 CV seeds for the ensemble test.
until grep -q BATCH3_DONE stage2/aux_signal_experiments/logs/batches.log 2>/dev/null; do sleep 15; done
export P=10; R=stage2/aux_signal_experiments/run.sh
$R R3_w1 "" "3 4 5 6 7" --risk R3 --w-risk 1
$R R2_w1 "" "3 4 5 6 7" --risk R2 --w-risk 1
echo SEEDS_RISK_DONE >> stage2/aux_signal_experiments/logs/batches.log
