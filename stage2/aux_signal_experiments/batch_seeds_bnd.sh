#!/bin/bash
# Stage C: boundary finalist BND2 (h_t, dh_t), width 1 -> 8 CV seeds + 4 fixed seeds. Runs after the risk seed extension.
until grep -q SEEDS_RISK_DONE stage2/aux_signal_experiments/logs/batches.log 2>/dev/null; do sleep 15; done
export P=10; R=stage2/aux_signal_experiments/run.sh
$R B_bnd2_w1 "2 3" "3 4 5 6 7" --boundary bnd2 --bnd-width 1 --w-bnd 1
echo SEEDS_BND_DONE >> stage2/aux_signal_experiments/logs/batches.log
