#!/bin/bash
# Exp 6: pairwise combinations of the strongest single signals (after the finalist seed extensions).
# PH = phase_study recipe ORD lambda 1 + transition 0.75 (8-seed control T_ord1_tr0.75 in phase_study/results).
until grep -q SEEDS_LANE_DONE stage2/aux_signal_experiments/logs/batches.log 2>/dev/null; do sleep 15; done
export P=10; R=stage2/aux_signal_experiments/run.sh; F="0 1"; CV="0 1 2"; PH="--phase-rep ord --w-phase 1 --w-tr 0.75"
$R X_PH_bnd2  "$F" "$CV" $PH --boundary bnd2 --bnd-width 1 --w-bnd 1
$R X_bnd2_L2  "$F" "$CV" --boundary bnd2 --bnd-width 1 --w-bnd 1 --lane ord --w-lane 1
$R X_PH_R3    "$F" "$CV" $PH --risk R3 --w-risk 1
echo COMBO_DONE >> stage2/aux_signal_experiments/logs/batches.log
