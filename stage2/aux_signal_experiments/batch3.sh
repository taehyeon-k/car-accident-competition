#!/bin/bash
# Exp 2: lane-intrusion auxiliary supervision (v2 occupancy pseudo-labels, passed QA). NT base = matched control A0_d1.
until grep -q BATCH2_DONE stage2/aux_signal_experiments/logs/batches.log 2>/dev/null; do sleep 15; done
export P=10; R=stage2/aux_signal_experiments/run.sh; F="0 1"; CV="0 1 2"
$R L1_cat4   "$F" "$CV" --lane cat4 --w-lane 0.5
$R L2_ord    "$F" "$CV" --lane ord --w-lane 1
$R L3_cat4_tr "$F" "$CV" --lane cat4 --w-lane 0.5 --w-lane-tr 0.5
echo BATCH3_DONE >> stage2/aux_signal_experiments/logs/batches.log
