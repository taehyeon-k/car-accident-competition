#!/bin/bash
# Stage C: lane finalist L2 (ordinal intrusion score) -> 8 CV seeds + 4 fixed seeds. After the boundary seed extension.
until grep -q SEEDS_BND_DONE stage2/aux_signal_experiments/logs/batches.log 2>/dev/null; do sleep 15; done
export P=10; R=stage2/aux_signal_experiments/run.sh
$R L2_ord "2 3" "3 4 5 6 7" --lane ord --w-lane 1
echo SEEDS_LANE_DONE >> stage2/aux_signal_experiments/logs/batches.log
