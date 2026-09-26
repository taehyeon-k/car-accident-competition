#!/bin/bash
# Old PH_phase recipe reproduced exactly (A5 loss: 1 phase + .75 tr + .25 direct + .05 mono, structured-decoder selection).
# Starts when the A0/A2 seed extension frees its 4 slots.
until grep -q SEEDS_A_DONE stage2/phase_study/logs/batches.log 2>/dev/null; do sleep 15; done
export P=4; R=stage2/phase_study/run.sh
$R PH_repro_structsel "0 1" "0 1 2 3 4 5 6 7" --w-direct 0.25 --phase-rep cat --w-phase 1 --w-tr 0.75 --w-mono 0.05 --selection structured
echo PH_REPRO_DONE >> stage2/phase_study/logs/batches.log
