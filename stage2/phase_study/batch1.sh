#!/bin/bash
# Batch 1: direct-weight confound (A0-A5) + phase-semantics controls (shuffled targets, detached head). CAT phase, lambda = 1.
export P=6; R=stage2/phase_study/run.sh; F="0 1"; CV="0 1 2"
$R A0_d1            "$F" "$CV" --w-direct 1
$R A1_d025          "$F" "$CV" --w-direct 0.25
$R A2_d1_cat1       "$F" "$CV" --w-direct 1    --phase-rep cat --w-phase 1
$R A3_d025_cat1     "$F" "$CV" --w-direct 0.25 --phase-rep cat --w-phase 1
$R A4_d1_cat1_tr_mono   "$F" "$CV" --w-direct 1    --phase-rep cat --w-phase 1 --w-tr 0.75 --w-mono 0.05
$R A5_d025_cat1_tr_mono "$F" "$CV" --w-direct 0.25 --phase-rep cat --w-phase 1 --w-tr 0.75 --w-mono 0.05
$R E1_d1_cat1_shuffled  "$F" "$CV" --w-direct 1    --phase-rep cat --w-phase 1 --phase-target shuffled
$R E1_d1_cat1_detach    "$F" "$CV" --w-direct 1    --phase-rep cat --w-phase 1 --attach detach
echo BATCH1_DONE >> stage2/phase_study/logs/batches.log
