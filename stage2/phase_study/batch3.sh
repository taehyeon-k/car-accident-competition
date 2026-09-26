#!/bin/bash
# Batch 3 (Exp 3): best representation = ORD (hard). direct 1.0 fixed. Phase-weight sweep; transition / mono added individually.
# --grad-cos 8: cosine of direct/phase/transition gradients on shared pyramid params, 8 fixed train batches, every epoch.
export P=6; R=stage2/phase_study/run.sh; F="0 1"; CV="0 1 2"; B="--w-direct 1 --phase-rep ord --grad-cos 8"
$R L_ord0.1   "$F" "$CV" $B --w-phase 0.1
$R L_ord0.25  "$F" "$CV" $B --w-phase 0.25
$R L_ord0.5   "$F" "$CV" $B --w-phase 0.5
$R L_ord2     "$F" "$CV" $B --w-phase 2
$R L_ord1_gradlog "$F" "" $B --w-phase 1
$R T_ord1_tr0.25  "$F" "$CV" $B --w-phase 1 --w-tr 0.25
$R T_ord1_tr0.75  "$F" "$CV" $B --w-phase 1 --w-tr 0.75
$R M_ord1_mono0.05 "$F" "$CV" $B --w-phase 1 --w-mono 0.05
$R M_ord1_mono0.2  "$F" "$CV" $B --w-phase 1 --w-mono 0.2
echo BATCH3_DONE >> stage2/phase_study/logs/batches.log
