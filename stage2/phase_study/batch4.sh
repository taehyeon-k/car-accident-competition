#!/bin/bash
# Batch 4 (Exp 4): where phase attaches. ORD hard, direct 1.0, phase 1.0 (S1 final head = R_ord1). Starts when PH_repro frees slots.
until grep -q PH_REPRO_DONE stage2/phase_study/logs/batches.log 2>/dev/null; do sleep 15; done
export P=4; R=stage2/phase_study/run.sh; F="0 1"; CV="0 1 2"; B="--w-direct 1 --phase-rep ord --w-phase 1"
$R S2_ord1_multiscale  "$F" "$CV" $B --attach multiscale
$R S3_ord1_cond_detach "$F" "$CV" $B --attach cond_detach
$R S3_ord1_cond_e2e    "$F" "$CV" $B --attach cond_e2e
echo BATCH4_DONE >> stage2/phase_study/logs/batches.log
