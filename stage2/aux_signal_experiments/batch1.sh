#!/bin/bash
# Exp 1 (risk progression) + Exp 4 (boundary heads), NT base = exact match of phase_study A0_d1 (8-seed control B0).
export P=10; R=stage2/aux_signal_experiments/run.sh; F="0 1"; CV="0 1 2"
$R R1_w0.5 "$F" "$CV" --risk R1 --w-risk 0.5
$R R1_w1   "$F" "$CV" --risk R1 --w-risk 1
$R R2_w1   "$F" "$CV" --risk R2 --w-risk 1
$R R3_w1   "$F" "$CV" --risk R3 --w-risk 1
for h in bnd1 bnd2 bnd3; do for w in 1 2; do
  $R B_${h}_w$w "$F" "$CV" --boundary $h --bnd-width $w --w-bnd 1
done; done
echo BATCH1_DONE >> stage2/aux_signal_experiments/logs/batches.log
