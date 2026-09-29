#!/bin/bash
set -euo pipefail
source /venv/main/bin/activate
cd /workspace/car-accident

root=/workspace/runs/stage3_tcp_transfer
for run in C0_control T1_frozen_tcp T2_tcp_projection; do
  while [[ ! -s "$root/$run/predictions.npz" ]]; do
    if (( $(date -u +%s) >= $(date -u -d '2026-09-28 23:00:00' +%s) )); then
      echo 'Deadline reached before ensemble inputs completed.'
      exit 0
    fi
    sleep 30
  done
done

if [[ -s "$root/E1_weighted_vote/metrics.json" ]]; then
  echo 'Weighted-vote ensemble already complete.'
  exit 0
fi
exec python -u -m stage3.experiments.tcp_transfer.ensemble_vote
