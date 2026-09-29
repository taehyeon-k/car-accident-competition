#!/bin/bash
set -euo pipefail
source /venv/main/bin/activate
cd /workspace/car-accident

root=/workspace/runs/stage3_tcp_transfer
while [[ ! -s "$root/C0_control/metrics.json" || \
         ! -s "$root/T1_frozen_tcp/metrics.json" || \
         ! -s "$root/T2_tcp_projection/metrics.json" ]]; do
  if (( $(date -u +%s) >= $(date -u -d '2026-09-28 23:00:00' +%s) )); then
    echo 'Deadline reached before the screen completed.'
    exit 0
  fi
  sleep 30
done

exec python -u -m stage3.experiments.tcp_transfer.extend_best
