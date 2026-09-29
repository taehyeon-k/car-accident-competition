#!/bin/bash
set -euo pipefail
source /venv/main/bin/activate
cd /workspace/car-accident

control=/workspace/runs/stage3_tcp_transfer/C0_control/metrics.json
visual=/workspace/cache/stage3/tcp_repro_rgb_v2/meta.json
while [[ ! -s "$control" || ! -s "$visual" ]]; do
  sleep 30
done

run=/workspace/runs/stage3_tcp_transfer/T1_frozen_tcp
if [[ -s "$run/metrics.json" ]]; then
  echo 'T1 already complete.'
  exit 0
fi
args=(--base stage3/experiments/tcp_transfer/t1_frozen_tcp.yaml)
if [[ -s "$run/last.pt" ]]; then
  args=(--resume)
fi
exec python -m stage3.experiments.tcp_transfer.run_tcp \
  --name T1_frozen_tcp \
  "${args[@]}" \
  --root /workspace/runs/stage3_tcp_transfer
