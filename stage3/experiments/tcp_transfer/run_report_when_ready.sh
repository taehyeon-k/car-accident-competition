#!/bin/bash
set -euo pipefail
source /venv/main/bin/activate
cd /workspace/car-accident

t2=/workspace/runs/stage3_tcp_transfer/T2_tcp_projection/metrics.json
while [[ ! -s "$t2" ]]; do
  sleep 30
done

exec python -m stage3.experiments.tcp_transfer.summarize
