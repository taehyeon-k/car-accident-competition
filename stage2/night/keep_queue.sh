#!/bin/bash
# restart the Stage 2 job queue whenever it is not running (it exits when momentarily idle); stops at 23:00 UTC
cd /workspace/car-accident
while [ $(date -u +%H%M) -lt 2300 ]; do
  if ! ps -eo args | awk '$1 ~ /python/ && /aux_signal_experiments\.queue/' | grep -q .; then
    nohup /venv/main/bin/python -u -m stage2.aux_signal_experiments.queue >> stage2/aux_signal_experiments/logs/queue_s2exp.out 2>&1 &
  fi
  sleep 60
done
