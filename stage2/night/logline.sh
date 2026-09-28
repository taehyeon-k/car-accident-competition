#!/bin/bash
# append a timestamped (KST) line to the night research log
echo "* $(TZ=Asia/Seoul date +%H:%M) — $*" >> /workspace/car-accident/reports/stage2_night_campaign_2026-09-28.md
