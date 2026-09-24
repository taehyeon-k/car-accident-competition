#!/bin/bash

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/cleanup_generic.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "Stage 2 External Labeling"

source /venv/main/bin/activate
cd /workspace/car-accident
pty python -m stage2.external.labeling_app --host 127.0.0.1 --port 17070 2>&1
