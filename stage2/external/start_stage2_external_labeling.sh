#!/bin/bash
set -euo pipefail

cd /workspace/car-accident
install -m 0755 stage2/external/stage2_external_labeling_service.sh /opt/supervisor-scripts/stage2_external_labeling.sh
install -m 0644 stage2/external/stage2_external_labeling.supervisor.conf /etc/supervisor/conf.d/stage2_external_labeling.conf

python - <<'PY'
import yaml

path = "/etc/portal.yaml"
with open(path, encoding="utf-8") as stream:
    portal = yaml.safe_load(stream) or {"applications": {}}
portal.setdefault("applications", {})["Stage 2 External Labeling"] = {
    "hostname": "localhost",
    "external_port": 10100,
    "internal_port": 17070,
    "open_path": "/",
    "name": "Stage 2 External Labeling",
}
with open(path, "w", encoding="utf-8") as stream:
    yaml.safe_dump(portal, stream, sort_keys=False)
PY

supervisorctl reread
supervisorctl update
supervisorctl restart caddy
supervisorctl restart stage2_external_labeling
supervisorctl status stage2_external_labeling
echo "Open the Stage 2 External Labeling entry in the Vast portal (external port 10100)."
