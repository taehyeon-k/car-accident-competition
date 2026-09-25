#!/bin/bash
# Rebuild submit_v5: base = R2 submissions/2026-09-24/submit_v4_P2ens-full349_V3_steer1p5.zip (Stage 1 + V3 Stage 3 + DINOv3 vendor/backbone),
# replace model/stage2 with this directory's runtime + 15 exported members, set Stage 3 decoder overrides, zip.
set -e
cd /workspace/car-accident; source /venv/main/bin/activate >/dev/null
W=${W:-/tmp/v5_build}; rm -rf $W; mkdir -p $W
rclone copy r2:car-accident-dataset/submissions/2026-09-24/submit_v4_P2ens-full349_V3_steer1p5.zip $W
unzip -q $W/submit_v4_P2ens-full349_V3_steer1p5.zip -d $W/pkg
S2=$W/pkg/model/stage2; rm -rf $S2/members $S2/pyramid_models.py $S2/spotter_models.py $S2/runtime.py $S2/README.txt $S2/config.json
cp submission_tools/v5_stage2/{runtime.py,lc_models.py,config.json,README.txt} $S2/
cp submission_tools/v5_stage2/stage3_decoder_overrides.json $W/pkg/model/stage3/decoder_overrides.json
python - <<PY
import json, torch
from pathlib import Path
s2 = Path("$S2"); (s2 / "members").mkdir()
for m in json.loads((s2 / "config.json").read_text())["members"]:
    st = torch.load(f"stage2/long_context_v2_experiments/results/{m['run']}/{m['seed']}/checkpoint.pt", map_location="cpu", weights_only=False)
    torch.save({k: v.detach().cpu().clone() for k, v in st["model"].items()}, s2 / m["file"])
PY
find $W/pkg -name __pycache__ -prune -exec rm -rf {} +
(cd $W/pkg && zip -q -r -X ${OUT:-/workspace/outputs/submit_v5_LCv2ens4-motion_V3_acc1_steer5.zip} inference.py requirements.txt model)
