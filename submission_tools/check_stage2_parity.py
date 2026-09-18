"""Compare packaged Stage 2 preprocessing, backbone features and head with training."""
import importlib.util
import json
import sys
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
import numpy as np

ROOT = Path('/workspace/car-accident')
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('packaged_stage2', ROOT/'submission_tools/learned_stage2/runtime.py')
runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
from stage2.geometry_pretrain.downstream_probe import letterbox, TemporalProbe
from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone

torch.set_num_threads(4); cv2.setNumThreads(1)
row = json.loads(Path('/workspace/data/stage2/manifests/val.jsonl').read_text().splitlines()[0])
paths = sorted(Path(row['frames_dir']).glob('*.jpg'))[:4]
for path in paths:
    expected = np.ascontiguousarray(letterbox(cv2.imread(str(path))))
    assert np.array_equal(runtime.letterbox(path).permute(1,2,0).numpy(), expected)
backbone = DinoBackbone('vits16', str(ROOT/'submission_tools/learned_stage2/backbone.pth')).cuda().eval()
with torch.inference_mode():
    actual = runtime.encode(backbone.model, list(enumerate(paths)), torch.device('cuda'), 4, 2)[0]
    x = torch.stack([runtime.letterbox(p) for p in paths]).cuda().float()
    x = (x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]*255)/ (torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]*255)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        ref = backbone(x)['patch']
    ref = F.adaptive_avg_pool2d(ref.float(),(7,10)).flatten(2).transpose(1,2).half()
    diff = (actual-ref).abs().max().item()
    torch.testing.assert_close(actual,ref,rtol=0,atol=0)
    # Same model weights and FP32 head math, no changes to decoding scores.
    state = torch.load(ROOT/'submission_tools/learned_stage2/probe.pt',map_location='cpu',weights_only=False)
    training_head = TemporalProbe().cuda().eval(); training_head.load_state_dict(state['model'])
    packaged_head = runtime.TemporalProbe().cuda().eval(); packaged_head.load_state_dict(state['model'])
    mask=torch.ones((1,len(paths)),device='cuda',dtype=torch.bool)
    a=training_head(ref[None],mask); b=packaged_head(actual[None],mask)
    for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
print(json.dumps({'preprocessing':'pixel-identical','backbone_max_abs_error':diff,'head':'bit-identical'}))
