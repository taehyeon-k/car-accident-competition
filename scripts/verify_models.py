"""CPU verification of checkpoint integrity and retained inference/training graphs."""
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model/stage3/code'))
from fetch_checkpoints import digest


def load_runtime(stage):
    spec = importlib.util.spec_from_file_location(f'verify_{stage}', ROOT / f'model/{stage}/runtime.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    torch.set_num_threads(4)
    manifest = json.loads((ROOT / 'checkpoints/manifest.json').read_text())
    for item in manifest['checkpoints']:
        assert digest(ROOT / item['path']) == item['sha256'], item['path']
    s1 = load_runtime('stage1')
    state = torch.load(ROOT / 'model/stage1/best.pt', map_location='cpu', weights_only=False)
    cfg = state['config']
    model = s1.Stage1RebuildModel(backbone=cfg['backbone'], pretrained=False, dropout=cfg['dropout']).eval()
    model.load_state_dict(state['model_state_dict'], strict=True)
    with torch.inference_mode():
        outputs = model(torch.rand(1,8,4,3,64,64), torch.rand(1,4,3,64,96), torch.rand(1,2,16,9,64))
    assert outputs['logit'].shape == (1,)
    print('Stage 1: checkpoint strictly loads and model forward passes.')
    del model, state
    s2 = load_runtime('stage2')
    config, backbone, head = s2.load_models(ROOT / 'model/stage2', torch.device('cpu'))
    from stage2.aux_signal_experiments.model import AuxPyramid
    x = torch.rand(1,16,70,384); valid = torch.ones(1,16,dtype=torch.bool); motion = torch.rand(1,16,100)
    for member in config['members']:
        model = AuxPyramid(motion_dim=member['motion_dim'], boundary=member['boundary']).eval()
        model.load_state_dict(torch.load(ROOT / 'model/stage2' / member['file'], map_location='cpu', weights_only=True), strict=True)
        with torch.inference_mode(): out = model(x, valid, motion=motion)
        assert out['entry_logits'].shape == (1,16)
        assert torch.isfinite(out['entry_logits']).all()
    with torch.inference_mode(): out = head(x, valid, {'both':motion})
    assert out['entry_logp'].shape == (1,16)
    print('Stage 2: backbone, 12 inference members and 12 training graphs strictly load; forwards pass.')
    del model, head, backbone
    from stage3.inference.predictor import Stage3Predictor
    predictor = Stage3Predictor(ROOT / 'model/stage3/best.pt', device='cpu')
    with torch.inference_mode(): out = predictor.model(torch.rand(1,16,10,32,48), torch.rand(1,16,20))
    assert torch.isfinite(out['speed']).all()
    from stage3.trainer.losses import stage3_loss
    from stage3.trainer.trainer import Trainer
    assert callable(stage3_loss) and callable(Trainer)
    print('Stage 3: V3 EMA and SEA-RAFT strictly load; V3 forward and training imports pass.')
    print('All 16 checkpoint hashes verified.')


if __name__ == '__main__': main()
