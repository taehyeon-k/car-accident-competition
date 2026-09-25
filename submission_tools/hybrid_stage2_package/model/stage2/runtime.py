"""FPS-blind Stage 2 hybrid: P2 for short clips, NEXAR specialist for long clips."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
sys.path.insert(0, str(HERE))
from preprocessing import indexed_frames, select_positions, encode
from pyramid_models import TemporalPyramid
from models import Pyramid
from specialist_runtime import Specialist

COLUMNS = ['ID', 'collision_frame', 'entry_frame', 'evasion_space', 'entry_side']


class P2Ensemble(torch.nn.Module):
    """The unchanged probability-averaged P2 decoder from offline validation."""
    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    @torch.inference_mode()
    def predict_features(self, features, frames):
        x = torch.as_tensor(features, device=next(self.parameters()).device)[None]
        valid = torch.ones(1, len(frames), dtype=torch.bool, device=x.device)
        parts = [member(x, valid) for member in self.members]
        event = []
        for name in ('entry', 'collision'):
            probs = torch.stack([part[name + '_logits'].float().softmax(-1) for part in parts]).mean(0)
            event.append(probs.clamp_min(1e-12).log())
        prefix, entry_at = torch.cummax(event[0], -1)
        collision = (prefix + event[1]).argmax(-1)
        entry = entry_at.gather(1, collision[:, None]).squeeze(1)
        side = torch.stack([part['side_logits'].float().softmax(-1) for part in parts]).mean(0)[0]
        evasion = torch.stack([part['evasion_logits'].float().sigmoid() for part in parts]).mean(0)[0]
        return {'collision_frame': int(frames[int(collision)]),
                'entry_frame': int(frames[int(entry)]),
                'evasion_space': int(evasion >= 0.5),
                'entry_side': ('LEFT', 'RIGHT')[int(side.argmax())]}


def load_models(model_dir, device):
    from dinov3.hub.backbones import dinov3_vits16
    config = json.loads((model_dir / 'config.json').read_text())
    backbone = dinov3_vits16(pretrained=False)
    backbone.load_state_dict(torch.load(model_dir / 'backbone.pth', map_location='cpu', weights_only=True), strict=True)
    backbone.to(device).eval()

    p2_members = []
    for name in config['p2_members']:
        state = torch.load(model_dir / name, map_location='cpu', weights_only=True)
        assert state['config']['experiment'] == 'P2'
        model = TemporalPyramid(hidden=128, token_dim=16, dropout=0.35, levels=4)
        model.load_state_dict(state['model'], strict=True)
        p2_members.append(model.to(device).eval())
    p2 = P2Ensemble(p2_members).to(device).eval()

    specialists = []
    for name in config['specialist_members']:
        state = torch.load(model_dir / name, map_location='cpu', weights_only=True)
        model = Pyramid(kind=state['kind'], dropout=state['dropout'])
        model.load_state_dict(state['model'], strict=True)
        specialists.append(model.to(device).eval())
    specialist = Specialist(specialists, config['prior'], device)
    return config, backbone, p2, specialist


@torch.inference_mode()
def predict_folder(folder, config, backbone, p2, specialist, device):
    pairs = indexed_frames(folder)
    native_frame_count = len(pairs)
    numbers = np.asarray([index for index, _ in pairs], dtype=np.int64)
    chosen = select_positions(numbers)
    unique, inverse = np.unique(chosen, return_inverse=True)
    features = encode(backbone, [pairs[index][1] for index in unique], device,
                      int(config['batch_size']), int(config['workers']), tuple(config['grid']))
    features = features[torch.as_tensor(inverse, device=device)]
    sampled_frames = numbers[chosen]
    if native_frame_count > config['routing_threshold_frames']:
        result = specialist.predict_features(features, sampled_frames)
    else:
        result = p2.predict_features(features, sampled_frames)
    return {'ID': folder.name, **result}, native_frame_count, len(unique)


def predict(data_dir, model_dir):
    cv2.setNumThreads(1)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    root = Path(data_dir).resolve()
    root = root / 'stage2' if (root / 'stage2').is_dir() else root
    images = root / 'images' if (root / 'images').is_dir() else root
    folders = sorted(path for path in images.iterdir() if path.is_dir())
    if not folders:
        raise FileNotFoundError(f'No Stage 2 image folders in {images}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config, backbone, p2, specialist = load_models(Path(model_dir), device)
    rows = []
    for folder in folders:
        row, total, encoded = predict_folder(folder, config, backbone, p2, specialist, device)
        rows.append(row)
        print(f'[stage2] {folder.name}: {total} frames, {encoded} encoded', flush=True)
    return pd.DataFrame(rows, columns=COLUMNS)
