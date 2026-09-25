"""Stage 2 NEXAR specialist on frozen geometry-DINOv3-S features.

Uses frame count for FPS-blind adaptive sampling; no source, FPS, duration,
timestamp, or labels enter the model or its decoder.
"""
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
from models import Pyramid

COLUMNS = ['ID', 'collision_frame', 'entry_frame', 'evasion_space', 'entry_side']


class Specialist:
    def __init__(self, members, prior, device):
        self.members = members
        self.prior = prior
        self.device = device

    @torch.inference_mode()
    def predict_features(self, features, frames):
        frames = np.asarray(frames, dtype=np.int64)
        x = torch.as_tensor(features, device=self.device)[None]
        valid = torch.ones(1, len(frames), dtype=torch.bool, device=self.device)
        parts = [member(x, valid) for member in self.members]
        positions = (frames - frames[0]) / max(int(frames[-1] - frames[0]), 1)
        event_logits = []
        for event in ('entry', 'collision'):
            probabilities = torch.stack([p[event + '_logits'].float().softmax(-1) for p in parts]).mean(0)[0]
            probabilities = probabilities.cpu().numpy()
            params = self.prior[event]
            bandwidth = float(params['bandwidth'])
            centers = np.asarray(params['positions'], dtype=np.float64)
            density = np.exp(-0.5 * ((positions[:, None] - centers) / bandwidth) ** 2).mean(-1)
            density /= bandwidth * np.sqrt(2 * np.pi)
            density = ((1 - params['uniform_weight']) * density + params['uniform_weight'])
            event_logits.append(np.log(np.maximum(probabilities, 1e-12)) + np.log(density))
        entry_logits, collision_logits = event_logits
        collision_index = int(np.argmax(np.maximum.accumulate(entry_logits) + collision_logits))
        entry_index = int(np.argmax(entry_logits[:collision_index + 1]))
        side = torch.stack([p['side_logits'].float().softmax(-1) for p in parts]).mean(0)[0]
        evasion = torch.stack([p['evasion_logits'].float().sigmoid() for p in parts]).mean(0)[0]
        return {'collision_frame': int(frames[collision_index]),
                'entry_frame': int(frames[entry_index]),
                'evasion_space': int(evasion >= 0.5),
                'entry_side': ('LEFT', 'RIGHT')[int(side.argmax())]}


def load_models(model_dir, device):
    from dinov3.hub.backbones import dinov3_vits16
    config = json.loads((model_dir / 'config.json').read_text())
    backbone = dinov3_vits16(pretrained=False)
    backbone.load_state_dict(torch.load(model_dir / 'backbone.pth', map_location='cpu', weights_only=True), strict=True)
    backbone.to(device).eval()
    members = []
    for member in config['members']:
        state = torch.load(model_dir / member['file'], map_location='cpu', weights_only=True)
        model = Pyramid(kind=state['kind'], dropout=state['dropout']).to(device)
        model.load_state_dict(state['model'], strict=True)
        members.append(model.eval())
    return config, backbone, Specialist(members, config['prior'], device)


@torch.inference_mode()
def predict_folder(folder, config, backbone, specialist, device):
    pairs = indexed_frames(folder)
    frame_numbers = np.asarray([index for index, _ in pairs], dtype=np.int64)
    selected = select_positions(frame_numbers)
    unique, inverse = np.unique(selected, return_inverse=True)
    features = encode(backbone, [pairs[index][1] for index in unique], device,
                      int(config['batch_size']), int(config['workers']), tuple(config['grid']))
    features = features[torch.as_tensor(inverse, device=device)]
    result = specialist.predict_features(features, frame_numbers[selected])
    return {'ID': folder.name, **result}, len(pairs), len(unique)


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
    config, backbone, specialist = load_models(Path(model_dir), device)
    rows = []
    for folder in folders:
        row, total, encoded = predict_folder(folder, config, backbone, specialist, device)
        rows.append(row)
        print(f'[stage2] {folder.name}: {total} frames, {encoded} encoded', flush=True)
    return pd.DataFrame(rows, columns=COLUMNS)
