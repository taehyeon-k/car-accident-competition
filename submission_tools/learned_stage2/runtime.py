"""Geometry-adapted DINOv3-S + trained small temporal probe.

Feature precision, letterbox colors, resolution and pooling match downstream_probe.py.
"""
import json
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
sys.path.insert(0, str(HERE))
from probe import TemporalProbe

COLUMNS = ['ID', 'collision_frame', 'entry_frame', 'evasion_space', 'entry_side']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def letterbox(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f'Cannot decode Stage 2 frame: {path}')
    h, w = bgr.shape[:2]
    scale = min(448 / h, 800 / w)
    nh, nw = round(h * scale), round(w * scale)
    out = np.empty((448, 800, 3), np.uint8)
    out[:] = (124, 116, 104)  # Exact training BGR padding, including its channel order.
    y, x = (448 - nh) // 2, (800 - nw) // 2
    out[y:y+nh, x:x+nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(out[:, :, ::-1])).permute(2, 0, 1)


def indexed_frames(folder):
    pairs = []
    for path in folder.iterdir():
        if path.suffix.lower() not in EXTENSIONS or not path.is_file():
            continue
        match = re.search(r'(\d+)$', path.stem)
        if match is None:
            raise ValueError(f'Frame filename must end in its original frame number: {path}')
        pairs.append((int(match.group(1)), path))
    pairs.sort(key=lambda item: item[0])
    if not pairs or len({i for i, _ in pairs}) != len(pairs):
        raise ValueError(f'Empty or duplicate frame indices in {folder}')
    return pairs


def encode(backbone, pairs, device, batch_size, workers):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) * 255
    features = []
    # Bounded batches avoid materializing all full-resolution frames in RAM.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(pairs), batch_size):
            batch = torch.stack(list(pool.map(letterbox, [p for _, p in pairs[start:start+batch_size]])))
            x = (batch.to(device).float() - mean) / std
            context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
            with context:
                patch = backbone.forward_features(x)['x_norm_patchtokens']
            patch = patch.transpose(1, 2).reshape(len(batch), 384, 28, 50)
            features.append(F.adaptive_avg_pool2d(patch.float(), (7, 10)).flatten(2).transpose(1, 2).half())
    return torch.cat(features)[None]


@torch.inference_mode()
def predict(data_dir, model_dir):
    cv2.setNumThreads(1)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    root = Path(data_dir).resolve()
    root = root / 'stage2' if (root / 'stage2').is_dir() else root
    images = root / 'images' if (root / 'images').is_dir() else root
    folders = sorted(p for p in images.iterdir() if p.is_dir())
    if not folders:
        raise FileNotFoundError(f'No Stage 2 image folders in {images}')
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'config.json').read_text())
    head_path = model_dir / 'probe.pt'
    if not head_path.is_file():
        raise FileNotFoundError(f'Missing trained temporal head: {head_path}; see submission README')
    from dinov3.hub.backbones import dinov3_vits16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = dinov3_vits16(pretrained=False)
    backbone.load_state_dict(torch.load(model_dir / 'backbone.pth', map_location='cpu', weights_only=True), strict=True)
    backbone.to(device).eval()
    state = torch.load(head_path, map_location='cpu', weights_only=False)
    head = TemporalProbe(**state.get('model_kwargs', {})).to(device).eval()
    head.load_state_dict(state['model'], strict=True)
    rows = []
    for folder in folders:
        pairs = indexed_frames(folder)
        fps = float(config['source_fps'])
        # Optional per-video metadata; never read labels or infer FPS from image count.
        metadata = folder / 'metadata.json'
        if metadata.is_file():
            fps = float(json.loads(metadata.read_text()).get('fps', fps))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'Invalid source FPS for {folder.name}: {fps}')
        stride = max(1, round(fps / 10))
        selected = pairs[::stride]
        features = encode(backbone, selected, device, int(config['batch_size']), int(config['workers']))
        valid = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
        out = head(features, valid)  # FP32 head, matching the training/evaluation probe.
        prefix_score, prefix_index = torch.cummax(out['entry_logits'], dim=-1)
        collision = (prefix_score + out['collision_logits']).argmax(-1)
        entry = prefix_index.gather(1, collision[:, None]).squeeze(1)
        rows.append(dict(ID=folder.name, collision_frame=selected[int(collision.item())][0],
                         entry_frame=selected[int(entry.item())][0],
                         evasion_space=int(out['evasion_logits'].item() >= 0),
                         entry_side=('LEFT', 'RIGHT')[int(out['side_logits'].argmax(-1).item())]))
        print(f'[stage2] {folder.name}: {len(pairs)} frames, {len(selected)} encoded', flush=True)
    return pd.DataFrame(rows, columns=COLUMNS)
