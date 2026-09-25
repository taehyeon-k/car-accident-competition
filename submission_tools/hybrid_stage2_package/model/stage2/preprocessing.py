"""P2 temporal-pyramid 4-seed probability ensemble on frozen geometry-tuned DINOv3-S features.

Reproduces stage2/temporal_pyramid_experiments evaluation: frame-count-adaptive sampling
(all frames for N<=128; 128/192/256/320 positions above) taken from the FPS-blind
192- or 383-position normalized grids -> 7x10 pooled patch tokens -> four P2 pyramids
-> averaged probabilities -> entry <= collision constrained decoding.
Only the number of available frames is used, for sampling density; no FPS or duration.
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
sys.path.insert(0, str(HERE))
from sampling import normalized_indices, sample_from_cache

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


def encode(backbone, paths, device, batch_size, workers, grid):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) * 255
    features = []
    # Bounded batches avoid materializing all full-resolution frames in RAM.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(paths), batch_size):
            batch = torch.stack(list(pool.map(letterbox, paths[start:start+batch_size])))
            x = (batch.to(device).float() - mean) / std
            context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
            with context:
                patch = backbone.forward_features(x)['x_norm_patchtokens']
            patch = patch.transpose(1, 2).reshape(len(batch), backbone.embed_dim, 448 // 16, 800 // 16)
            features.append(F.adaptive_avg_pool2d(patch.float(), grid).flatten(2).transpose(1, 2).half())
    return torch.cat(features)


def adaptive_count(n):
    if n <= 128: return n
    if n <= 256: return 128
    if n <= 512: return 192
    if n <= 768: return 256
    return 320


def select_positions(frame_numbers):
    """Indices into the sorted frame list, exactly as PyramidFeatures(P2, eval) selects them."""
    n = len(frame_numbers)
    count = adaptive_count(n)
    grid = normalized_indices(frame_numbers, 192 if count <= 192 else 383)
    positions = sample_from_cache(np.asarray(frame_numbers)[grid], count)
    if n <= 128:
        positions = np.unique(np.asarray(frame_numbers)[grid], return_index=True)[1]
    return grid[positions]


