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
from pyramid_models import TemporalPyramid

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


def constrained_anchors(entry_logits, collision_logits):
    prefix, indices = torch.cummax(entry_logits, -1); collision = (prefix + collision_logits).argmax(-1)
    entry = indices.gather(1, collision[:, None]).squeeze(1); return entry, collision


def decode(outputs, batch):
    entry_i, collision_i = constrained_anchors(outputs["entry_logits"], outputs["collision_logits"])
    positions = batch["normalized_positions"].float()
    ep = positions.gather(1, entry_i[:, None]).squeeze(1); cp = positions.gather(1, collision_i[:, None]).squeeze(1)
    ep = ep.clamp(0, 1); cp = cp.clamp(0, 1); ep = torch.minimum(ep, cp)
    frames = batch["frame_numbers"].float(); start = frames[:, 0]; end = frames.gather(1, batch["time_valid"].sum(-1)[:, None] - 1).squeeze(1)
    entry_frame = (start + ep * (end - start)).round().long(); collision_frame = (start + cp * (end - start)).round().long()
    return entry_frame, collision_frame


class ProbabilityEnsemble(torch.nn.Module):
    """Equal-weight average of member probabilities, returned as logits."""

    def __init__(self, models):
        super().__init__(); self.models = torch.nn.ModuleList(models)

    def forward(self, x, valid):
        parts = [model(x, valid) for model in self.models]
        out = {}
        for event in ("entry", "collision"):
            probabilities = torch.stack([p[f"{event}_logits"].float().softmax(-1) for p in parts]).mean(0)
            out[f"{event}_logits"] = probabilities.clamp_min(1e-12).log()
        side = torch.stack([p["side_logits"].float().softmax(-1) for p in parts]).mean(0)
        evasion = torch.stack([p["evasion_logits"].float().sigmoid() for p in parts]).mean(0)
        out["side_logits"] = side.clamp_min(1e-12).log()
        out["evasion_logits"] = torch.logit(evasion.clamp(1e-6, 1 - 1e-6))
        return out


def load_models(model_dir, device):
    from dinov3.hub.backbones import dinov3_vits16
    config = json.loads((model_dir / 'config.json').read_text())
    backbone = dinov3_vits16(pretrained=False)
    backbone.load_state_dict(torch.load(model_dir / 'backbone.pth', map_location='cpu', weights_only=True), strict=True)
    backbone.to(device).eval()
    members = []
    for name in config['members']:
        state = torch.load(model_dir / name, map_location='cpu', weights_only=True)
        cfg = state['config']
        if cfg.get('experiment') != 'P2':
            raise ValueError(f'Stage 2 member {name} is not a P2 pyramid; sampling would not match')
        model = TemporalPyramid(hidden=cfg['hidden'], token_dim=cfg['token_dim'], dropout=cfg['dropout'], levels=cfg['levels'],
                                global_coarse=cfg.get('global_coarse', False), category_head=cfg.get('category_head', 'global'))
        model.load_state_dict(state['model'], strict=True)
        members.append(model)
    return config, backbone, ProbabilityEnsemble(members).to(device).eval()


@torch.inference_mode()
def predict_folder(folder, config, backbone, head, device):
    pairs = indexed_frames(folder)
    frame_numbers = np.asarray([i for i, _ in pairs], dtype=np.int64)
    chosen = select_positions(frame_numbers)
    unique, inverse = np.unique(chosen, return_inverse=True)
    features = encode(backbone, [pairs[i][1] for i in unique], device, int(config['batch_size']),
                      int(config['workers']), tuple(config['grid']))[torch.as_tensor(inverse, device=device)]
    frames = frame_numbers[chosen]
    span = max(int(frames[-1] - frames[0]), 1)
    batch = {'frame_numbers': torch.as_tensor(frames, device=device)[None],
             'normalized_positions': torch.as_tensor((frames - frames[0]).astype(np.float32) / span, device=device)[None],
             'time_valid': torch.ones(1, len(frames), dtype=torch.bool, device=device)}
    out = head(features[None], batch['time_valid'])  # FP32 heads, as in training/evaluation.
    entry, collision = decode(out, batch)
    return dict(ID=folder.name, collision_frame=int(collision.item()), entry_frame=int(entry.item()),
                evasion_space=int(out['evasion_logits'].item() >= 0),
                entry_side=('LEFT', 'RIGHT')[int(out['side_logits'].argmax(-1).item())]), len(pairs), len(unique)


def predict(data_dir, model_dir):
    cv2.setNumThreads(1)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    root = Path(data_dir).resolve()
    root = root / 'stage2' if (root / 'stage2').is_dir() else root
    images = root / 'images' if (root / 'images').is_dir() else root
    folders = sorted(p for p in images.iterdir() if p.is_dir())
    if not folders:
        raise FileNotFoundError(f'No Stage 2 image folders in {images}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config, backbone, head = load_models(Path(model_dir), device)
    rows = []
    for folder in folders:
        row, total, encoded = predict_folder(folder, config, backbone, head, device)
        rows.append(row)
        print(f'[stage2] {folder.name}: {total} frames, {encoded} encoded', flush=True)
    return pd.DataFrame(rows, columns=COLUMNS)
