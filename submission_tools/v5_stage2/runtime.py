"""Stage 2 long-context v2 runtime: 4-run LCPyramid probability ensemble on frozen geometry-tuned DINOv3-S features
+ camera-shift motion fusion for COLLISION + native-frame snap.

* Sampling (unchanged from v1-v4): frame-count-adaptive positions from FPS-blind 192/383 normalized grids; 7x10 pooled tokens.
* Motion: per native frame, phase-correlation global shift between consecutive frames on 160x90 grayscale (all frames).
  Used (a) as input to the motion-input members (max/mean over the native frames since the previous sampled position) and
  (b) as late fusion: collision score += beta * robust-z(max shift within +-half sample spacing); then the chosen collision
  is snapped to the native frame with the largest shift within +-half spacing.
* Decoding: argmax over pairs entry <= collision of log p_entry + fused log p_collision; attributes = ensemble-mean heads.
Only frame numbers / counts are used; no FPS, duration or timestamps. Clips are processed one at a time (batch 1).
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
from lc_models import LCPyramid

COLUMNS = ['ID', 'collision_frame', 'entry_frame', 'evasion_space', 'entry_side']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MW, MH = 160, 90

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


# ------------------------------------------------------------------------------------------ motion (identical to training)
def _gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if img is None:
        raise ValueError(f'Cannot decode Stage 2 frame: {path}')
    return cv2.resize(img, (MW, MH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def motion_features(paths, workers=4):
    """[N, 14] raw per-frame motion (row 0 zeros): dx, dy, |shift|, response, aligned |diff| mean, 3x3 grid."""
    feats = np.zeros((len(paths), 14), np.float32)
    if len(paths) < 2:
        return feats
    win = cv2.createHanningWindow((MW, MH), cv2.CV_32F)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(_gray, paths))
    for t in range(1, len(frames)):
        prev, cur = frames[t - 1], frames[t]
        (dx, dy), resp = cv2.phaseCorrelate(prev, cur, win)
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        aligned = cv2.warpAffine(cur, m, (MW, MH), borderMode=cv2.BORDER_REFLECT)
        d = np.abs(aligned - prev)[3:-3, 3:-3]
        grid = [d[i * d.shape[0] // 3:(i + 1) * d.shape[0] // 3, j * d.shape[1] // 3:(j + 1) * d.shape[1] // 3].mean()
                for i in range(3) for j in range(3)]
        feats[t] = [dx, dy, np.hypot(dx, dy), resp, d.mean(), *grid]
    return feats


def transform_motion(raw):
    return np.concatenate([np.abs(raw[:, :3]), raw[:, 3:4], np.log1p(100 * raw[:, 4:])], 1).astype(np.float32)


def motion_segments(m, abs_idx):
    feats = np.zeros((len(abs_idx), 28), np.float32); prev = abs_idx[0] - 1
    for t, a in enumerate(abs_idx):
        lo = min(prev + 1, a); seg = m[max(lo, 0):a + 1]
        feats[t] = np.concatenate([seg.max(0), seg.mean(0)]); prev = a
    return feats


def shift_z(shift, idx, zmax):
    half = max(1, int(np.ceil(np.median(np.diff(idx)) / 2))) if len(idx) > 1 else 1
    s = np.array([shift[max(i - half, 0):i + half + 1].max() for i in idx])
    med = np.median(s); mad = np.median(np.abs(s - med)) * 1.4826 + 1e-3
    return np.clip((s - med) / mad, 0, zmax), half


# ------------------------------------------------------------------------------------------ models
class ProbabilityEnsemble(torch.nn.Module):
    def __init__(self, models):
        super().__init__(); self.models = torch.nn.ModuleList(models)

    def forward(self, x, valid, motion):
        parts = [m(x, valid, motion=motion) if m.uses_motion else m(x, valid) for m in self.models]
        out = {}
        for event in ('entry', 'collision'):
            out[f'{event}_logp'] = torch.stack([p[f'{event}_logits'].float().softmax(-1) for p in parts]).mean(0).clamp_min(1e-12).log()
        out['side_prob'] = torch.stack([p['side_logits'].float().softmax(-1) for p in parts]).mean(0)
        out['evasion_prob'] = torch.stack([p['evasion_logits'].float().sigmoid() for p in parts]).mean(0)
        return out


def load_models(model_dir, device):
    from dinov3.hub.backbones import dinov3_vits16
    config = json.loads((model_dir / 'config.json').read_text())
    backbone = dinov3_vits16(pretrained=False)
    backbone.load_state_dict(torch.load(model_dir / 'backbone.pth', map_location='cpu', weights_only=True), strict=True)
    backbone.to(device).eval()
    members = []
    for spec in config['members']:
        model = LCPyramid(pool=spec['pool'], phase=spec['phase'], coarse=spec['coarse'], motion=spec['motion'],
                          phase_refine=spec['phase_refine'])
        model.load_state_dict(torch.load(model_dir / spec['file'], map_location='cpu', weights_only=True), strict=True)
        members.append(model)
    return config, backbone, ProbabilityEnsemble(members).to(device).eval()


# ------------------------------------------------------------------------------------------ inference
@torch.inference_mode()
def predict_folder(folder, config, backbone, head, device):
    pairs = indexed_frames(folder)
    frame_numbers = np.asarray([i for i, _ in pairs], dtype=np.int64)
    if len(pairs) < 2:
        return dict(ID=folder.name, collision_frame=int(frame_numbers[0]), entry_frame=int(frame_numbers[0]),
                    evasion_space=1, entry_side='LEFT'), len(pairs), 0
    chosen = select_positions(frame_numbers)
    unique, inverse = np.unique(chosen, return_inverse=True)
    features = encode(backbone, [pairs[i][1] for i in unique], device, int(config['batch_size']),
                      int(config['workers']), tuple(config['grid']))[torch.as_tensor(inverse, device=device)]
    motion = transform_motion(motion_features([p for _, p in pairs]))
    seg = torch.as_tensor(motion_segments(motion, chosen), device=device)[None]
    valid = torch.ones(1, len(chosen), dtype=torch.bool, device=device)
    out = head(features[None], valid, seg)  # FP32 heads, batch 1, as in training/evaluation.
    se = out['entry_logp'][0].cpu().numpy(); sc = out['collision_logp'][0].cpu().numpy()
    fusion = config['fusion']
    shift = motion[:, 2]
    z, half = shift_z(shift, chosen, float(fusion['zmax']))
    sc = sc + float(fusion['beta']) * z
    pair = se[:, None] + sc[None, :]
    pair = np.where(np.triu(np.ones_like(pair, bool)), pair, -np.inf)
    i, j = np.unravel_index(np.argmax(pair), pair.shape)
    entry, collision = int(frame_numbers[chosen[i]]), int(frame_numbers[chosen[j]])
    if fusion.get('snap', True):
        a0 = int(chosen[j]); lo = max(a0 - half, 0)
        collision = int(frame_numbers[lo + int(np.argmax(shift[lo:a0 + half + 1]))])
        entry = min(entry, collision)
    return dict(ID=folder.name, collision_frame=collision, entry_frame=entry,
                evasion_space=int(float(out['evasion_prob'][0]) >= 0.5),
                entry_side=('LEFT', 'RIGHT')[int(out['side_prob'][0].argmax().item())]), len(pairs), len(unique)


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
