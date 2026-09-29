"""Stage 2 v6 runtime: event-specific LCPyramid / AuxPyramid ensemble on frozen geometry-tuned DINOv3-S features
+ camera-shift motion fusion for COLLISION + native-frame snap (stage2/aux_signal_experiments/event_select.py, CV-ES).

* Sampling (unchanged from v1-v5): frame-count-adaptive positions from FPS-blind 192/383 normalized grids; 7x10 pooled tokens.
* Motion, per native frame t vs t-1 on 160x90 grayscale (all frames):
  - global (14-d, as v5): phase-correlation shift + aligned |diff| statistics;
  - residual (36-d, extract_residual.py): similarity camera motion from sparse Lucas-Kanade + RANSAC (phase-correlation
    fallback), residual after warping. Inputs per sampled position: max/mean since the previous sampled position.
  motion=global members get 28 dims, motion=both members get 28 + 72 = 100 dims.
* Ensemble: ENTRY = mean probability over the members of config.decoding.entry_families, COLLISION = mean over
  collision_families (+ beta * robust-z camera shift), side/evasion = mean over all members.
* Decoding: argmax over entry <= collision of log p_entry + fused log p_collision, collision snapped to the native frame with
  the largest shift within +-half sample spacing.
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
from aux_models import AuxPyramid

COLUMNS = ['ID', 'collision_frame', 'entry_frame', 'evasion_space', 'entry_side']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MW, MH = 160, 90
RESIDUAL_DIM = 36


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


def _global_pair(prev, cur, win):
    """long_context_v2_experiments/extract_motion.py: dx, dy, |shift|, response, aligned |diff| mean, 3x3 grid."""
    (dx, dy), resp = cv2.phaseCorrelate(prev, cur, win)
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned = cv2.warpAffine(cur, m, (MW, MH), borderMode=cv2.BORDER_REFLECT)
    d = np.abs(aligned - prev)[3:-3, 3:-3]
    grid = [d[i * d.shape[0] // 3:(i + 1) * d.shape[0] // 3, j * d.shape[1] // 3:(j + 1) * d.shape[1] // 3].mean()
            for i in range(3) for j in range(3)]
    return [dx, dy, np.hypot(dx, dy), resp, d.mean(), *grid]


def _sq(x): return np.log1p(100 * np.maximum(x, 0))
def _ssq(x): return np.sign(x) * np.log1p(100 * np.abs(x))


def _camera(prev, cur, win):
    """aux_signal_experiments/extract_residual.py: 2x3 cur -> prev matrix, inlier ratio, outlier fraction, outlier residual."""
    pts = cv2.goodFeaturesToTrack((prev * 255).astype(np.uint8), maxCorners=200, qualityLevel=0.01, minDistance=5)
    if pts is not None and len(pts) >= 12:
        nxt, st, _ = cv2.calcOpticalFlowPyrLK((prev * 255).astype(np.uint8), (cur * 255).astype(np.uint8), pts, None,
                                             winSize=(15, 15), maxLevel=2)
        ok = st[:, 0] == 1
        if ok.sum() >= 12:
            a, b = pts[ok, 0], nxt[ok, 0]
            m, inl = cv2.estimateAffinePartial2D(b, a, method=cv2.RANSAC, ransacReprojThreshold=1.0)
            if m is not None:
                inl = inl[:, 0].astype(bool)
                pred = b @ m[:, :2].T + m[:, 2]
                res = np.linalg.norm(pred - a, axis=1)
                out = ~inl
                return m, float(inl.mean()), float(out.mean()), float(res[out].mean()) if out.any() else 0.0
    (dx, dy), _ = cv2.phaseCorrelate(prev, cur, win)
    return np.float32([[1, 0, -dx], [0, 1, -dy]]), 0.0, 0.0, 0.0


def _residual_pair(prev, cur, win):
    feats = np.zeros(RESIDUAL_DIM, np.float32)
    m, inl, out_frac, out_res = _camera(prev, cur, win)
    aligned = cv2.warpAffine(cur, m, (MW, MH), borderMode=cv2.BORDER_REFLECT)
    d = np.abs(aligned - prev)[4:-4, 4:-4]; h, w = d.shape
    grid = [d[i * h // 5:(i + 1) * h // 5, j * w // 5:(j + 1) * w // 5].mean() for i in range(5) for j in range(5)]
    scale = float(np.hypot(m[0, 0], m[1, 0])); rot = float(np.arctan2(m[1, 0], m[0, 0]))
    feats[0:2] = _sq(np.array([d.mean(), np.percentile(d, 95)]))
    feats[2:27] = _sq(np.array(grid))
    feats[27] = _ssq(d[:, :w // 2].mean() - d[:, w // 2:].mean())
    feats[28] = _sq(d[h // 2:, w // 3:2 * w // 3].mean())
    feats[29:33] = _ssq(np.array([scale - 1, rot, m[0, 2], m[1, 2]]))
    feats[33:36] = [inl, out_frac, np.log1p(out_res)]
    return feats


def _stream(frames, pair_fn, dim):
    """Sequential over t, exactly as the training extractors. cv2.phaseCorrelate multiplies BOTH inputs by the window in place
    (OpenCV 4.10 and 5.0), so frame t enters pair t+1 already windowed; each stream therefore needs its own frame copies and
    must not be split across threads."""
    frames = [f.copy() for f in frames]
    win = cv2.createHanningWindow((MW, MH), cv2.CV_32F)
    feats = np.zeros((len(frames), dim), np.float32)
    for t in range(1, len(frames)):
        feats[t] = pair_fn(frames[t - 1], frames[t], win)
    return feats


def motion_features(paths, residual, workers=4):
    """([N, 14] raw global motion, [N, 36] residual motion or None); row 0 zeros."""
    n = len(paths)
    if n < 2:
        return np.zeros((n, 14), np.float32), np.zeros((n, RESIDUAL_DIM), np.float32) if residual else None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(_gray, paths))
        glob = pool.submit(_stream, frames, _global_pair, 14)
        res = pool.submit(_stream, frames, _residual_pair, RESIDUAL_DIM) if residual else None
        return glob.result(), res.result() if residual else None


def transform_motion(raw):
    return np.concatenate([np.abs(raw[:, :3]), raw[:, 3:4], np.log1p(100 * raw[:, 4:])], 1).astype(np.float32)


def segments(per_frame, abs_idx):
    """max and mean over native frames (previous sample, this sample]; frame indices only."""
    feats = np.zeros((len(abs_idx), 2 * per_frame.shape[1]), np.float32); prev = abs_idx[0] - 1
    for t, a in enumerate(abs_idx):
        lo = min(prev + 1, a); seg = per_frame[max(lo, 0):a + 1]
        feats[t] = np.concatenate([seg.max(0), seg.mean(0)]); prev = a
    return feats


def shift_z(shift, idx, zmax):
    half = max(1, int(np.ceil(np.median(np.diff(idx)) / 2))) if len(idx) > 1 else 1
    s = np.array([shift[max(i - half, 0):i + half + 1].max() for i in idx])
    med = np.median(s); mad = np.median(np.abs(s - med)) * 1.4826 + 1e-3
    return np.clip((s - med) / mad, 0, zmax), half


# ------------------------------------------------------------------------------------------ models
class EventEnsemble(torch.nn.Module):
    """Per-member probabilities; ENTRY / COLLISION averaged over their own family subsets, attributes over all members."""

    def __init__(self, models, specs, entry_families, collision_families):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.motion = [s['motion'] for s in specs]
        self.entry_idx = [i for i, s in enumerate(specs) if s['run'] in entry_families]
        self.collision_idx = [i for i, s in enumerate(specs) if s['run'] in collision_families]
        assert self.entry_idx and self.collision_idx

    def members(self, x, valid, motion):
        return [m(x, valid, motion=motion[kind]) if kind != 'none' else m(x, valid)
                for m, kind in zip(self.models, self.motion)]

    def forward(self, x, valid, motion):
        parts = self.members(x, valid, motion)
        prob = lambda key, idx: torch.stack([parts[i][key].float().softmax(-1) for i in idx]).mean(0)
        return {'entry_logp': prob('entry_logits', self.entry_idx).clamp_min(1e-12).log(),
                'collision_logp': prob('collision_logits', self.collision_idx).clamp_min(1e-12).log(),
                'side_prob_right': prob('side_logits', range(len(parts)))[..., 1],
                'evasion_prob': torch.stack([p['evasion_logits'].float().sigmoid() for p in parts]).mean(0)}


def build_member(spec):
    if spec['arch'] == 'aux':
        return AuxPyramid(motion_dim=int(spec['motion_dim']), boundary=spec.get('boundary', 'none'),
                          entry_aux=bool(spec.get('entry_aux', False)))
    return LCPyramid(pool=spec['pool'], phase=spec['phase'], coarse=spec['coarse'], motion=spec['motion'] == 'global',
                     phase_refine=spec['phase_refine'])


def load_models(model_dir, device):
    from dinov3.hub.backbones import dinov3_vits16
    config = json.loads((model_dir / 'config.json').read_text())
    backbone = dinov3_vits16(pretrained=False)
    backbone.load_state_dict(torch.load(model_dir / 'backbone.pth', map_location='cpu', weights_only=True), strict=True)
    backbone.to(device).eval()
    members = []
    for spec in config['members']:
        model = build_member(spec)
        model.load_state_dict(torch.load(model_dir / spec['file'], map_location='cpu', weights_only=True), strict=True)
        members.append(model)
    dec = config['decoding']
    head = EventEnsemble(members, config['members'], dec['entry_families'], dec['collision_families'])
    return config, backbone, head.to(device).eval()


# ------------------------------------------------------------------------------------------ inference
def prepare(pairs, config, backbone, head, device):
    """Frame numbers, sampled positions, backbone features [1,T,70,384] and motion inputs for one clip (len >= 2)."""
    frame_numbers = np.asarray([i for i, _ in pairs], dtype=np.int64)
    chosen = select_positions(frame_numbers)
    unique, inverse = np.unique(chosen, return_inverse=True)
    features = encode(backbone, [pairs[i][1] for i in unique], device, int(config['batch_size']),
                      int(config['workers']), tuple(config['grid']))[torch.as_tensor(inverse, device=device)]
    need_residual = 'both' in head.motion
    raw, residual = motion_features([p for _, p in pairs], need_residual, int(config.get('motion_workers', 4)))
    motion = transform_motion(raw)
    glob = torch.as_tensor(segments(motion, chosen), device=device)[None]
    inputs = {'global': glob}
    if need_residual:
        inputs['both'] = torch.cat([glob, torch.as_tensor(segments(residual, chosen), device=device)[None]], -1)
    valid = torch.ones(1, len(chosen), dtype=torch.bool, device=device)
    return frame_numbers, chosen, len(unique), features[None], valid, inputs, motion[:, 2]


@torch.inference_mode()
def predict_folder(folder, config, backbone, head, device):
    pairs = indexed_frames(folder)
    if len(pairs) < 2:
        f = int(pairs[0][0])
        return dict(ID=folder.name, collision_frame=f, entry_frame=f, evasion_space=1, entry_side='LEFT'), len(pairs), 0
    frame_numbers, chosen, n_encoded, x, valid, inputs, shift = prepare(pairs, config, backbone, head, device)
    out = head(x, valid, inputs)  # FP32 heads, batch 1, as in training/evaluation.
    se = out['entry_logp'][0].cpu().numpy(); sc = out['collision_logp'][0].cpu().numpy()
    dec = config['decoding']
    z, half = shift_z(shift, chosen, float(dec['zmax']))
    sc = sc + float(dec['beta']) * z
    pair = se[:, None] + sc[None, :]
    pair = np.where(np.triu(np.ones_like(pair, bool)), pair, -np.inf)
    i, j = np.unravel_index(np.argmax(pair), pair.shape)
    entry, collision = int(frame_numbers[chosen[i]]), int(frame_numbers[chosen[j]])
    if dec.get('snap', True):
        a0 = int(chosen[j]); lo = max(a0 - half, 0)
        collision = int(frame_numbers[lo + int(np.argmax(shift[lo:a0 + half + 1]))])
        entry = min(entry, collision)
    return dict(ID=folder.name, collision_frame=collision, entry_frame=entry,
                evasion_space=int(float(out['evasion_prob'][0]) >= 0.5),
                entry_side=('LEFT', 'RIGHT')[int(float(out['side_prob_right'][0]) >= 0.5)]), len(pairs), n_encoded


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
