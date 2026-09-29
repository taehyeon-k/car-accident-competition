"""Cache RGB-only TCP features on the existing BATON 10 Hz frame selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet34

from stage3.data.timing import decode_external_training_video
from stage3.utils.checkpoint import load_artifact
from stage3.utils.config import read_jsonl

SOURCE = Path('/workspace/pretrained/tcp_repro/tcp_perception_state.pt')
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def decode(row: dict) -> tuple[dict, np.ndarray]:
    meta = row['metadata']
    video = decode_external_training_video(
        row['video_path'], 10.0, 0.04, 0.2,
        start_time=meta.get('segment_start_time'), end_time=meta.get('segment_end_time'),
    )
    cache = load_artifact(row['cache_path'])
    actual = cache['actual_times'].numpy()
    if len(video.actual_times) != len(actual) or not np.allclose(video.actual_times, actual, atol=1e-6):
        raise ValueError(f"Frame timing mismatch: {row['clip_id']}")
    return row, np.stack(video.frames)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='/workspace/data/stage3/manifests/all.jsonl')
    parser.add_argument('--output', default='/workspace/cache/stage3/tcp_repro_rgb')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--decoders', type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # The public checkpoint exactly matches torchvision's ResNet34 tensor names.
    encoder = resnet34(weights=None).cuda().eval()
    encoder.load_state_dict(torch.load(SOURCE, map_location='cpu', weights_only=True), strict=True)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    pooled: list[torch.Tensor] = []
    encoder.avgpool.register_forward_hook(lambda _module, _input, output: pooled.append(output.flatten(1)))
    mean, std = MEAN.cuda(), STD.cuda()
    rows = [row for row in read_jsonl(args.manifest) if not (output / f"{row['clip_id']}.npz").exists()]

    def encode(frames: np.ndarray) -> np.ndarray:
        features = []
        for start in range(0, len(frames), args.batch):
            batch = torch.from_numpy(frames[start:start + args.batch]).cuda().permute(0, 3, 1, 2).float() / 255
            # Keep BATON's camera aspect ratio. The TCP ResNet is fully convolutional
            # until its adaptive pool, so a 256-pixel-high image is valid here.
            width = round(batch.shape[-1] * 256 / batch.shape[-2])
            batch = F.interpolate(batch, (256, width), mode='bilinear', align_corners=False, antialias=True)
            batch = (batch - mean) / std
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = encoder(batch)
            feature = torch.cat((pooled.pop(), logits), dim=-1)
            features.append(feature.float().half().cpu())
        return torch.cat(features).unsqueeze(1).numpy()

    def bounded(pool):
        chunk = max(1, 2 * args.decoders)
        for start in range(0, len(rows), chunk):
            yield from pool.map(decode, rows[start:start + chunk])

    with ThreadPoolExecutor(args.decoders) as pool:
        for index, (row, frames) in enumerate(bounded(pool), 1):
            tokens = encode(frames)
            flipped = encode(np.ascontiguousarray(frames[:, :, ::-1]))
            target = output / f"{row['clip_id']}.npz"
            temp = target.with_suffix('.tmp.npz')
            np.savez(temp, tokens=tokens, tokens_flip=flipped)
            os.replace(temp, target)
            if index == 1 or index % 25 == 0:
                print(f'cached {index}/{len(rows)} {row["clip_id"]} {tokens.shape}', flush=True)
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    (output / 'meta.json').write_text(json.dumps({
        'source': str(SOURCE), 'sha256': digest,
        'feature': 'TCP ResNet34 pooled layer4 (512) plus pretrained fc (1000)',
        'normalization': {'mean': MEAN.flatten().tolist(), 'std': STD.flatten().tolist()},
        'resize': 'height 256, aspect ratio preserved', 'timing': 'existing BATON PTS 10 Hz selection',
        'horizontal_flip': 'separately encoded', 'rows': len(read_jsonl(args.manifest)),
    }, indent=2))


if __name__ == '__main__':
    main()
