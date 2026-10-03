"""Reconstruct tensor training inputs using the exact submitted Stage 1 preprocessing."""
import argparse
import json
from pathlib import Path
import torch
from train import runtime, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='JSONL with video_path and binary label')
    parser.add_argument('--output', required=True, help='Output folder for tensors and manifest.jsonl')
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    cfg = json.loads((ROOT / 'training/stage1/recovered_config.json').read_text())
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    rows = []
    for i,line in enumerate(manifest.read_text().splitlines()):
        if not line.strip(): continue
        row = json.loads(line); path = Path(row['video_path'])
        if not path.is_absolute(): path = manifest.parent / path
        tensors = runtime._prepare_video_inputs(path, image_size=cfg['image_size'], center_ratio=cfg['center_ratio'],
            patch_ratio=cfg['patch_ratio'], row_bins=cfg['row_bins'], global_height=cfg['global_height'],
            global_width=cfg['global_width'], residual_profiles=False, residual_sigma_fine=1.0,
            residual_sigma_coarse=2.0, residual_profile_scales=(1.0,)*5)
        name = f'{i:06d}.pt'
        torch.save({k:v.squeeze(0) for k,v in zip(('patches','global_frames','burst_profiles'), tensors)}, output / name)
        rows.append({'ID':row.get('ID', path.stem), 'tensor_path':name, 'label':int(row['label'])})
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(f'Prepared {len(rows)} clips')


if __name__ == '__main__': main()
