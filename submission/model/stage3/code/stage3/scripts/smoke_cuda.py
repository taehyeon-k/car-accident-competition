"""Exercise real SEA-RAFT caching, bf16 training, resume and CUDA inference.

All caches/checkpoints are temporary; no existing training artifacts are changed.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile

import torch
from accelerate import Accelerator

from stage3.data.adapters.baton import BatonAdapter
from stage3.data.cache import cache_record, cache_key
from stage3.inference.predictor import Stage3Predictor
from stage3.trainer.trainer import Trainer
from stage3.utils.checkpoint import load_artifact
from stage3.utils.config import load_config, seed_everything


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    cfg = load_config('stage3/configs/baseline_v1_2.workspace.yaml')
    seed_everything(42)
    record = BatonAdapter('/workspace/data/stage3/BATON-Sample').discover()[0]
    record = replace(record, metadata={**record.metadata, 'segment_start_time': 0., 'segment_end_time': 3.1})
    with tempfile.TemporaryDirectory(prefix='stage3-cuda-smoke-') as directory:
        root = Path(directory)
        cached = root / 'motion.pt'
        cache_record(record, cfg, cached, max_frames=32, device='cuda')
        artifact = load_artifact(cached)
        physics = artifact['physics'][artifact['time_valid']]
        center = physics.median(0).values
        scale = (physics - center).abs().median(0).values * 1.4826
        scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
        stats = root / 'stats.pt'
        torch.save({'center': center, 'scale': scale, 'cache_key': cache_key(cfg)}, stats)
        manifest = root / 'manifest.jsonl'
        manifest.write_text(json.dumps({'clip_id': record.clip_id, 'cache_path': str(cached)}) + '\n')
        cfg['data'].update(manifest=str(manifest), val_manifest=str(manifest), statistics=str(stats),
                           batch_size=1, crop_frames=16, num_workers=0)
        cfg['output_dir'] = str(root / 'checkpoints')
        cfg['optimization'].update(epochs=1, accumulation_steps=1, mixed_precision='bf16', warmup_ratio=0.)
        cfg['logging']['wandb']['enabled'] = False
        accelerator = Accelerator(mixed_precision='bf16')
        trainer = Trainer(accelerator, cfg)
        trainer.build()
        trainer.train_loop()
        assert trainer.global_step == 1
        metrics = trainer.validate()
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        checkpoint = Path(cfg['output_dir']) / 'last.pt'
        trainer.resume(str(checkpoint))
        assert trainer.start_epoch == 1 and trainer.global_step == 1
        predictor = Stage3Predictor(checkpoint, device='cuda')
        acceleration, steering, timing = predictor.predict_video(record.video_path, max_frames=8)
        assert len(acceleration) == len(steering) == 8
        print(json.dumps({'gpu': torch.cuda.get_device_name(), 'cached_frames': len(artifact['time_valid']),
                          'training_steps': trainer.global_step, 'mixed_precision': accelerator.mixed_precision,
                          'resume_ok': True, 'inference_rows': len(acceleration), 'inference_seconds': timing['total']}, indent=2))
        accelerator.end_training()


if __name__ == '__main__':
    main()
