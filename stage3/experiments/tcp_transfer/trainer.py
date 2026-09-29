"""Isolated TCP student trainer; inherits Stage 3 losses, decoder, and validation."""
from __future__ import annotations

import math
from itertools import groupby

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from stage3.data.cache import cache_key
from stage3.data.dataset import CachedMotionDataset, motion_collate
from stage3.trainer.ema import EMA
from stage3.trainer.trainer import Trainer
from stage3.utils.checkpoint import load_checkpoint

from .model import TCPFeatureStudent


class TCPTrainer(Trainer):
    def build(self) -> None:
        data = self.cfg['data']
        visual_dir = data['visual_cache_dir']
        self.train_set = CachedMotionDataset(
            data['manifest'], data['crop_frames'], True, self.cfg['seed'],
            data.get('flip_probability', 0.5), self.cfg['targets'],
            data.get('event_fraction', 0.5), data.get('event_position_margin', 8),
            cache_key(self.cfg), visual_dir, load_motion=False,
        )
        self.val_set = CachedMotionDataset(
            data['val_manifest'], data['crop_frames'], False, self.cfg['seed'],
            0, self.cfg['targets'], 0, data.get('event_position_margin', 8),
            cache_key(self.cfg), visual_dir, load_motion=False,
        )
        loader_args = dict(batch_size=data['batch_size'], num_workers=data['num_workers'],
                           collate_fn=motion_collate, pin_memory=True)
        if data['num_workers'] > 0:
            loader_args['prefetch_factor'] = int(data.get('prefetch_factor', 2))
        self.train_loader = DataLoader(self.train_set, shuffle=True, **loader_args)
        val_batch_size = int(data.get('val_batch_size', data['batch_size']))
        val_workers = int(data.get('val_num_workers', data['num_workers']))
        val_args = {'num_workers': val_workers, 'collate_fn': motion_collate,
                    'pin_memory': bool(data.get('val_pin_memory', True))}
        if val_workers > 0:
            val_args['prefetch_factor'] = int(data.get('val_prefetch_factor', data.get('prefetch_factor', 2)))
        # BiSSM is bidirectional, so a short sequence must not inherit a longer
        # padded horizon from another sample. Batch only consecutive clips with
        # the same manifest duration; order remains identical for ensembling.
        indexed = enumerate(self.val_set.rows)
        batches = []
        for _, group in groupby(indexed, key=lambda item: round(
                item[1]['metadata']['segment_end_time'] - item[1]['metadata']['segment_start_time'], 6)):
            indices = [index for index, _ in group]
            batches.extend(indices[start:start + val_batch_size]
                           for start in range(0, len(indices), val_batch_size))
        self.val_loader = DataLoader(self.val_set, batch_sampler=batches, **val_args)
        stats = torch.load(data['statistics'], map_location='cpu', weights_only=True)
        if stats.get('cache_key') != cache_key(self.cfg):
            raise ValueError('Stale physics statistics')
        self.physics_center = stats['center'].float()
        self.physics_scale = stats['scale'].float().clamp_min(1e-6)
        tune_projection = bool(self.cfg['model'].get('tcp_projection_finetune', False))
        projection_state = None
        if tune_projection:
            encoder_state = torch.load(self.cfg['model']['tcp_encoder_state'], map_location='cpu', weights_only=True)
            projection_state = {'weight': encoder_state['fc.weight'], 'bias': encoder_state['fc.bias']}
        self.model = TCPFeatureStudent(self.cfg['model'], tune_projection, projection_state)
        warm_start = self.cfg['model'].get('warm_start_from')
        if warm_start:
            state = load_checkpoint(warm_start, 'cpu')['ema_model']
            result = self.model.load_state_dict(state, strict=False)
            if set(result.missing_keys) != {'tcp_projection.weight', 'tcp_projection.bias'} or result.unexpected_keys:
                raise ValueError(f'Unexpected TCP warm-start mismatch: {result}')
        self.ema = EMA(self.model, self.cfg['optimization'].get('ema_decay', 0.999))
        opt = self.cfg['optimization']
        if tune_projection:
            main = [p for name, p in self.model.named_parameters() if not name.startswith('tcp_projection.')]
            self.optimizer = AdamW([
                {'params': main, 'lr': float(opt['learning_rate'])},
                {'params': self.model.tcp_projection.parameters(),
                 'lr': float(opt.get('pretrained_learning_rate', 1e-5))},
            ], weight_decay=float(opt['weight_decay']))
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=float(opt['learning_rate']),
                                   weight_decay=float(opt['weight_decay']))
        updates = max(1, math.ceil(len(self.train_loader) / opt['accumulation_steps']) * opt['epochs'])
        warmup = round(updates * opt.get('warmup_ratio', 0.05))

        def schedule(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(warmup, 1)
            progress = (step - warmup) / max(updates - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

        self.scheduler = LambdaLR(self.optimizer, schedule)
        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler,
        )
        self.ema.model.to(self.accelerator.device)
        self.start_epoch, self.global_step, self.best, self.bad_validations = 0, 0, -float('inf'), 0
