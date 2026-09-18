"""Two-process regression worker, invoked by test_distributed_validation."""
import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace

from stage3.data.dataset import motion_collate
from stage3.trainer.trainer import Trainer
from stage3.tests.test_decoder import CFG


class Samples(Dataset):
    def __len__(self): return 5  # not divisible by two ranks * batch size two
    def __getitem__(self, index):
        n = 5 + index
        direct = [1., -1., 0., 0., 1.][index]
        stopped = index == 3
        angle = [10., -10., 0., 0., -10.][index]
        return {'clip_id': str(index), 'motion': torch.full((n, 10, 4, 4), float(index)),
                'physics': torch.zeros(n, 20), 'time_valid': torch.ones(n, dtype=torch.bool),
                'a_long_s1': torch.full((n,), direct), 'a_long_s2': torch.full((n,), direct),
                'speed': torch.full((n,), 0. if stopped else 1.), 'stopped': torch.full((n,), float(stopped)),
                'steering_angle': torch.full((n,), angle),
                'valid_accel': torch.ones(n, dtype=torch.bool), 'valid_speed': torch.ones(n, dtype=torch.bool),
                'valid_steer': torch.full((n,), not stopped)}


class Predictions(torch.nn.Module):
    def forward(self, motion, physics, lengths):
        ids = motion[:, :, 0, 0, 0].long()
        # Last clip deliberately wrong: duplicated padding must not overweight it.
        accel = torch.tensor([1., -1., 0., 0., -1.], device=motion.device)[ids]
        angle = torch.tensor([10., -10., 0., 0., 10.], device=motion.device)[ids]
        stopped = torch.where(ids == 3, 10., -10.)
        return {'acceleration': accel[..., None].expand(-1, -1, 2), 'stop_logit': stopped,
                'speed': torch.where(ids == 3, 0., 1.), 'steering_angle': angle}


def run():
    accelerator = Accelerator(cpu=True)
    loader = DataLoader(Samples(), batch_size=2, collate_fn=motion_collate)
    trainer = Trainer(accelerator, {'decoder': CFG})
    trainer.ema = SimpleNamespace(model=Predictions())
    trainer.physics_center, trainer.physics_scale = torch.zeros(20), torch.ones(20)
    trainer.val_loader = accelerator.prepare(loader)
    actual = trainer.validate()
    reference = Trainer(SimpleNamespace(device='cpu'), {'decoder': CFG})
    reference.ema = SimpleNamespace(model=Predictions())
    reference.physics_center, reference.physics_scale = torch.zeros(20), torch.ones(20)
    reference.val_loader = loader
    expected = reference.validate()
    assert actual == expected, (actual, expected)
    Path(os.environ['STAGE3_DIST_OUTPUT'], f'rank-{accelerator.process_index}.json').write_text(json.dumps(actual))
    accelerator.wait_for_everyone()


if __name__ == '__main__': run()
