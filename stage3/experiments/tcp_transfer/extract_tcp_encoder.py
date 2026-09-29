"""Safely extract the public TCP reproduction's visual encoder tensors.

The Lightning checkpoint contains a callback object.  A harmless placeholder
is allowlisted so ``weights_only=True`` can read its tensor state without
unpickling arbitrary code from the downloaded file.
"""
from pathlib import Path

import torch


class ModelCheckpoint:
    pass


def main() -> None:
    source = Path('/workspace/pretrained/tcp_repro/best_model.ckpt')
    target = Path('/workspace/pretrained/tcp_repro/tcp_perception_state.pt')
    torch.serialization.add_safe_globals(
        [(ModelCheckpoint, 'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint')]
    )
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    prefix = 'model.perception.'
    encoder = {
        key.removeprefix(prefix): value
        for key, value in checkpoint['state_dict'].items()
        if key.startswith(prefix)
    }
    if len(encoder) != 218 or encoder['conv1.weight'].shape != (64, 3, 7, 7):
        raise ValueError('Checkpoint does not contain the expected TCP ResNet34 encoder')
    torch.save(encoder, target)
    print(f'{len(encoder)} tensors saved to {target}')


if __name__ == '__main__':
    main()
