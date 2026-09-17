"""Read only metadata from user checkpoints with the restricted torch loader."""
import json
from pathlib import Path

import numpy as np
import torch
from numpy._core.multiarray import _reconstruct

root = Path(__file__).resolve().parents[2]
paths = list(root.glob('runs/joint_online_lora*/best.pt'))
paths += list(root.glob('runs/joint_online_lora*/last.pt'))
paths += list(root.parent.glob('outputs/submit*/model/stage2/joint_model.pt'))
safe = [(_reconstruct, 'numpy.core.multiarray._reconstruct'), np.ndarray,
        np.dtype, type(np.dtype('uint32'))]
result = {}
with torch.serialization.safe_globals(safe):
    for path in sorted(paths):
        try:
            checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
            result[str(path.relative_to(root.parent))] = {
                key: checkpoint.get(key) for key in [
                    'epoch', 'step', 'config', 'validation_metrics', 'early_stopping',
                    'best_validation_loss', 'best_competition_score', 'source_checkpoint',
                ]
            }
        except Exception as error:
            result[str(path)] = {'error': str(error)}
print(json.dumps(result, indent=2, default=str))
