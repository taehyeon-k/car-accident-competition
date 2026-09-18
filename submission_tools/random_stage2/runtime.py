"""User-requested random Stage 2 baseline, independent of FPS and clip duration."""
import hashlib
import random
import re
from pathlib import Path

import pandas as pd

COLUMNS = ['ID', 'collision_frame', 'entry_frame', 'evasion_space', 'entry_side']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def frame_indices(folder):
    indices = []
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        match = re.search(r'(\d+)$', path.stem)
        if match is None:
            raise ValueError(f'Frame filename must end in its original frame number: {path}')
        indices.append(int(match.group(1)))
    if not indices or len(set(indices)) != len(indices):
        raise ValueError(f'Empty or duplicate frame indices in {folder}')
    return sorted(indices)


def predict(data_dir, model_dir):
    root = Path(data_dir).resolve()
    root = root / 'stage2' if (root / 'stage2').is_dir() else root
    images = root / 'images' if (root / 'images').is_dir() else root
    folders = sorted(p for p in images.iterdir() if p.is_dir())
    if not folders:
        raise FileNotFoundError(f'No Stage 2 image folders in {images}')
    rows = []
    for folder in folders:
        indices = frame_indices(folder)
        # Stable per-ID randomness makes retries reproducible, independent of folder order.
        seed = int.from_bytes(hashlib.sha256(('stage2-random-v1:' + folder.name).encode()).digest()[:8], 'big')
        rng = random.Random(seed)
        entry, collision = sorted((rng.choice(indices), rng.choice(indices)))
        rows.append(dict(ID=folder.name, collision_frame=collision, entry_frame=entry,
                         evasion_space=rng.randrange(2), entry_side=rng.choice(('LEFT', 'RIGHT'))))
    return pd.DataFrame(rows, columns=COLUMNS)
