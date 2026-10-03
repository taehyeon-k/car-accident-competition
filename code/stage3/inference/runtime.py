"""Current Stage 3 EMA model with bundled SEA-RAFT and CUDA geometry."""
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))


def predict(data_dir, model_dir):
    import cv2
    import torch
    from stage3.inference.predictor import Stage3Predictor
    from stage3.data.adapters.dacon import DaconAdapter
    cv2.setNumThreads(1)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    checkpoint = Path(model_dir) / 'best.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Copy your completed Stage 3 best.pt to {checkpoint}')
    # Loading optimizer tensors onto CUDA wastes memory; packaged predictor loads on CPU.
    predictor = Stage3Predictor(checkpoint)
    # Optional decoder threshold overrides (post-processing only; model weights unchanged).
    overrides = Path(model_dir) / 'decoder_overrides.json'
    if overrides.is_file():
        import json
        for section, values in json.loads(overrides.read_text()).items():
            predictor.cfg['decoder'][section] = {**predictor.cfg['decoder'][section], **values}
    print(f'[stage3] decoder: {predictor.cfg["decoder"]}', flush=True)
    root = Path(data_dir).resolve()
    root = root / 'stage3' if (root / 'stage3').is_dir() else root
    tables = []
    for record in DaconAdapter(root).discover():
        accel, steer, timing = predictor.predict_video(record.video_path)
        tables.append(pd.DataFrame({'ID': record.clip_id, 'sample_index': range(len(accel)),
                                    'accel_label': accel, 'steer_label': steer}))
        print(f'[stage3] {record.clip_id}: {len(accel)} frames, {timing["total"]:.2f}s', flush=True)
    return pd.concat(tables, ignore_index=True)
