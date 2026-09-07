"""Manifests point at precomputed frozen features or raw sample directories.

Feature records are intentionally supported for GPU-free preparation: detector/depth outputs are
fixed and may be cached, while V-JEPA/DINO tensors must be produced live when their LoRA trains.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
class Stage2Dataset(Dataset):
    def __init__(self,manifest:str):
        with open(manifest) as f:self.rows=[json.loads(line) for line in f if line.strip()]
        if not self.rows: raise ValueError(f"Empty manifest: {manifest}")
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        if 'feature_path' not in row: raise ValueError("This trainer requires feature_path records; use scripts/cache_frozen_features.py for raw frames")
        item=torch.load(row['feature_path'],map_location='cpu',weights_only=False);item.update({k:v for k,v in row.items() if k!='feature_path'});return item
def collate(batch):
    # Fixed architecture lengths make normal stacking valid; metadata stays per-sample.
    answer={}
    for k in batch[0]:
        values=[x[k] for x in batch]
        answer[k]=torch.stack(values) if torch.is_tensor(values[0]) else values
    return answer
