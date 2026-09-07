"""Submission-side decoding utilities; model loading is deliberately local only."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from data.sampling import recover_region,sliding_windows
def merge_window_logits(windows:list[np.ndarray], logits:list[np.ndarray]) -> tuple[int,float]:
    values=defaultdict(list)
    for frames,window_logits in zip(windows,logits):
        for frame,logit in zip(frames,window_logits[:len(frames)]):values[int(frame)].append(float(logit))
    if not values:raise ValueError('No fine logits to merge')
    best=max(((np.mean(v),frame) for frame,v in values.items()),key=lambda x:x[0]);return best[1],float(best[0])
def side_label(direction_logit:np.ndarray)->str:return 'LEFT' if int(np.argmax(direction_logit))==0 else 'RIGHT'
def evasion_label(logit:float)->int:return int(1/(1+np.exp(-logit))>=.5)
