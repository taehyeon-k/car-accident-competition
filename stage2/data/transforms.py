from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

IMAGENET_MEAN = (0.485, 0.456, 0.406); IMAGENET_STD = (0.229, 0.224, 0.225)
@dataclass(frozen=True)
class Letterbox: scale: float; pad_x: float; pad_y: float; width: int; height: int; size: int

def letterbox(image: torch.Tensor, size: int) -> tuple[torch.Tensor, Letterbox]:
    """RGB CHW uint8/[0,1] -> normalized, aspect-preserving square."""
    _, h, w = image.shape; scale = min(size/w, size/h); nw, nh = round(w*scale), round(h*scale)
    x = F.interpolate(image.float().unsqueeze(0), size=(nh,nw), mode="bilinear", align_corners=False)[0]
    canvas = torch.tensor(IMAGENET_MEAN, device=x.device).view(3,1,1).expand(3,size,size).clone()
    px, py = (size-nw)//2, (size-nh)//2; canvas[:,py:py+nh,px:px+nw] = x
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3,1,1); std = torch.tensor(IMAGENET_STD, device=x.device).view(3,1,1)
    return (canvas-mean)/std, Letterbox(scale, px, py, w, h, size)

def box_to_grid(box: torch.Tensor, transform: Letterbox, patch: int) -> torch.Tensor:
    out = box.clone(); out[0::2] = (out[0::2] * transform.scale + transform.pad_x) / patch; out[1::2] = (out[1::2] * transform.scale + transform.pad_y) / patch
    return out
