"""End-to-end learned branches: visual LoRA remains live during training."""
from __future__ import annotations
import torch
import torch.nn as nn
from .backbones import VJEPAAdapter,DINOAdapter
from .model import CoarseModel,FineModel
class CoarseSystem(nn.Module):
    def __init__(self,config):
        super().__init__(); self.visual=VJEPAAdapter(config['vjepa_factory'],config['vjepa_checkpoint'],config['lora_rank'],config['lora_alpha'],config['lora_dropout']);self.head=CoarseModel()
    def forward(self,batch): return self.head(self.visual(batch['coarse_rgb']),batch['boxes_grid'],batch['geometry'],batch['object_valid'].bool(),batch['bin_valid'].bool())
class FineSystem(nn.Module):
    def __init__(self,config):
        super().__init__();self.visual=DINOAdapter(config['dino_factory'],config['dino_checkpoint'],config['lora_rank'],config['lora_alpha'],config['lora_dropout']);self.head=FineModel()
    def forward(self,batch):
        b,k= batch['fine_rgb'].shape[:2]; rgb=batch['fine_rgb'].reshape(b*k,*batch['fine_rgb'].shape[2:]);global_token,dense=self.visual(rgb)
        return self.head(global_token.reshape(b,k,384),dense.reshape(b,k,24,24,384),batch['boxes_grid'],batch['geometry'],batch['object_valid'].bool(),batch['time_valid'].bool())
