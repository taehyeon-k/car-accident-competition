"""Strict local-only adapters. Factories are project-owned Python callables, never Hub downloads."""
from __future__ import annotations
import importlib
from pathlib import Path
import torch
import torch.nn as nn
from .lora import add_lora_to_last_blocks

def resolve_factory(path: str):
    if not path or ':' not in path: raise ValueError("Factory must be 'package.module:callable'")
    module,name=path.split(':',1); return getattr(importlib.import_module(module),name)
def load_local(factory_path: str, checkpoint: str, **kwargs) -> nn.Module:
    path=Path(checkpoint)
    if not path.is_file(): raise FileNotFoundError(f"Required local checkpoint is missing: {path}")
    factory=resolve_factory(factory_path); model=factory(**kwargs)
    state=torch.load(path,map_location='cpu',weights_only=False); state=state.get('state_dict',state) if isinstance(state,dict) else state
    missing,unexpected=model.load_state_dict(state,strict=False)
    if unexpected: raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:5]}")
    model.requires_grad_(False);return model
class VJEPAAdapter(nn.Module):
    def __init__(self,factory,checkpoint,rank=8,alpha=16,dropout=.05):
        super().__init__();self.encoder=load_local(factory,checkpoint);self.targets=add_lora_to_last_blocks(self.encoder,rank,alpha,dropout)
    def forward(self,x):
        out=self.encoder(x); out=out['dense'] if isinstance(out,dict) else out
        if out.ndim==5 and out.shape[1]==768: out=out.permute(0,2,3,4,1)
        if tuple(out.shape[1:]) != (16,24,24,768): raise RuntimeError(f"V-JEPA returned {tuple(out.shape)}; expected [B,16,24,24,768]")
        return out
class DINOAdapter(nn.Module):
    def __init__(self,factory,checkpoint,rank=8,alpha=16,dropout=.05):
        super().__init__();self.encoder=load_local(factory,checkpoint);self.targets=add_lora_to_last_blocks(self.encoder,rank,alpha,dropout)
    def forward(self,x):
        out=self.encoder(x); global_token,dense=(out['global'],out['dense']) if isinstance(out,dict) else out
        if tuple(global_token.shape[1:])!=(384,) or tuple(dense.shape[1:])!=(24,24,384): raise RuntimeError("DINO factory must return global [B,384] and dense [B,24,24,384]")
        return global_token,dense
class FrozenAdapter(nn.Module):
    def __init__(self,factory,checkpoint): super().__init__();self.model=load_local(factory,checkpoint);self.model.eval()
    @torch.inference_mode()
    def forward(self,*args,**kwargs): return self.model(*args,**kwargs)
