from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank:int=8, alpha:int=16, dropout:float=.05):
        super().__init__(); self.base=base; self.rank=rank; self.scale=alpha/rank; self.drop=nn.Dropout(dropout)
        self.A=nn.Parameter(torch.empty(rank,base.in_features));self.B=nn.Parameter(torch.zeros(base.out_features,rank));nn.init.kaiming_uniform_(self.A,a=math.sqrt(5)); base.requires_grad_(False)
    def forward(self,x): return F.linear(x,self.base.weight,self.base.bias)+self.scale*F.linear(F.linear(self.drop(x),self.A),self.B)
    def merge(self):
        linear=nn.Linear(self.base.in_features,self.base.out_features,bias=self.base.bias is not None).to(self.base.weight.device,self.base.weight.dtype)
        linear.weight.data.copy_(self.base.weight+self.scale*(self.B@self.A));
        if self.base.bias is not None: linear.bias.data.copy_(self.base.bias)
        return linear
def _set(root:nn.Module,path:str,value:nn.Module):
    parent,name=path.rsplit('.',1) if '.' in path else ('',path); obj=root.get_submodule(parent) if parent else root; setattr(obj,name,value)
def add_lora_to_last_blocks(model:nn.Module, rank:int=8, alpha:int=16, dropout:float=.05, blocks=range(8,12)) -> list[str]:
    model.requires_grad_(False); attached=[]
    for name,module in list(model.named_modules()):
        if not isinstance(module,nn.Linear): continue
        parts=name.split('.'); block=next((int(p) for p in parts if p.isdigit()),None)
        leaf=parts[-1].lower()
        if block in blocks and any(k in leaf for k in ('qkv','query','key','value','q_proj','k_proj','v_proj','out_proj','proj')):
            _set(model,name,LoRALinear(module,rank,alpha,dropout));attached.append(name)
    if not attached: raise RuntimeError("No LoRA targets found in blocks 8-11; inspect the local backbone naming")
    return attached
def merge_lora(model:nn.Module):
    for name,module in list(model.named_modules()):
        if isinstance(module,LoRALinear): _set(model,name,module.merge())
