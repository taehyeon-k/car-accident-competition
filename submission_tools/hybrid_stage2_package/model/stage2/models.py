"""Padding-invariant compact temporal heads; no FPS or source inputs."""
import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, h, dropout, dilation=1):
        super().__init__()
        self.norm = nn.LayerNorm(h)
        self.dw = nn.Conv1d(h, h, 5, padding=2*dilation, dilation=dilation, groups=h)
        self.pw = nn.Conv1d(h, h, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        y = self.norm(x.transpose(1,2)).transpose(1,2) * mask
        return (x + self.drop(self.pw(F.gelu(self.dw(y))))) * mask


class Pyramid(nn.Module):
    def __init__(self, hidden=128, dropout=.35, kind='pyramid', prior=None):
        super().__init__()
        self.norm = nn.LayerNorm(384)
        self.token = nn.Linear(384,16)
        self.frame = nn.Sequential(nn.Dropout(dropout),nn.Linear(1120,hidden),nn.GELU())
        self.kind = kind
        self.blocks = nn.ModuleList(Block(hidden,dropout,d) for d in ([1]*4 if kind=='pyramid' else [1,2,4,8,16,1]))
        if kind=='pyramid': self.fusions=nn.ModuleList(Block(hidden,dropout) for _ in range(3))
        self.event=nn.Linear(hidden,2)
        self.attn=nn.Linear(hidden,1)
        self.side=nn.Linear(hidden,2)
        self.evasion=nn.Linear(hidden,1)
        self.drop=nn.Dropout(dropout)
        self.prior=prior
        if prior:
            for e in ['entry','collision']:
                self.register_buffer(e+'_prior_positions',torch.tensor(prior[e]['positions']))

    def forward(self,x,valid):
        h=self.frame(self.token(self.norm(x.float())).flatten(2)).transpose(1,2)
        mask=valid[:,None].float(); h=h*mask
        levels=[]; masks=[]
        for i,b in enumerate(self.blocks):
            if i and self.kind=='pyramid':
                denom=F.avg_pool1d(mask,2,2,ceil_mode=True)
                h=F.avg_pool1d(h,2,2,ceil_mode=True)/denom.clamp_min(1e-8)
                mask=(denom>0).float()
            h=b(h,mask); levels.append(h); masks.append(mask)
        if self.kind=='pyramid':
            for i in range(2,-1,-1):
                # Fixed factor avoids aligning interpolation to batch padding extent.
                up=h.repeat_interleave(2,-1)[...,:levels[i].shape[-1]]
                h=self.fusions[i](levels[i]+up,masks[i])
        h=h.transpose(1,2); logits=self.event(self.drop(h))
        weights=self.attn(h).squeeze(-1).masked_fill(~valid,-1e4).softmax(-1)
        pooled=torch.einsum('bt,bth->bh',weights,h)
        out=dict(entry_logits=logits[...,0].masked_fill(~valid,-1e4),collision_logits=logits[...,1].masked_fill(~valid,-1e4),side_logits=self.side(self.drop(pooled)),evasion_logits=self.evasion(self.drop(pooled)).squeeze(-1))
        if self.prior:
            pos=torch.arange(valid.shape[1],device=x.device)[None]/(valid.sum(-1,keepdim=True)-1).clamp_min(1)
            for e in ['entry','collision']:
                d=self.prior[e]; bw=d['bandwidth']; loc=getattr(self,e+'_prior_positions')
                density=torch.exp(-.5*((pos[...,None]-loc)/bw)**2).mean(-1)/(bw*2.50662827463)
                out[e+'_logits']=out[e+'_logits']+((1-d['uniform_weight'])*density+d['uniform_weight']).log()
        return out
