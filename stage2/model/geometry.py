from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from .tracking import Track

EPS=1e-8
def _slope(values: list[float]) -> float:
    if len(values)<2:return 0.
    x=np.arange(len(values),dtype=np.float32); return float(np.polyfit(x,np.log(np.asarray(values)+EPS),1)[0])
def build_geometry(tracks: list[Track], depth_maps: list[np.ndarray], sizes: list[tuple[int,int]], length: int, slots: int=12) -> tuple[np.ndarray,np.ndarray]:
    """Returns native-coordinate [T,N,9] geometry and [T,N] observation mask."""
    geo=np.zeros((length,slots,9),np.float32); mask=np.zeros((length,slots),bool)
    for slot,track in enumerate(tracks[:slots]):
        history=[]
        for t,det in sorted(track.observations.items()):
            x1,y1,x2,y2=det.box; w,h=sizes[t]; area=(x2-x1)*(y2-y1)/(w*h); history.append(area)
            d=depth_maps[t]; ix1,ix2=int(x1+.25*(x2-x1)),int(x2-.25*(x2-x1)); iy1,iy2=int(y1+.55*(y2-y1)),int(y2-.1*(y2-y1))
            patch=d[max(0,iy1):min(d.shape[0],iy2),max(0,ix1):min(d.shape[1],ix2)]; q=float(np.median(patch)) if patch.size else float(np.median(d))
            mad=float(np.median(np.abs(d-np.median(d))))+EPS
            delta=0. if len(history)==1 else float(np.log(history[-1]+EPS)-np.log(history[-2]+EPS))
            geo[t,slot]=[(x1+x2)/(2*w),y2/h,(x2-x1)/w,(y2-y1)/h,area,(q-np.median(d))/mad,0.,delta,_slope(history[-4:])];mask[t,slot]=True
    # rank is within-frame and only observed tracks
    for t in range(length):
        active=np.flatnonzero(mask[t]);
        if len(active)==1: geo[t,active,6]=.5
        elif len(active)>1:
            order=active[np.argsort(geo[t,active,5])]; geo[t,order,6]=np.linspace(0,1,len(order))
    return geo,mask
def tubelet_geometry(frame: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    """Average first 7 features, take latest observed looming features."""
    b,t,n,_=frame.shape; assert t==32
    f=frame.reshape(b,16,2,n,9); m=mask.reshape(b,16,2,n); out=torch.zeros_like(f[:,:,0]); valid=m.any(2)
    denom=m.sum(2).clamp_min(1).unsqueeze(-1); out[...,:7]=(f[...,:7]*m.unsqueeze(-1)).sum(2)/denom
    latest=torch.where(m[:,:,1].unsqueeze(-1),f[:,:,1, ...,7:],f[:,:,0,...,7:]); out[...,7:]=latest
    return out,valid
class GeometryMLP(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(9,64),nn.LayerNorm(64),nn.GELU(),nn.Linear(64,128),nn.LayerNorm(128))
    def forward(self,x): return self.net(x)
