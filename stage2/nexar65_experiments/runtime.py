"""Reproducible feature-to-prediction runtime for the frozen NEXAR specialist.

Caller supplies the existing geometry-DINO adaptive frame features and their
frame numbers. This interface accepts no labels, source, FPS or timestamps.
"""
import json
from pathlib import Path
import numpy as np
import torch
from .models import Pyramid


class NexarSpecialist:
    def __init__(self, candidate_dir, device='cpu'):
        self.root=Path(candidate_dir);self.device=torch.device(device)
        cfg=json.loads((self.root/'candidate.json').read_text())
        self.prior=cfg['prior'];self.models=[]
        for member in cfg['members']:
            s=torch.load(self.root/member['file'],map_location='cpu',weights_only=True)
            m=Pyramid(kind=s['kind'],dropout=s['dropout']).to(self.device)
            m.load_state_dict(s['model']);self.models.append(m.eval())

    @torch.inference_mode()
    def predict(self, features, frame_numbers):
        x=torch.as_tensor(features).to(self.device)
        frames=np.asarray(frame_numbers,dtype=np.int64)
        if x.ndim!=3 or x.shape[0]!=len(frames):raise ValueError('Expected [T,70,384] features and T frame numbers')
        valid=torch.ones(1,len(frames),device=self.device,dtype=torch.bool)
        out=[m(x[None],valid) for m in self.models]
        pos=(frames-frames[0])/max(frames[-1]-frames[0],1);event=[]
        for e in ['entry','collision']:
            probs=torch.stack([o[e+'_logits'].float().softmax(-1) for o in out]).mean(0)[0].cpu().numpy()
            d=self.prior[e];bw=d['bandwidth'];loc=np.asarray(d['positions'])
            density=np.exp(-.5*((pos[:,None]-loc[None])/bw)**2).mean(-1)/(bw*np.sqrt(2*np.pi))
            logits=np.log(np.maximum(probs,1e-12))+np.log((1-d['uniform_weight'])*density+d['uniform_weight'])
            event.append(logits)
        el,cl=event;ci=int(np.argmax(np.maximum.accumulate(el)+cl));ei=int(np.argmax(el[:ci+1]))
        side=torch.stack([o['side_logits'].float().softmax(-1) for o in out]).mean(0)[0]
        ev=torch.stack([o['evasion_logits'].float().sigmoid() for o in out]).mean(0)[0]
        return {'entry_frame':int(frames[ei]),'collision_frame':int(frames[ci]),'entry_side':int(side.argmax()),'evasion_space':int(ev>=.5)}
