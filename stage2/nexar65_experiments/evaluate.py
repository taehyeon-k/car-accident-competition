"""Reload checkpoints and verify reproducibility; optional equal probability ensemble."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from stage2.long_context_v2_experiments import common as C
from .models import Pyramid
from .prior import fit,apply

ROOT=Path(__file__).resolve().parent

def load(path,device):
    s=torch.load(path,map_location='cpu',weights_only=False);c=s['config']
    model=Pyramid(kind=c['kind'],dropout=c['dropout'],prior=c.get('prior_config')).to(device)
    model.load_state_dict(s['model']);model.eval()
    return model

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--post-prior',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12)
    out=ROOT/'results'/'verified'/a.name
    if out.exists():raise RuntimeError(f'Refusing overwrite {out}')
    models=[load(path,'cuda') for path in a.checkpoints]
    model=models[0] if len(models)==1 else C.ProbEnsemble(models).eval()
    ps=C.run(model,[C.make_item(r,'adaptive') for r in C.rows('val')],'cuda',keep_logits=True)
    if a.post_prior:ps=apply(ps,fit([r for r in C.rows('train') if C.source(r)=='NEXAR']))
    res=C.breakdown(ps)
    C.dump(out/'predictions.json',ps);C.dump(out/'metrics.json',res);C.dump(out/'config.json',vars(a))
    print(a.name,C.short_table(res),flush=True)

if __name__=='__main__':main()
