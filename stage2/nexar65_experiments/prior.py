"""Training-only NEXAR temporal KDE prior; fixed recipe, no validation fitting.

Adds log density to visual event logits. 10% uniform contamination protects
events outside the typical range. Source identity is not an inference input.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from stage2.long_context_v2_experiments import common as C
from stage2.spotting_experiments.objective import constrained_anchors

ROOT=Path(__file__).resolve().parent

def fit(rows):
    out={}
    for event in ['entry','collision']:
        x=np.array([r[event+'_frame']/max(r['num_frames']-1,1) for r in rows])
        scale=min(x.std(ddof=1),(np.quantile(x,.75)-np.quantile(x,.25))/1.349)
        bandwidth=max(.01,.9*scale*len(x)**(-.2))
        out[event]={'positions':x.tolist(),'bandwidth':float(bandwidth),'uniform_weight':.1}
    return out

def apply(preds,prior):
    result=[]
    for p in preds:
        q=dict(p);frames=np.array(p['frames']);pos=(frames-frames[0])/max(frames[-1]-frames[0],1)
        ls=[]
        for e in ['entry','collision']:
            d=prior[e];x=np.array(d['positions']);bw=d['bandwidth']
            density=np.exp(-.5*((pos[:,None]-x[None])/bw)**2).mean(-1)/(bw*np.sqrt(2*np.pi))
            density=(1-d['uniform_weight'])*density+d['uniform_weight']
            logits=np.array(p[e+'_logits'])+np.log(density)
            q[e+'_logits']=logits.tolist();ls.append(torch.tensor(logits)[None])
        ei,ci=constrained_anchors(*ls)
        q['entry_frame']=int(frames[int(ei)]);q['collision_frame']=int(frames[int(ci)])
        result.append(q)
    return result

def main():
    a=argparse.ArgumentParser();a.add_argument('--models',nargs='+',default=list(C.BASELINES));args=a.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12)
    tr=[r for r in C.rows('train') if C.source(r)=='NEXAR'];prior=fit(tr)
    out=ROOT/'results'/'train_prior';out.mkdir(exist_ok=True)
    C.dump(out/'prior.json',{'fit_ids':[r['sample_id'] for r in tr],'prior':prior})
    for name in args.models:
        path=out/name
        if path.exists():raise RuntimeError(f'Refusing overwrite {path}')
        model,sampling=C.load_named(name,'cuda')
        ps=C.run(model,[C.make_item(r,sampling) for r in C.rows('val')],'cuda',keep_logits=True)
        new=apply(ps,prior)
        C.dump(path/'predictions_base.json',ps);C.dump(path/'predictions.json',new)
        res={'base':C.breakdown(ps),'prior':C.breakdown(new)};C.dump(path/'metrics.json',res)
        print(name,'BASE',C.short_table(res['base']),'PRIOR',C.short_table(res['prior']),flush=True)
        del model

if __name__=='__main__':main()
