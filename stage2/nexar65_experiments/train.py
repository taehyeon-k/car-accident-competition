"""Isolated NEXAR-targeted experiments. FPS used only in post-prediction diagnostics.

Fixed original train/val split; checkpoint selection uses fixed frame tolerances
on NEXAR (3/6/9 frames), never native FPS. All attempted epochs are recorded.
"""
import argparse,copy,json,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments.train import random_crop_item
from stage2.spotting_experiments.data import collate
from stage2.spotting_experiments.train import macro_f1
from .models import Pyramid

ROOT=Path(__file__).resolve().parent

def selection(ps,radius=0):
    ps=[p for p in ps if C.source(p)=='NEXAR']
    radii=[radius] if radius else [3,6,9]
    events=sum(np.mean([abs(p[e+'_frame']-p[e+'_gt'])<=t for p in ps]) for e in ['entry','collision'] for t in radii)/len(radii)
    return float(.35*events+.15*macro_f1([p['entry_side_gt'] for p in ps],[p['entry_side'] for p in ps])+.15*macro_f1([p['evasion_gt'] for p in ps],[p['evasion_space'] for p in ps]))

def loss(out,b,sigma):
    mask=b['time_valid']; frames=b['frame_numbers'].float(); terms=[]
    for e in ['entry','collision']:
        gt=b[e+'_frame'][:,None].float()
        if sigma==0:
            target=F.one_hot(b[e+'_index'],frames.shape[1]).float()
        else:
            width=sigma if sigma>0 else .015*(frames.max(-1).values-frames[:,0])[:,None].clamp_min(1)
            logw=-.5*((frames-gt)/width)**2
            target=logw.masked_fill(~mask,-1e4).softmax(-1)
        terms.append(-(target*F.log_softmax(out[e+'_logits'].float(),-1)).sum(-1))
    side=F.cross_entropy(out['side_logits'],b['entry_side'],reduction='none')
    ev=F.binary_cross_entropy_with_logits(out['evasion_logits'],b['evasion'].float(),reduction='none')
    pe=out['entry_logits'].softmax(-1); pc=out['collision_logits'].softmax(-1)
    before=pc.cumsum(-1)-pc
    return (.35*(terms[0]+terms[1])+.15*(side+ev)+.05*(pe*before).sum(-1)).mean()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--sigma',type=float,default=-1); p.add_argument('--crop',type=float,default=0)
    p.add_argument('--nexar-repeat',type=int,default=1); p.add_argument('--nexar-only',action='store_true')
    p.add_argument('--kind',default='pyramid',choices=['pyramid','tcn']); p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--patience',type=int,default=15); p.add_argument('--ema',type=float,default=0)
    p.add_argument('--lr',type=float,default=.001); p.add_argument('--dropout',type=float,default=.35)
    p.add_argument('--selection-radius',type=int,default=0);p.add_argument('--prior',action='store_true');p.add_argument('--init')
    a=p.parse_args(); out=ROOT/'results'/a.name/f'seed{a.seed}'
    if out.exists(): raise RuntimeError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.12)
    device=torch.device('cuda'); rng=np.random.default_rng(a.seed)
    tr=C.rows('train'); va=C.rows('val')
    if a.nexar_only: tr=[r for r in tr if C.source(r)=='NEXAR']
    ids=set(r['sample_id'] for r in tr); assert not ids.intersection(r['sample_id'] for r in va)
    train=[C.make_item(r,'adaptive') for r in tr]; val=[C.make_item(r,'adaptive') for r in va]
    prior=None
    if a.prior:
        from .prior import fit
        prior=fit([r for r in tr if C.source(r)=='NEXAR'])
        assert not a.crop,'Position prior and remapped crops require a separate ablation'
    model=Pyramid(dropout=a.dropout,kind=a.kind,prior=prior).to(device)
    if a.init:
        state=torch.load(a.init,map_location='cpu',weights_only=False)['model']
        missing,unexpected=model.load_state_dict(state,strict=False)
        assert not unexpected and all(k.endswith('_prior_positions') for k in missing),(missing,unexpected)
    ema=copy.deepcopy(model).eval() if a.ema else None
    order_base=[i for i,r in enumerate(tr) for _ in range(a.nexar_repeat if C.source(r)=='NEXAR' else 1)]
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.05)
    steps=(len(order_base)+3)//4
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=a.epochs*steps,pct_start=.1)
    cfg={**vars(a),'n_train':len(tr),'n_val':len(va),'train_ids':sorted(ids),'val_ids':[r['sample_id'] for r in va],'prior_config':prior,'selection':'NEXAR fixed frame tolerance plus attributes','params':sum(p.numel() for p in model.parameters())}
    C.dump(out/'config.json',cfg)
    history=[]; best=-1; stale=0; start=time.time()
    for epoch in range(1,a.epochs+1):
        model.train(); order=rng.permutation(order_base); losses=[]
        for s in range(0,len(order),4):
            items=[random_crop_item(tr[i],rng) if a.crop and tr[i]['num_frames']>256 and rng.random()<a.crop else train[i] for i in order[s:s+4]]
            b=collate(items); b={k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()}
            z=loss(model(b['x'],b['time_valid']),b,a.sigma)
            opt.zero_grad();z.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sched.step();losses.append(float(z.detach()))
            if ema is not None:
                with torch.no_grad():
                    for e,m in zip(ema.parameters(),model.parameters()):e.lerp_(m,1-a.ema)
        ev=ema if ema is not None else model; ev.eval()
        ps=C.run(ev,val,device); score=selection(ps,a.selection_radius); met=C.breakdown(ps)
        rec={'epoch':epoch,'loss':float(np.mean(losses)),'selection':score,'nexar':met['source:NEXAR'],'overall':met['overall']['score']}
        history.append(rec);C.dump(out/'history.json',history)
        print(a.name,a.seed,epoch,round(score,4),round(rec['nexar']['score'],4),flush=True)
        if score>best+1e-8:
            best=score;stale=0
            state={k:v.detach().cpu().clone() for k,v in ev.state_dict().items()}
            torch.save({'model':state,'config':cfg,'epoch':epoch},out/'checkpoint.pt')
            C.dump(out/'predictions.json',C.run(ev,val,device,keep_logits=True))
            C.dump(out/'metrics.json',{'breakdown':met,'selection':score,'epoch':epoch,'wall_seconds':time.time()-start})
        else: stale+=1
        if stale>=a.patience:break
    print('RESULT',a.name,a.seed,(out/'metrics.json').read_text(),flush=True)

if __name__=='__main__':main()
