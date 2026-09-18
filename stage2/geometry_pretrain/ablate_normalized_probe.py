"""Controlled cached-feature ablations; never writes to the submission folder."""
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from stage2.geometry_pretrain.downstream_probe import TemporalProbe
from stage2.utils.joint_losses import joint_loss, constrained_decode

ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=ROOT.parent
OUT=WORKSPACE/'outputs/geometry_pretrain/fps_ablation'
VARIANTS=[
    dict(name='baseline128',count=128,jitter=True,temporal=True,backbone='adapted_noanchor'),
    dict(name='frames64',count=64,jitter=True,temporal=True,backbone='adapted_noanchor'),
    dict(name='frames256',count=256,jitter=True,temporal=True,backbone='adapted_noanchor'),
    dict(name='no_jitter',count=128,jitter=False,temporal=True,backbone='adapted_noanchor'),
    dict(name='no_temporal',count=128,jitter=True,temporal=False,backbone='adapted_noanchor'),
    dict(name='original_backbone',count=128,jitter=True,temporal=True,backbone='original'),
]


def read(name):
    return [json.loads(s) for s in (WORKSPACE/f'data/stage2/manifests/{name}.jsonl').read_text().splitlines() if s.strip()]


def sample(idx,count,jitter=False):
    pos=np.linspace(idx[0],idx[-1],count)
    if jitter: pos[1:-1]+=np.random.uniform(-.45,.45,count-2)*(idx[-1]-idx[0])/(count-1)
    hi=np.searchsorted(idx,pos).clip(0,len(idx)-1); lo=np.maximum(hi-1,0)
    return np.where(pos-idx[lo]<=idx[hi]-pos,lo,hi)


class Features(Dataset):
    def __init__(self,rows,config,train=False,stride=1):
        self.rows,self.config,self.train=rows,config,train
        self.arrays=[]
        for row in rows:
            a=np.load(WORKSPACE/f'cache/geometry_pretrain/stage2_probe_features/{config["backbone"]}/{row["sample_id"]}.npy',mmap_mode='r')
            idx=np.arange(0,row['num_frames'],max(1,round(row['native_fps']/10)))
            assert len(a)==len(idx)
            self.arrays.append((a[::stride],idx[::stride]))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]; a,idx=self.arrays[i]; selected=sample(idx,self.config['count'],self.train and self.config['jitter'])
        frames=idx[selected]
        return dict(x=torch.from_numpy(np.array(a[selected],copy=True)),time_valid=torch.ones(len(frames),dtype=torch.bool),
                    frame_seconds=torch.tensor(frames/row['native_fps'],dtype=torch.float32),
                    entry_index=int(np.abs(frames-row['entry_frame']).argmin()),collision_index=int(np.abs(frames-row['collision_frame']).argmin()),
                    entry_side=int(row['entry_side']=='RIGHT'),evasion=int(row['evasion_space']),
                    entry_s=row['entry_frame']/row['native_fps'],collision_s=row['collision_frame']/row['native_fps'])


class Probe(TemporalProbe):
    def __init__(self,temporal=True):
        super().__init__(); self.use_temporal=temporal
    def forward(self,x,valid):
        if self.use_temporal: return super().forward(x,valid)
        b,t,n,c=x.shape
        h=self.frame(self.tok(self.norm(x.float())).reshape(b,t,-1))
        ev=self.event(self.drop(h)); neg=torch.finfo(ev.dtype).min/4
        a=self.attn(h)[...,0].masked_fill(~valid,neg).softmax(-1)
        pooled=torch.einsum('bt,bth->bh',a,h)
        return dict(entry_logits=ev[...,0].masked_fill(~valid,neg),collision_logits=ev[...,1].masked_fill(~valid,neg),
                    side_logits=self.side(self.drop(pooled)),evasion_logits=self.evasion(self.drop(pooled))[:,0])


def dev(batch): return {k:v.cuda() for k,v in batch.items()}


def score(records):
    a=np.asarray(records); out={}
    out['entry_accuracy']=float((np.abs(a[:,0]-a[:,1])<=.300001).mean())
    out['collision_accuracy']=float((np.abs(a[:,2]-a[:,3])<=.300001).mean())
    for name,start in [('side_f1',4),('evasion_f1',6)]:
        pred=a[:,start].astype(int); target=a[:,start+1].astype(int)
        cm=np.bincount(target*2+pred,minlength=4).reshape(2,2)
        f1=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
        out[name]=float(f1.mean())
    out['score']=.35*(out['entry_accuracy']+out['collision_accuracy'])+.15*(out['side_f1']+out['evasion_f1'])
    return out


@torch.inference_mode()
def evaluate(model,loader):
    model.eval(); records=[]
    for batch in loader:
        batch=dev(batch); out=model(batch['x'],batch['time_valid']); e,c=constrained_decode(out['entry_logits'],out['collision_logits'])
        values=torch.stack([batch['frame_seconds'].gather(1,e[:,None])[:,0],batch['entry_s'],
                            batch['frame_seconds'].gather(1,c[:,None])[:,0],batch['collision_s'],
                            out['side_logits'].argmax(-1),batch['entry_side'],(out['evasion_logits']>=0).long(),batch['evasion']],dim=1)
        records.extend(values.cpu().tolist())
    return score(records),records


def fingerprint():
    root=ROOT/'submission'
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def run(config,seed,train,val):
    out=OUT/config['name']/f'seed{seed}'; out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model=Probe(config['temporal']).cuda()
    tl=DataLoader(Features(train,config,True),batch_size=8,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
    vl=DataLoader(Features(val,config),batch_size=8,num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.001,total_steps=40*len(tl),pct_start=.1)
    best=-1; history=[]; start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,41):
        model.train(); total=0
        for batch in tl:
            batch=dev(batch); pred=model(batch['x'],batch['time_valid'])
            loss,_=joint_loss(pred,batch,entry_sigma_seconds=.15,collision_sigma_seconds=.10)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); scheduler.step(); total+=loss.item()
        metrics,records=evaluate(model,vl)
        history.append(dict(epoch=epoch,loss=total/len(tl),**metrics))
        if metrics['score']>best:
            best=metrics['score']; best_epoch=epoch; best_metrics=metrics; best_records=records
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch%10==0: print(json.dumps(dict(variant=config['name'],seed=seed,epoch=epoch,score=metrics['score'],best=best)),flush=True)
    final_metrics,final_records=metrics,records
    model.load_state_dict(best_state)
    stress={}
    for stride in (2,4):
        stress[str(stride)],_=evaluate(model,DataLoader(Features(val,config,stride=stride),batch_size=8))
    torch.save(dict(model=best_state,config=config,seed=seed,epoch=best_epoch),out/'best.pt')
    result=dict(config=config,seed=seed,best_epoch=best_epoch,best=best_metrics,final=final_metrics,stress=stress,
                seconds=time.perf_counter()-start,peak_gpu_MiB=torch.cuda.max_memory_allocated()/2**20,
                best_records=best_records,final_records=final_records,val_ids=[r['sample_id'] for r in val],history=history)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',config['name'],seed,'best',best,'final',final_metrics['score'],'seconds',result['seconds'],flush=True)
    return result


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.cuda.set_per_process_memory_fraction(.20)
    OUT.mkdir(parents=True,exist_ok=True)
    before=fingerprint(); (OUT/'submission_before.json').write_text(json.dumps(before,indent=2))
    train,val=read('train'),read('val'); assert len(train)==201 and len(val)==50
    assert not {r['sample_id'] for r in train}&{r['sample_id'] for r in val}
    results=[]
    for config in VARIANTS:
        for seed in (0,1,2):
            path=OUT/config['name']/f'seed{seed}/result.json'
            result=json.loads(path.read_text()) if path.exists() else run(config,seed,train,val)
            results.append(result)
            (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    after=fingerprint(); assert before==after,'Submission files unexpectedly changed during ablations'
    (OUT/'submission_unchanged.json').write_text(json.dumps({'unchanged':True,'file_count':len(before)}))


if __name__=='__main__':main()
