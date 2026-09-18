"""FPS-free normalized-clip head: held-out selection, then refit on all 251 clips."""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'submission_tools/fps_stage2'))
from probe import TemporalProbe
from sampling import normalized_indices
from stage2.utils.joint_losses import joint_loss, constrained_decode
from stage2.utils.joint_metrics import JointMetricAccumulator, joint_metric_packet


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


class Features(Dataset):
    def __init__(self, rows, train=False):
        self.rows, self.train = rows, train
        self.arrays = []
        for row in rows:
            a = np.load('/workspace/cache/geometry_pretrain/stage2_probe_features/adapted_noanchor/'+row['sample_id']+'.npy', mmap_mode='r')
            idx = np.arange(0,row['num_frames'],max(1,round(row['native_fps']/10)))
            if len(a) != len(idx): raise ValueError('Cached features do not match frame count')
            self.arrays.append((a,idx))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]; a, idx = self.arrays[i]
        selected = normalized_indices(idx, 128, np.random if self.train else None)
        selected_idx = idx[selected]
        return {'x':torch.from_numpy(np.array(a[selected],copy=True)),
                'time_valid':torch.ones(128,dtype=torch.bool),
                'frame_seconds':torch.tensor(selected_idx/row['native_fps'],dtype=torch.float32),
                'entry_index':int(np.abs(selected_idx-row['entry_frame']).argmin()),
                'collision_index':int(np.abs(selected_idx-row['collision_frame']).argmin()),
                'entry_side':int(row['entry_side']=='RIGHT'),'evasion':int(row['evasion_space']),
                'entry_s':row['entry_frame']/row['native_fps'],
                'collision_s':row['collision_frame']/row['native_fps']}


def dev(batch): return {k:v.cuda() for k,v in batch.items()}


@torch.inference_mode()
def evaluate(model,loader):
    model.eval(); metrics=JointMetricAccumulator(); hits_e=hits_c=n=0
    for batch in loader:
        batch=dev(batch); out=model(batch['x'],batch['time_valid'])
        metrics.update(joint_metric_packet(out,batch))
        e,c=constrained_decode(out['entry_logits'],out['collision_logits'])
        es=batch['frame_seconds'].gather(1,e[:,None])[:,0]; cs=batch['frame_seconds'].gather(1,c[:,None])[:,0]
        hits_e += int(((es-batch['entry_s']).abs()<=.300001).sum())
        hits_c += int(((cs-batch['collision_s']).abs()<=.300001).sum()); n+=len(e)
    m=metrics.compute(); m['acc_entry_native']=hits_e/n; m['acc_collision_native']=hits_c/n
    m['competition_score_native']=.35*(hits_e+hits_c)/n+.15*(m['f1_entry_side_macro']+m['f1_evasion_space_macro'])
    return m


def fit(rows,epochs,horizon,out,validation=None):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model=TemporalProbe(dropout=.3).cuda()
    loader=DataLoader(Features(rows,True),batch_size=8,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(0))
    vl=DataLoader(Features(validation),batch_size=8,num_workers=0) if validation else None
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.001,total_steps=horizon*len(loader),pct_start=.1)
    best=-1; best_epoch=epochs; history=[]
    for epoch in range(1,epochs+1):
        model.train(); total=0
        for batch in loader:
            batch=dev(batch); pred=model(batch['x'],batch['time_valid'])
            loss,_=joint_loss(pred,batch,entry_sigma_seconds=.15,collision_sigma_seconds=.10)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); sched.step(); total+=loss.item()
            time.sleep(.015)  # Yield between short GPU steps to the ongoing Stage 3 job.
        result={'epoch':epoch,'loss':total/len(loader)}
        if vl:
            result.update(evaluate(model,vl))
            if result['competition_score_native']>best:
                best=result['competition_score_native']; best_epoch=epoch
                save(model,out/'selection.pt',rows,epoch,result)
        history.append(result); print(('selection' if vl else 'all251'),json.dumps(result),flush=True)
        (out/('selection_history.json' if vl else 'refit_history.json')).write_text(json.dumps(history,indent=2))
    if not vl: save(model,out/'probe.pt',rows,epochs,{'held_out_score':None,'note':'All 251 labels used for final refit; no independent evaluation of this checkpoint'})
    return best_epoch,history


def save(model,path,rows,epoch,metrics):
    torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'model_kwargs':{'dropout':.3},'sampling_version':'normalized-clip-v1','sample_count':128,
                'backbone_variant':'phase1_partial_noanchor','epochs':epoch,'seed':0,
                'train_ids':sorted(r['sample_id'] for r in rows),'validation':metrics},path)


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.10)
    out=ROOT/'submission_tools/fps_stage2'; out.mkdir(exist_ok=True)
    train=read('/workspace/data/stage2/manifests/train.jsonl'); val=read('/workspace/data/stage2/manifests/val.jsonl')
    all_rows=read('/workspace/data/stage2/manifests/all.jsonl')
    ids=lambda rows:{r['sample_id'] for r in rows}
    assert len(all_rows)==251 and len(ids(all_rows))==251 and not ids(train)&ids(val) and ids(train+val)==ids(all_rows)
    t=time.perf_counter(); best_epoch,history=fit(train,40,40,out,val)
    best=max(history,key=lambda m:m['competition_score_native'])
    # Same LR trajectory as the selection run, stopped at the selected epoch.
    fit(all_rows,best_epoch,40,out)
    report={'selection_train_clips':len(train),'selection_val_clips':len(val),'final_train_clips':len(all_rows),
            'selected_epoch':best_epoch,'held_out_selection_metrics':best,'seconds':time.perf_counter()-t,
            'peak_gpu_MiB':torch.cuda.max_memory_allocated()/2**20,
            'note':'FPS is never a model input; source FPS is used only for training loss widths and validation scoring.'}
    (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report),flush=True)


if __name__=='__main__': main()
