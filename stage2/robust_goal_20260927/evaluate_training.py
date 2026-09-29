"""Evaluate completed grouped-fold CPU checkpoints; never reuse original CV folds."""
import json
import argparse
from pathlib import Path
import torch
from stage2.generalization import robust_eval as R
from stage2.long_context_v2_experiments import common as C
from .model import load
from .resource_guard import resource_guard

OUT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',type=int,default=0,choices=range(5))
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--arms',nargs='+',default=['base','aux_sharp','aux_broad','transport'])
    parser.add_argument('--allow-adjacent-gpu-parity',action='store_true',
                        help='For GPU-trained checkpoints only: record and tolerate at most two adjacent-frame timing differences')
    args=parser.parse_args()
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    rows=C.rows(str(OUT/f'group_folds/fold{args.fold}_val.jsonl'))
    results={}
    for arm in args.arms:
        folder=OUT/'training'/arm/f'cv/fold{args.fold}_seed{args.seed}'
        if not (folder/'predictions.json').exists():continue
        dest=folder/'rate_eval.json'
        if dest.exists():results[arm]=json.loads(dest.read_text());continue
        resource_guard(10.5)
        config=json.loads((folder/'config.json').read_text())
        torch.set_num_threads(config.get('cpu_threads',1))
        m=load(folder/'checkpoint.pt',torch.device('cpu'))
        output={}
        for stride in [1,2,3]:
            ps=[]
            for row in rows:
                resource_guard(10.5)
                it=R.item(row,stride);ps.append(R.predict([(m,'both')],it,torch.device('cpu')))
                del it;C.dense.cache_clear()
            output[f'k{stride}']={'metrics':C.breakdown(ps),'predictions':ps}
        saved={p['sample_id']:p for p in json.loads((folder/'predictions.json').read_text())}
        mismatches=[]
        for p in output['k1']['predictions']:
            for key in ['entry_frame','collision_frame','entry_side','evasion_space']:
                if p[key]!=saved[p['sample_id']][key]:mismatches.append([p['sample_id'],key,p[key],saved[p['sample_id']][key]])
        output['native_parity_mismatches']=mismatches
        if mismatches:
            allowed=(args.allow_adjacent_gpu_parity and config.get('training_device')=='cuda:0'
                     and len(mismatches)<=2 and all(k in ('entry_frame','collision_frame')
                         and abs(a-b)==1 for _,k,a,b in mismatches))
            if not allowed:raise RuntimeError(f'Native prediction parity failed: {mismatches[:3]}')
        output['native_parity_policy']='strict' if not args.allow_adjacent_gpu_parity else 'gpu_adjacent_max2'
        C.dump(dest,output);results[arm]=output
        print(arm,{k:round(output[k]['metrics']['overall']['score'],6) for k in ['k1','k2','k3']},flush=True)
    name=('training_comparison.json' if args.fold==0 else f'training_comparison_fold{args.fold}.json') if args.seed==0 else f'training_comparison_fold{args.fold}_seed{args.seed}.json'
    previous=json.loads((OUT/name).read_text()).get('completed',{}) if (OUT/name).exists() else {}
    results={**previous,**results}
    C.dump(OUT/name,{'protocol':f'one grouped fold seed{args.seed}; native prediction parity verified; plain decoder','fold':args.fold,'seed':args.seed,'completed':results})

if __name__=='__main__':main()
