"""Export and independently verify the seed-0 successful ensemble, without replacing submission."""
import argparse,hashlib,json
from pathlib import Path
import torch
from .prior import fit
from .runtime import NexarSpecialist
from stage2.long_context_v2_experiments import common as C

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--all-seeds',action='store_true');args=ap.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12)
    out=ROOT/('candidate_all_seeds' if args.all_seeds else 'candidate_seed0')
    if out.exists():raise RuntimeError('Candidate already exists; refusing overwrite')
    out.mkdir()
    members=[]
    jobs=[(name,seed) for seed in ([0,1,2] if args.all_seeds else [0]) for name in ['LN_soft','LN_exact','TCN_frame6']]
    for name,seed in jobs:
        path=ROOT/'results'/name/f'seed{seed}'/'checkpoint.pt'
        st=torch.load(path,map_location='cpu',weights_only=False);cfg=st['config']
        filename=f'{name}_seed{seed}.pt'
        torch.save({'model':st['model'],'kind':cfg['kind'],'dropout':cfg['dropout']},out/filename)
        members.append({'file':filename,'source':str(path),'epoch':st['epoch'],'sha256':hashlib.sha256((out/filename).read_bytes()).hexdigest()})
    rows=[r for r in C.rows('train') if C.source(r)=='NEXAR']
    cfg={'members':members,'prior':fit(rows),'prior_fit_ids':[r['sample_id'] for r in rows],
         'sampling':'long_context_v2_experiments.common.select_adaptive',
         'backbone':'/workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor/backbone_best.pth',
         'scope':'NEXAR specialist; no source identity is used at inference'}
    C.dump(out/'candidate.json',cfg)
    runtime=NexarSpecialist(out,'cuda')
    expected_dir='nine_heads_prior' if args.all_seeds else 'first_three_prior'
    expected={p['sample_id']:p for p in json.loads((ROOT/'results/verified'/expected_dir/'predictions.json').read_text())}
    mismatches=[];preds=[]
    for row in C.rows('val'):
        item=C.make_item(row,'adaptive');p=runtime.predict(item['x'],item['frame_numbers'].numpy())
        old=expected[row['sample_id']]
        for k,v in p.items():
            if v!=old[k]:mismatches.append([row['sample_id'],k,v,old[k]])
        preds.append({**old,**p})
    C.dump(out/'verification.json',{'n_videos':len(preds),'mismatches':mismatches,'breakdown':C.breakdown(preds)})
    assert not mismatches,mismatches
    print('Exported candidate; exact prediction parity on',len(preds),'videos;',C.short_table(C.breakdown(preds)))

if __name__=='__main__':main()
