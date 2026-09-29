"""Arithmetic vs median on existing 12-member E4/E2 head-level LOSO."""
import json
from pathlib import Path
from .inference_experiment import pool, EXCL, paired
from stage2.long_context_v2_experiments import common as C

OUT=Path(__file__).resolve().parent
ROOT=C.REPO/'stage2/generalization/results'

def main():
    results={}; predictions={}
    for source in ['AIHUB','CCD','MMAU','NEXAR']:
        members=[]
        for family in ['LOSO_E4_sa','LOSO_E2_sa']:
            for seed in range(6):
                members.append({p['sample_id']:p for p in json.loads((ROOT/family/f'{source}_seed{seed}'/'predictions.json').read_text())})
        ids=sorted(members[0]); assert all(sorted(m)==ids for m in members)
        for mode in ['arithmetic','median']:
            for decoder,beta,snap in [('plain',0.,False),('deployment',.5,True)]:
                ps=[]
                for sid in ids:
                    items=[m[sid] for m in members]
                    assert all(p['frames']==items[0]['frames'] for p in items)
                    ps.append(pool(items,mode,beta,snap))
                key=f'{source}/{decoder}/{mode}'
                predictions[key]=ps
                results[key]={'original':C.metrics(ps),'clean':C.metrics([p for p in ps if p['sample_id'] not in EXCL])}
        print(source,'complete',flush=True)
    for source in ['AIHUB','CCD','MMAU','NEXAR']:
        for decoder in ['plain','deployment']:
            key=f'{source}/{decoder}'
            a=[p for p in predictions[key+'/median'] if p['sample_id'] not in EXCL]
            b=[p for p in predictions[key+'/arithmetic'] if p['sample_id'] not in EXCL]
            results[key+'/median']['paired_clean']=paired(a,b)
    C.dump(OUT/'source_confirmation.json',{'protocol':'12 members: E4+E2 each seeds0-5; not the packaged three-family v8. Head-level source holdout; shared backbone previously exposed to some source footage. Clean excludes common audit ids. Plain and frozen deployment decoder. No selection/fitting.', 'results':results})
    for key,r in results.items():
        if 'paired_clean' in r: print(key,r['paired_clean'],flush=True)

if __name__=='__main__':main()
