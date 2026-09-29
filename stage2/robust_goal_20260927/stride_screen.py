"""Matched completed balanced-stride arm; cached predictions, no training."""
import json
from pathlib import Path
from stage2.aux_signal_experiments import ensemble_test as ET
from stage2.long_context_v2_experiments import common as C
from .inference_experiment import EXCL, pool, paired

OUT=Path(__file__).resolve().parent

def main():
    recipes={'E4_sa':['E4_sa'],'E4_sb':['E4_sb'],
             'v8':['E4_sa','E2_sa','XN4_sa'], 'v8_balanced_E4':['E4_sb','E2_sa','XN4_sa']}
    results={};preds={}
    for name,families in recipes.items():
        for decoder,beta,snap in [('plain',0.,False),('deployment',.5,True)]:
            ps=[]
            for fold in range(5):
                members=[]
                for f in families:
                    for seed in range(3):
                        path=ET.PE.root(f)/f/'cv'/f'fold{fold}_seed{seed}'/'predictions.json'
                        members.append({p['sample_id']:p for p in json.loads(path.read_text())})
                ids=sorted(members[0]);assert all(sorted(m)==ids for m in members)
                for sid in ids:
                    items=[m[sid] for m in members];assert all(p['frames']==items[0]['frames'] for p in items)
                    ps.append(pool(items,'arithmetic',beta,snap))
            key=name+'/'+decoder
            preds[key]=[p for p in ps if p['sample_id'] not in EXCL]
            results[key]={'original':C.breakdown(ps),'clean':C.breakdown(preds[key])}
    for candidate,control in [('E4_sb','E4_sa'),('v8_balanced_E4','v8')]:
        for decoder in ['plain','deployment']:
            key=candidate+'/'+decoder;results[key]['paired']=paired(preds[key],preds[control+'/'+decoder])
            print(key,results[key]['clean']['overall']['score'],results[key]['paired'],flush=True)
    C.dump(OUT/'stride_screen_results.json',{'protocol':'Three completed seeds0-2, original five folds, 284 duplicate-excluded clips; E4_sb uses .34/.33/.33 versus .5/.25/.25. Frozen .5 decoder; no further tuning. Source/rate confirmation needed.', 'results':results})

if __name__=='__main__':main()
