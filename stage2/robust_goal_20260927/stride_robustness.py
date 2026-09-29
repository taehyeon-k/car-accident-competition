"""Matched rate test of completed E4_sa and E4_sb; CPU only."""
import json
from pathlib import Path
import torch
from stage2.generalization import robust_eval as R
from stage2.long_context_v2_experiments import common as C
from stage2.aux_signal_experiments.model import load
from .resource_guard import resource_guard
from .inference_experiment import EXCL, paired

OUT=Path(__file__).resolve().parent

def main():
    assert not torch.cuda.is_available()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    dest=OUT/'stride_robust_folds';dest.mkdir(exist_ok=True)
    for fold in range(5):
        resource_guard(10.5)
        models={family:[(load(R.root(family)/family/'cv'/f'fold{fold}_seed{s}'/'checkpoint.pt',torch.device('cpu')),'both') for s in range(3)] for family in ['E4_sa','E4_sb']}
        rows=[r for r in C.rows(str(R.FOLDS/f'fold{fold}_val.jsonl')) if r['sample_id'] not in EXCL]
        for stride in [1,2,3]:
            path=dest/f'fold{fold}_k{stride}.json'
            if path.exists():continue
            outputs={f:[] for f in models}
            for row in rows:
                resource_guard(10.5);item=R.item(row,stride)
                for family,mem in models.items():outputs[family].append(R.predict(mem,item,torch.device('cpu')))
                del item;C.dense.cache_clear()
            C.dump(path,outputs);print(path.name,'complete',flush=True)
        del models
    results={}
    for stride in [1,2,3]:
        ps={'E4_sa':[],'E4_sb':[]}
        for fold in range(5):
            data=json.loads((dest/f'fold{fold}_k{stride}.json').read_text())
            for family in ps:ps[family]+=data[family]
        results[f'k{stride}']={'breakdown':{f:C.breakdown(p) for f,p in ps.items()},'paired':paired(ps['E4_sb'],ps['E4_sa'])}
        print('stride',stride,results[f'k{stride}']['paired'],flush=True)
    C.dump(OUT/'stride_robustness_results.json',{'protocol':'Matched seeds0-2; 284 known-duplicate-excluded OOF clips; plain decoder; native numbering; no parameter fitting; source breakdowns included','results':results})

if __name__=='__main__':main()
