"""Stream held-out CPU predictions; compare pooling under rate/context changes.

CPU only, one torch thread, clear memory maps per clip; yield on low RAM.
No shared source modifications. Saves resumable per-fold outputs locally.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from stage2.generalization import robust_eval as R
from stage2.aux_signal_experiments.model import load
from stage2.long_context_v2_experiments import common as C
from stage2.spotting_experiments.objective import constrained_anchors
from .inference_experiment import EXCL, paired

OUT = Path(__file__).resolve().parent

def available_gb():
    return next(int(s.split()[1])/2**20 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))

def yield_if_busy():
    while available_gb() < 10.5:
        print('Yielding: available RAM below 10.5 GiB', flush=True)
        time.sleep(20)

@torch.inference_mode()
def predict(models, it):
    x = it['x'][None]
    valid = torch.ones(1, len(it['frames']), dtype=torch.bool)
    outs = [m(x, valid, motion=it['both'][None]) for m in models]
    side = float(torch.stack([o['side_logits'].float().softmax(-1)[0, 1] for o in outs]).mean())
    eva = float(torch.stack([o['evasion_logits'].float().sigmoid()[0] for o in outs]).mean())
    row = it['row']; result = {}
    for mode in ['arithmetic', 'median']:
        lp = []
        for event in ['entry', 'collision']:
            p = torch.stack([o[event+'_logits'].float().softmax(-1) for o in outs])
            p = p.mean(0) if mode == 'arithmetic' else p.median(dim=0).values
            lp.append(p.clamp_min(1e-12).log())
        ei, ci = constrained_anchors(*lp)
        result[mode] = {'sample_id': it['sid'], 'source_id': row['source_id'],
            'entry_frame': int(it['frames'][int(ei[0])]), 'collision_frame': int(it['frames'][int(ci[0])]),
            'entry_side': int(side >= .5), 'evasion_space': int(eva >= .5),
            'entry_gt': int(row['entry_frame']), 'collision_gt': int(row['collision_frame']),
            'entry_side_gt': int(row['entry_side'] == 'RIGHT'), 'evasion_gt': int(row['evasion_space']),
            'num_available_frames': it['n_native']}
    return result

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pilot', action='store_true'); args = ap.parse_args()
    assert not torch.cuda.is_available(), 'Launch with CUDA_VISIBLE_DEVICES empty'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    views = [(1, 0.0), (2, 0.0), (3, 0.0), (1, 0.5)]
    (OUT/'robust_folds').mkdir(exist_ok=True)
    started = time.monotonic()
    for fold in range(5):
        paths = [R.root(f) / f / 'cv' / f'fold{fold}_seed{s}' / 'checkpoint.pt' for f in ['E4_sa','E2_sa','XN4_sa'] for s in range(3)]
        assert all(p.exists() for p in paths)
        yield_if_busy()
        models = [load(p, torch.device('cpu')).eval() for p in paths]
        for stride, crop in views:
            dest = OUT/'robust_folds'/f'fold{fold}_k{stride}_crop{crop}.json'
            if dest.exists() and not args.pilot: continue
            preds = {'arithmetic': [], 'median': []}; timings=[]
            rows = [r for r in C.rows(str(R.FOLDS/f'fold{fold}_val.jsonl')) if r['sample_id'] not in EXCL]
            for row in rows[:3] if args.pilot else rows:
                yield_if_busy(); t0=time.monotonic()
                it=R.item(row, stride, crop); result=predict(models,it)
                for mode,p in result.items(): preds[mode].append(p)
                timings.append({'sample_id':row['sample_id'], 'sampled_frames':len(it['frames']), 'seconds':time.monotonic()-t0})
                del it; C.dense.cache_clear()
            record={'fold':fold,'stride':stride,'crop':crop,'predictions':preds,'timings':timings,'checkpoints':[str(p.relative_to(C.REPO)) for p in paths]}
            if args.pilot:
                C.dump(OUT/'cpu_pilot.json',record); print('PILOT',timings,flush=True); return
            C.dump(dest,record)
            print(dest.name, 'completed', len(rows), 'clips', flush=True)
        del models
    res={}
    for stride,crop in views:
        ps={'arithmetic':[],'median':[]}; ts=[]
        for fold in range(5):
            d=json.loads((OUT/'robust_folds'/f'fold{fold}_k{stride}_crop{crop}.json').read_text())
            for mode in ps: ps[mode]+=d['predictions'][mode]
            ts+=d['timings']
        res[f'k{stride}_crop{crop}']={'breakdown':{mode:C.breakdown(p) for mode,p in ps.items()},'paired':paired(ps['median'],ps['arithmetic']),
            'cpu_head_and_cache_seconds_median':float(np.median([t['seconds'] for t in ts])),
            'cpu_head_and_cache_seconds_p95':float(np.quantile([t['seconds'] for t in ts],.95))}
    C.dump(OUT/'robust_inference_results.json',{'results':res,'elapsed_seconds':time.monotonic()-started,
        'protocol':'nine matched OOF members; duplicate-excluded284; plain ordered decoding without motion fusion/snap; no FPS inputs; context crop uses GT only to preserve events; CPU timing excludes video decoding/backbone'})
    print(json.dumps(res),flush=True)

if __name__=='__main__': main()
