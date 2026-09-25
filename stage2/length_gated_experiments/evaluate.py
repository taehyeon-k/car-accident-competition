"""Predeclared length switch between the old P2 ensemble and NEXAR specialist.

Both input prediction sets use the original train (279) / val (70) manifests.
The 500-frame switch is set from train lengths: MMAU <=415, NEXAR >=570;
the sole CAUSALCRASH train video is 1109 frames. The switch does not read source.
"""
import json
from pathlib import Path
import numpy as np
from stage2.long_context_v2_experiments import common as C

ROOT=Path(__file__).resolve().parent
BASE=C.REPO/'stage2/nexar65_experiments/results/train_prior/P2_ens4/predictions_base.json'
SPECIALISTS={
 'seed0':C.REPO/'stage2/nexar65_experiments/results/verified/first_three_prior/predictions.json',
 'seed1':C.REPO/'stage2/nexar65_experiments/results/verified/first_three_prior_seed1/predictions.json',
 'seed2':C.REPO/'stage2/nexar65_experiments/results/verified/first_three_prior_seed2/predictions.json',
 'all_seeds':C.REPO/'stage2/nexar65_experiments/results/verified/nine_heads_prior/predictions.json',
}
THRESHOLD=500

def read(path):
    rows=json.loads(path.read_text())
    assert len(rows)==70 and len({r['sample_id'] for r in rows})==70
    return {r['sample_id']:r for r in rows}

def evaluate():
    train=C.rows('train');val=C.rows('val')
    assert len(train)==279 and len(val)==70
    assert not {r['sample_id'] for r in train}&{r['sample_id'] for r in val}
    base=read(BASE)
    assert set(base)=={r['sample_id'] for r in val}
    out={'config':{'threshold_frames':THRESHOLD,'selection':'specialist if num_frames > threshold, else original P2 ensemble',
                   'train_manifest':'/workspace/data/stage2/manifests/train.jsonl','val_manifest':'/workspace/data/stage2/manifests/val.jsonl',
                   'base_predictions':str(BASE),'specialist_predictions':{k:str(v) for k,v in SPECIALISTS.items()}},
         'base':C.breakdown(list(base.values())),'variants':{}}
    for name,path in SPECIALISTS.items():
        specialist=read(path);assert set(specialist)==set(base)
        ps=[];routed=[]
        for row in val:
            sid=row['sample_id'];use_specialist=int(row['num_frames'])>THRESHOLD
            p=specialist[sid] if use_specialist else base[sid]
            assert p['entry_gt']==row['entry_frame'] and p['collision_gt']==row['collision_frame']
            ps.append(p)
            if use_specialist:routed.append(sid)
        metrics=C.breakdown(ps)
        target=metrics['source:NEXAR'];overall=metrics['overall']
        out['variants'][name]={'breakdown':metrics,'routed_ids':routed,
                               'meets_target':bool(target['score']>.65 and overall['score']>.73)}
        C.dump(ROOT/f'predictions_{name}.json',ps)
        print(name,C.short_table(metrics),'routed',len(routed))
    C.dump(ROOT/'results.json',out)
    return out

if __name__=='__main__':evaluate()
