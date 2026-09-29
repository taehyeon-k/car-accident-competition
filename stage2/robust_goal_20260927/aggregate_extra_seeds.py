"""Score complete matched OOF seeds separately; do not double-count clips across seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
from stage2.long_context_v2_experiments import common as C
from stage2.phase_study import analyze as A

OUT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True,choices=['aux_sharp','aux_broad','transport'])
    args=parser.parse_args()
    group=json.loads((OUT/'group_folds/metadata.json').read_text())['group_by_id']
    result={'candidate':args.candidate,'seed_results':{},
            'limitations':'Only complete five-fold seeds enter the mean. Paired bootstrap is descriptive; no search correction, pretrained-backbone exposure audit, or leaderboard calibration.'}
    table=['# Matched training-seed confirmation','',f'Candidate: {args.candidate}. Each row is one complete five-fold OOF seed, not a pooled repeat of the same clips.','','| Seed | Rate | Base | Candidate | Delta | 95% paired group interval |','|---:|---|---:|---:|---:|---|']
    for seed in [0,1,2]:
        matches={}
        for fold in range(5):
            pair={}
            expected={r['sample_id'] for r in C.rows(str(OUT/f'group_folds/fold{fold}_val.jsonl'))}
            for arm in ('base',args.candidate):
                folder=OUT/f'training/{arm}/cv/fold{fold}_seed{seed}'
                if not (folder/'rate_eval.json').exists():
                    break
                d=json.loads((folder/'rate_eval.json').read_text())
                cfg=json.loads((folder/'config.json').read_text())
                assert d['native_parity_mismatches']==[] and cfg['seed']==seed
                assert Path(cfg['train_split']).resolve()==OUT/f'group_folds/fold{fold}_train.jsonl'
                assert Path(cfg['val_split']).resolve()==OUT/f'group_folds/fold{fold}_val.jsonl'
                for rate in ('k1','k2','k3'):
                    ids=[p['sample_id'] for p in d[rate]['predictions']]
                    assert len(ids)==len(set(ids)) and set(ids)==expected
                pair[arm]=d
            if len(pair)==2:matches[fold]=pair
        seed_result={'matched_folds':sorted(matches),'complete_five_fold_seed':len(matches)==5,'rates':{}}
        result['seed_results'][str(seed)]=seed_result
        if len(matches)!=5:
            continue
        for rate in ('k1','k2','k3'):
            ps={arm:[p for fold in sorted(matches) for p in matches[fold][arm][rate]['predictions']]
                for arm in ('base',args.candidate)}
            arrays={arm:A.clip_arrays(p) for arm,p in ps.items()}
            ids=arrays['base']['ids']
            assert len(ids)==349 and len(set(ids))==349 and arrays[args.candidate]['ids']==ids
            groups={}
            for i,sid in enumerate(ids):groups.setdefault(group[sid],[]).append(i)
            groups=list(groups.values())
            rng=np.random.default_rng(20260927+seed)
            deltas=[]
            for _ in range(2000):
                idx=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])[None]
                deltas.append(float(A.score_idx(arrays[args.candidate],idx)[0]-A.score_idx(arrays['base'],idx)[0]))
            metrics={arm:C.breakdown(preds) for arm,preds in ps.items()}
            base=metrics['base']['overall']['score'];candidate=metrics[args.candidate]['overall']['score']
            ci=np.quantile(deltas,[.025,.975]).tolist()
            seed_result['rates'][rate]={'base':metrics['base'],'candidate':metrics[args.candidate],
                                        'delta':candidate-base,'ci95':ci}
            table.append(f'| {seed} | {rate} | {base:.6f} | {candidate:.6f} | {candidate-base:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |')
    complete=[v for v in result['seed_results'].values() if v['complete_five_fold_seed']]
    result['complete_seed_count']=len(complete)
    if len(complete)>1:
        result['across_seed_delta']={r:{'mean':float(np.mean([v['rates'][r]['delta'] for v in complete])),
                                        'std':float(np.std([v['rates'][r]['delta'] for v in complete],ddof=1))}
                                     for r in ('k1','k2','k3')}
        table+=['','Across complete seeds (mean ± sample SD of candidate-minus-base): '+', '.join(
            f"{r} {v['mean']:+.6f} ± {v['std']:.6f}" for r,v in result['across_seed_delta'].items())+'.']
    table+=['','Complete seeds: '+str(len(complete))+'/3. No leaderboard forecast or automatic deployment promotion.']
    C.dump(OUT/f'extra_seeds_{args.candidate}_aggregate.json',result)
    (OUT/f'EXTRA_SEEDS_{args.candidate.upper()}_SCORECARD.md').write_text('\n'.join(table)+'\n')
    print('Complete matched seeds:',len(complete))


if __name__=='__main__':main()
