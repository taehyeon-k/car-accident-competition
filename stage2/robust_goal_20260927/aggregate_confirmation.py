"""Aggregate only complete matched grouped folds; seed0 is not seed confirmation."""
import json
from pathlib import Path
import numpy as np
from stage2.long_context_v2_experiments import common as C
from stage2.phase_study import analyze as A

OUT=Path(__file__).resolve().parent
ARMS=['base','aux_sharp','aux_broad']

def main():
    meta=json.loads((OUT/'group_folds/metadata.json').read_text())
    completed={a:{} for a in ARMS};unreadable=[]
    for arm in ARMS:
        for fold in range(5):
            folder=OUT/'training'/arm/f'cv/fold{fold}_seed0';path=folder/'rate_eval.json'
            if not path.exists():continue
            try:data=json.loads(path.read_text())
            except json.JSONDecodeError:unreadable.append(str(path));continue
            assert data['native_parity_mismatches']==[]
            expected={r['sample_id'] for r in C.rows(str(OUT/f'group_folds/fold{fold}_val.jsonl'))}
            for rate in ['k1','k2','k3']:
                ps=data[rate]['predictions'];ids=[p['sample_id'] for p in ps]
                assert len(ids)==len(set(ids)) and set(ids)==expected
            cfg=json.loads((folder/'config.json').read_text())
            assert cfg['seed']==0
            assert Path(cfg['train_split']).resolve()==(OUT/f'group_folds/fold{fold}_train.jsonl').resolve()
            assert Path(cfg['val_split']).resolve()==(OUT/f'group_folds/fold{fold}_val.jsonl').resolve()
            completed[arm][fold]=data
    common=sorted(set.intersection(*(set(v) for v in completed.values())))
    result={'seed':0,'complete_folds_by_arm':{a:sorted(d) for a,d in completed.items()},'matched_folds':common,
        'all_five_folds_complete':len(common)==5,'partially_written_files_skipped':unreadable,
        'limitations':'One seed; checkpoint selection uses validation; known-duplicate grouping is incomplete; no calibrated leaderboard forecast. Partial matched cohorts are descriptive only.'}
    table=['# Grouped-fold confirmation scorecard','',f'Complete matched folds: {common}. Seed: 0. '+('All five folds complete.' if len(common)==5 else '**Incomplete: not a full-CV result.**'),'',
        '| Rate | Base | Sharp auxiliary | Broad auxiliary | Broad minus base | Broad minus sharp |','|---|---:|---:|---:|---:|---:|']
    if common:
        result['rates']={}
        for rate in ['k1','k2','k3']:
            preds={a:[p for fold in common for p in completed[a][fold][rate]['predictions']] for a in ARMS}
            arrays={a:A.clip_arrays(ps) for a,ps in preds.items()};ids=arrays['base']['ids']
            assert len(set(ids))==len(ids) and all(v['ids']==ids for v in arrays.values())
            group_ids=sorted({meta['group_by_id'][sid] for sid in ids})
            groups=[np.array([i for i,sid in enumerate(ids) if meta['group_by_id'][sid]==g]) for g in group_ids]
            rng=np.random.default_rng(20260927);scores={a:[] for a in ARMS}
            for _ in range(2000):
                idx=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])[None]
                for a in ARMS:scores[a].append(float(A.score_idx(arrays[a],idx)[0]))
            metrics={a:C.breakdown(ps) for a,ps in preds.items()};pairs={}
            for candidate,control in [('aux_sharp','base'),('aux_broad','base'),('aux_broad','aux_sharp')]:
                diff=np.array(scores[candidate])-scores[control]
                pairs[candidate+'_vs_'+control]={'delta':metrics[candidate]['overall']['score']-metrics[control]['overall']['score'],
                    'ci95':np.quantile(diff,[.025,.975]).tolist(),'groups':len(groups),'method':'2000 paired known-duplicate-group bootstraps; no search correction'}
            result['rates'][rate]={'breakdown':metrics,'pairs':pairs,'per_fold':{str(f):{a:completed[a][f][rate]['metrics']['overall']['score'] for a in ARMS} for f in common}}
            vals=[metrics[a]['overall']['score'] for a in ARMS]
            table.append(f'| {rate} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | {vals[2]-vals[0]:+.6f} | {vals[2]-vals[1]:+.6f} |')
    if common:
        table += ['', '## Source and sampling-rate tradeoffs', '',
                  'Composite-score deltas versus matched base. Small source slices are descriptive. A positive pooled score does not imply uniform robustness.', '',
                  '| Source | Clips | Sharp native | Sharp half | Sharp third | Broad native | Broad half | Broad third |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        native=result['rates']['k1']['breakdown']
        for source in sorted(k for k in native['base'] if k.startswith('source:') and k != 'source:non-NEXAR'):
            values=[]
            for arm in ['aux_sharp','aux_broad']:
                for rate in ['k1','k2','k3']:
                    by_arm=result['rates'][rate]['breakdown']
                    values.append(by_arm[arm][source]['score']-by_arm['base'][source]['score'])
            table.append(f"| {source.split(':')[1]} | {native['base'][source]['n']} | " + ' | '.join(f'{v:+.6f}' for v in values) + ' |')
        table += ['', '## Event-error tails', '',
                  'Catastrophic means absolute frame error greater than 10% of available clip length, following the repository metric. Lower is better; this is not a seconds-based threshold.', '',
                  '| Rate | Arm | ENTRY catastrophic | COLLISION catastrophic |',
                  '|---|---|---:|---:|']
        for rate in ['k1','k2','k3']:
            for arm in ARMS:
                m=result['rates'][rate]['breakdown'][arm]['overall']
                table.append(f"| {rate} | {arm} | {m['entry_catastrophic']:.6f} | {m['collision_catastrophic']:.6f} |")
    table+=['','Scores pool clip predictions and recompute macro-F1; they are not averages of fold scores. No candidate is promoted automatically. See confirmation_aggregate.json for source breakdowns, paired uncertainty, and per-fold scores.']
    C.dump(OUT/'confirmation_aggregate.json',result)
    (OUT/'CONFIRMATION_SCORECARD.md').write_text('\n'.join(table)+'\n')
    print('Matched complete folds:',common,'full five-fold result:',len(common)==5,flush=True)

if __name__=='__main__':main()
