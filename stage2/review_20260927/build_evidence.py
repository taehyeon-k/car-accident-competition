"""Read-only analysis of saved experiments; writes only this review's artifacts.
Run from repo root: PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python -m stage2.review_20260927.build_evidence
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import csv
import itertools
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from stage2.phase_study import analyze as A
from stage2.long_context_v2_experiments import common as C
from stage2.generalization.loso_analyze import summarize as loso_summary

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
GEN = ROOT / 'stage2/generalization/results'

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')

def bootstrap(a, b):
    seeds = sorted(set(a) & set(b))
    aa = [A.clip_arrays([p for k in range(5) for p in a[s][k]]) for s in seeds]
    bb = [A.clip_arrays([p for k in range(5) for p in b[s][k]]) for s in seeds]
    assert all(x['ids'] == y['ids'] == aa[0]['ids'] for x,y in zip(aa,bb))
    # Clip paired, stratified by source; seed sets explicitly matched.
    ps = sorted([p for k in range(5) for p in a[seeds[0]][k]], key=lambda p:p['sample_id'])
    src = np.array([C.source(p) for p in ps])
    rng = np.random.default_rng(20260927)
    idx = np.concatenate([rng.choice(np.flatnonzero(src == s), (2000, int((src == s).sum())), replace=True) for s in sorted(set(src))], axis=1)
    d = np.mean([A.score_idx(x, idx) for x in aa], 0) - np.mean([A.score_idx(x, idx) for x in bb], 0)
    full = np.arange(len(ps))[None]
    point = np.mean([A.score_idx(x, full) for x in aa]) - np.mean([A.score_idx(x, full) for x in bb])
    return {'matched_seeds':seeds,'delta':float(point),'ci95':np.quantile(d,[.025,.975]).tolist(),
            'method':'2000 paired clip bootstrap replicates, stratified by source, averaged over matched seeds; no search correction'}

def correlation(rows, field):
    rs=[r for r in rows if r.get(field) is not None]
    x=np.array([r[field] for r in rs]); y=np.array([r['leaderboard'] for r in rs])
    p=float(pearsonr(x,y).statistic); s=float(spearmanr(x,y).statistic)
    loo=[float(pearsonr(np.delete(x,i),np.delete(y,i)).statistic) for i in range(len(x))] if len(x)>3 else []
    return {'n':len(rs),'models':[r['name'] for r in rs],'pearson':p,'spearman':s,
            'leave_one_out_pearson_range':[min(loo),max(loo)] if loo else None,
            'warning':'descriptive only; non-independent selected submissions; mixed recipe/proxy protocols; not predictive calibration'}

def main():
    OUT.mkdir(exist_ok=True)
    timestamp=datetime.now(timezone.utc).isoformat()
    arms=['E4_sa','OT_sa','XU_mc','XU_mca','E2_sa','E2_sa_xu','XN4_sa','XN4_sa_xu','E4_sa_se2']
    cv={}; results={}
    for arm in arms:
        _,data,_=A.load_run(arm,GEN)
        data={s:f for s,f in data.items() if int(s)<3}; cv[arm]=data
        if not data: continue
        stats=[C.breakdown([p for k in range(5) for p in f[k]]) for f in data.values()]
        results[arm]={'complete_seeds':sorted(data),'overall_mean':{k:float(np.mean([m['overall'][k] for m in stats])) for k in stats[0]['overall']},
                      'overall_score_std':float(np.std([m['overall']['score'] for m in stats])),
                      'nexar_mean_score':float(np.mean([m['source:NEXAR']['score'] for m in stats])),
                      'seed_ensemble':C.breakdown(A.seed_ensemble(data))}
    for arm,ctrl in [('OT_sa','E4_sa'),('XU_mc','E4_sa'),('XU_mca','E4_sa'),('E2_sa_xu','E2_sa'),('XN4_sa_xu','XN4_sa'),('E4_sa_se2','E4_sa')]:
        if cv[arm]: results[arm]['paired_vs_'+ctrl]=bootstrap(cv[arm],cv[ctrl])
    loso={n:loso_summary(n) for n in ['LOSO_E4_sa:0,1,2','LOSO_OT_sa:0,1,2']}
    dump('latest_matched_results.json',{'snapshot_utc':timestamp,'cv':results,'loso':loso})
    for name,r in results.items():
        print(name,len(r['complete_seeds']),round(r['overall_mean']['score'],6),{k:v for k,v in r.items() if k.startswith('paired')})

    v6=json.loads((ROOT/'submission_tools/v6_stage2/parity_results.json').read_text())
    v7=json.loads((ROOT/'stage2/aux_signal_experiments/results/goal_E4_E2_XN4.json').read_text())
    rows=[
      {'name':'v1 ASFormer','fixed_all':.6893,'fixed_nexar':.522,'cv_all':None,'cv_nexar':None,'leaderboard':.4410,'provenance':'reports/stage2_leaderboard_vs_nexar.md; same submitted head'},
      {'name':'v3 P2 refit','fixed_all':.7567,'fixed_nexar':.5004,'cv_all':None,'cv_nexar':None,'leaderboard':.4618,'provenance':'batch-1 279-clip proxy; submitted model refit on 349'},
      {'name':'NEXAR specialist','fixed_all':.6701,'fixed_nexar':.7438,'cv_all':None,'cv_nexar':None,'leaderboard':.4370,'provenance':'nexar65 REPORT first_three_prior; recorded LB'},
      {'name':'Length hybrid','fixed_all':.8017,'fixed_nexar':.7104,'cv_all':None,'cv_nexar':None,'leaderboard':.4154,'provenance':'length_gated REPORT all-seed candidate; recorded LB'},
      {'name':'v5 motion ensemble','fixed_all':.767,'fixed_nexar':.619,'cv_all':.7688,'cv_nexar':.6702,'leaderboard':.5314,'provenance':'recorded submitted-recipe validation; reports and HANDOFF'},
      {'name':'v6 event-specific','fixed_all':v6['val_overall']['score'],'fixed_nexar':v6['val_NEXAR']['score'],'cv_all':.7884,'cv_nexar':.7103,'leaderboard':.5277,'provenance':'fixed package parity; CV config label; LB supplied by user 2026-09-27'},
      {'name':'v7 residual+expansion','fixed_all':v7['fixed']['overall']['score'],'fixed_nexar':v7['fixed']['source:NEXAR']['score'],'cv_all':v7['cv']['overall']['score'],'cv_nexar':v7['cv']['source:NEXAR']['score'],'leaderboard':.5464,'provenance':'HANDOFF user-reported LB; offline recipe proxy differs in seed count from full-data package'},
      {'name': 'v8 stride augmentation', 'fixed_all': None, 'fixed_nexar': None, 'cv_all': 0.7874193962706539, 'cv_nexar': 0.7071342619219573, 'leaderboard': 0.59293, 'provenance': 'LB supplied by user 2026-09-27; three-family seeds 0-2 OOF proxy; package uses four seeds per family refit on all 349; in-sample parity excluded'},
    ]
    with (OUT/'leaderboard_pairs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    corr={field:correlation(rows,field) for field in ['fixed_all','fixed_nexar','cv_all','cv_nexar']}
    corr['fixed_all_without_specialists']=correlation([r for r in rows if r['name'] not in ['NEXAR specialist','Length hybrid']],'fixed_all')
    dump('leaderboard_correlations.json',{'snapshot_utc':timestamp,'rows':rows,'correlations':corr})
    print('CORRELATIONS',json.dumps(corr))

    # Preserve all documented tables, with source attribution and section context.
    sources=sorted((ROOT/'reports').glob('stage2*.md'))
    sources=[p for p in sources if not p.name.startswith('stage2_24h_review')]
    sources+=sorted((ROOT/'stage2/reports').glob('*.md'))
    sources += [ROOT/'stage2'/d/'REPORT.md' for d in ['long_video_experiments','nexar65_experiments','length_gated_experiments']]
    archive=['**Stage 2 documented experiment-table archive — snapshot '+timestamp+'**',
             'Historical tables are preserved, including superseded claims. The main review explains protocol mismatches and later corrections. This is not a single comparable leaderboard.']
    for p in sources:
        if not p.exists():continue
        text=p.read_text();lines=text.splitlines(); archive+=['',f'**Source: [{p.relative_to(ROOT)}]({p})**','']
        heading=''; previous=False
        for line in lines:
            if line.startswith('#'): heading=line.lstrip('# ').strip()
            if line.startswith('|'):
                if not previous: archive+=['',f'*{heading}*','']
                archive.append(line)
            previous=line.startswith('|')
    (OUT/'all_reported_experiment_tables.md').write_text('\n'.join(archive)+'\n')

    # Atomic-run inventory: do not average these scores across folds.
    inventory=[]
    for p in sorted((ROOT/'stage2').rglob('metrics.json')):
        try:d=json.loads(p.read_text())
        except (ValueError,OSError):continue
        if not isinstance(d,dict):continue
        m=d.get('breakdown',{}).get('overall',{})
        inventory.append({'path':str(p.relative_to(ROOT)),'mtime_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),
                          'n':m.get('n',''),'score':m.get('score',''),'entry_acc':m.get('entry_acc',''),'collision_acc':m.get('collision_acc',''),
                          'side_f1':m.get('side_f1',''),'evasion_f1':m.get('evasion_f1',''),
                          'note':'saved per-run metrics; empty fields indicate another schema; may include full-data/in-sample runs'})
    with (OUT/'atomic_run_inventory.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=inventory[0]);w.writeheader();w.writerows(inventory)
    print('INVENTORY',len(inventory),'atomic metrics files;',len(sources),'report sources')

    print('For the chart, run plot_evidence.py using an environment with matplotlib.')

if __name__=='__main__': main()
