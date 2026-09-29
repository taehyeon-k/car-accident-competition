"""Locate persistent versus sampling-sensitive errors without fitting a decoder."""
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
from stage2.long_context_v2_experiments import common as C

OUT=Path(__file__).resolve().parent

def main():
    views={}
    for k,crop in [(1,0.),(2,0.),(3,0.),(1,.5)]:
        ps=[]
        for fold in range(5):
            d=json.loads((OUT/'robust_folds'/f'fold{fold}_k{k}_crop{crop}.json').read_text())
            ps+=d['predictions']['arithmetic']
        views[k,crop]={p['sample_id']:p for p in ps}
    fps=C.fps_table(); rows=[]
    for sid,p in sorted(views[1,0.].items()):
        row={'sample_id':sid,'source':C.source(p),'native_fps_for_diagnostics_only':fps[sid]}
        for e in ['entry','collision']:
            errors=[(views[k,0.][sid][e+'_frame']-p[e+'_gt'])/fps[sid] for k in [1,2,3]]
            hits=[abs(err)<=.300001 for err in errors]
            row[e+'_pattern']='all_hit' if all(hits) else 'all_miss' if not any(hits) else 'rate_sensitive'
            row[e+'_native_hit']=hits[0]
            row[e+'_rate_predictions_span_seconds']=max(errors)-min(errors)
            for k,err in zip([1,2,3],errors):row[f'{e}_error_seconds_k{k}']=err
            row[e+'_crop_error_seconds']=(views[1,.5][sid][e+'_frame']-p[e+'_gt'])/fps[sid]
            row[e+'_native_miss_crop_hit']=not hits[0] and abs(row[e+'_crop_error_seconds'])<=.300001
        rows.append(row)
    with (OUT/'rate_error_audit.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary={}
    for source in ['ALL','AIHUB','CCD','MMAU','NEXAR']:
        rs=[r for r in rows if source=='ALL' or r['source']==source]
        summary[source]={'n':len(rs)}
        for e in ['entry','collision']:
            summary[source][e]={'patterns':dict(Counter(r[e+'_pattern'] for r in rs)),
                'native_miss_recovered_at_lower_rate':sum(not r[e+'_native_hit'] and r[e+'_pattern']=='rate_sensitive' for r in rs),
                'native_miss_recovered_with_crop':sum(r[e+'_native_miss_crop_hit'] for r in rs)}
    # A reproducible review panel, deliberately separated from training labels.
    panel=[]
    for source in ['AIHUB','CCD','MMAU','NEXAR']:
        for pattern in ['all_hit','rate_sensitive','all_miss']:
            candidates=[r for r in rows if r['source']==source and r['entry_pattern']==pattern]
            candidates.sort(key=lambda r:hashlib.sha256(('audit20260927:'+r['sample_id']).encode()).hexdigest())
            panel += [{**r,'audit_status':'unreviewed','label_changes_authorized':False} for r in candidates[:4]]
    (OUT/'rate_error_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'label_review_panel.json').write_text(json.dumps({'selection':'Up to four clips per source x ENTRY rate-error pattern, stable hash ordering; diagnostic panel not a random population estimate; includes successful controls','rows':panel},indent=2)+'\n')
    print(json.dumps(summary,indent=2));print('Review panel clips:',len(panel))

if __name__=='__main__':main()
