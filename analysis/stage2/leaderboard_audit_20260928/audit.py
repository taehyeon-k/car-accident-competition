"""Reproducible retrospective leaderboard audit. No training or GPU inference.
Run from repository root: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /venv/main/bin/python analysis/stage2/leaderboard_audit_20260928/audit.py
Feature hypotheses are historically selected: even nested resampling is not a prospective test.
"""
import csv, hashlib, itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, t
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from stage2.generalization import robustness_analyze as A
C = A.C
OUT = Path(__file__).resolve().parent
RES = ROOT / 'stage2/generalization/results'
NAMES = ['v5','v6','v7','v8','v10','v12']
Y = np.array([.5314,.5277,.5464,.59293,.5590,.5635])

def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n')

def fit_predict(X, y, test, alpha=0):
    if X.shape[1] == 0: return np.full(len(test),y.mean())
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-10, 1, sd)
    z, zt = (X-mu)/sd, (test-mu)/sd
    if alpha:
        w = np.linalg.solve(z.T@z + alpha*np.eye(X.shape[1]),z.T@(y-y.mean()))
    else: w = np.linalg.lstsq(z,y-y.mean(),rcond=None)[0]
    return y.mean()+zt@w

def stats(y,p):
    return dict(mae=float(np.mean(abs(y-p))),rmse=float(np.sqrt(np.mean((y-p)**2))),max_error=float(max(abs(y-p))))

def cv(X, y, alpha=0):
    return np.array([fit_predict(np.delete(X,i,0),np.delete(y,i),X[i:i+1],alpha)[0] for i in range(len(y))])

def select(X,y):
    # Full feature selection inside each training set, using inner LOO prediction error.
    errors = [stats(y,cv(X[:,j:j+1],y))['mae'] for j in range(X.shape[1])]
    return int(np.argmin(errors))

profile_path = RES/'robustness_profile.json'
profile = json.loads(profile_path.read_text())
old = json.loads((RES/'robustness_analysis_obl.json').read_text())
ct, excl = A.clip_table(), A.excluded()
features = {}; validation = {}; reference_ids = None
for rec, conditions in profile.items():
    clean = {}
    for condition, rows in conditions.items():
        ids = [p['sample_id'] for p in rows]
        assert len(ids)==len(set(ids))==349, (rec,condition)
        if reference_ids is None: reference_ids=set(ids)
        assert set(ids)==reference_ids
        for p in rows:
            r=ct[p['sample_id']]
            assert p['source_id'].split(':')[0]==r['source']
            assert p['collision_gt']-p['entry_gt']==r['gap_frames']
        clean[condition]=[p for p in rows if p['sample_id'] not in excl]
    sl=A.slices(conditions,ct,excl)
    features[rec]=A.recipe_features(conditions,ct,excl,sl)
    validation[rec]={k:C.metrics(v) for k,v in clean.items()}
    # Correct the legacy view that gives every AIHUB clip double weight.
    maps={k:{p['sample_id']:p for p in rows} for k,rows in clean.items()}
    alternatives=[]
    for ai_k in [1,2]:
        rows=[]
        for sid,p in maps['k1_crop0.0'].items():
            src=C.source(p); k=ai_k if src=='AIHUB' else (1 if src=='CCD' else 3)
            rows.append(maps[f'k{k}_crop0.0'][sid])
        alternatives.append(C.metrics(rows)['score'])
    features[rec]['10fps_equal_clip_views']=float(np.mean(alternatives))

keys=sorted(set.intersection(*(set(features[r]) for r in NAMES)) - {'10fps_equal_clip_views'})
Xall=np.array([[features[r][k] for k in keys] for r in NAMES])
assert np.isfinite(Xall).all()
# Compare every reconstructed legacy feature against the saved analysis.
maxdiff=max(abs(features[r][k]-old['recipes'][r]['features'][k]) for r in NAMES for k in keys)
dump('data_integrity.json',dict(profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),n_raw=349,n_clean=len(reference_ids-excl),n_exclusion_ids=len(excl),legacy_feature_max_difference=maxdiff,leaderboard=dict(zip(NAMES,Y)),leaderboard_source='HANDOFF.md section 8; v8 precision from review_20260927/leaderboard_pairs.csv',conditions=validation))

specs={
 'mean_baseline':([],0),
 'native':(['score@k1_crop0.0'],0),
 'third_rate':(['score@k3_crop0.0'],0),
 'rate_drop':(['drop_native_to_k3'],0),
 'ten_fps_legacy':(['10fps_view'],0),
 'ten_fps_equal_clip':(['10fps_equal_clip_views'],0),
 'rate_plus_crop_entry':(['score@k3_crop0.0','entry@k1_crop0.25'],0),
 'rate_plus_crop_ridge1':(['score@k3_crop0.0','entry@k1_crop0.25'],1),
 'rate_plus_extras':(['score@k3_crop0.0','extras'],1),
}
for r in features: features[r]['extras']=float(r in ['v10','v11','v12'])
results={}; predictions=[]
for label,(ks,alpha) in specs.items():
    X=np.array([[features[r][k] for k in ks] for r in NAMES]).reshape(6,len(ks))
    p=cv(X,Y,alpha)
    full=fit_predict(X,Y,X,alpha)
    blocks={}
    for tag,test in [('middle_v6_v7',[1,2]),('middle_v7_v8',[2,3]),('extras_family_v10_v12',[4,5]),('early_family_v5_v6',[0,1])]:
        train=[i for i in range(6) if i not in test]
        pred=fit_predict(X[train],Y[train],X[test],alpha)
        blocks[tag]={'train':[NAMES[i] for i in train],'test':[NAMES[i] for i in test],'prediction':pred,'actual':Y[test],**stats(Y[test],pred)}
    forward=[]
    for i in [3,4,5]:
        pred=fit_predict(X[:i],Y[:i],X[i:i+1],alpha)[0]
        forward.append(dict(test=NAMES[i],prediction=pred,actual=Y[i],error=pred-Y[i]))
    pair_errors=[]
    pair_rows=[]
    for pair in itertools.combinations(range(6),2):
        train=[i for i in range(6) if i not in pair];test=list(pair)
        pred=fit_predict(X[train],Y[train],X[test],alpha)
        pair_errors.extend((pred-Y[test]).tolist())
        pair_rows.append(dict(test=[NAMES[i] for i in test],prediction=pred,actual=Y[test]))
    results[label]=dict(all_15_pair_masks={'mae':float(np.mean(np.abs(pair_errors))),'max_error':float(np.max(np.abs(pair_errors))),'folds':pair_rows},features=ks,alpha=alpha,loo=stats(Y,p),loo_predictions=dict(zip(NAMES,p)),in_sample=stats(Y,full),blocks=blocks,forward=forward)
    for r,y,pr in zip(NAMES,Y,p): predictions.append(dict(method=label,submission=r,actual=y,prediction=pr,error=pr-y))

nested=[]
for i in range(6):
    train=[j for j in range(6) if j!=i]
    chosen=select(Xall[train],Y[train])
    pred=fit_predict(Xall[train,chosen:chosen+1],Y[train],Xall[i:i+1,chosen:chosen+1])[0]
    nested.append(dict(test=NAMES[i],feature=keys[chosen],prediction=pred,actual=Y[i]))
results['nested_single_feature_search']=dict(loo=stats(Y,np.array([r['prediction'] for r in nested])),folds=nested)
# Exact max-|r| permutation accounts for searching all correlated features.
Z=Xall-Xall.mean(0); Z/=np.linalg.norm(Z,axis=0)
yc=Y-Y.mean(); yc/=np.linalg.norm(yc)
corr=Z.T@yc
permutations=np.array(list(itertools.permutations(Y)))
yp=permutations-permutations.mean(1,keepdims=True); yp/=np.linalg.norm(yp,axis=1,keepdims=True)
perm_corr=yp@Z
maxnull=np.max(abs(perm_corr),axis=1)
correlations=[]
for j,k in enumerate(keys):
    correlations.append(dict(feature=k,pearson=corr[j],spearman=float(np.corrcoef(rankdata(Xall[:,j]),rankdata(Y))[0,1]),permutation_p=float(np.mean(abs(perm_corr[:,j])>=abs(corr[j])-1e-12)),search_adjusted_p=float(np.mean(maxnull>=abs(corr[j])-1e-12))))
correlations.sort(key=lambda r:-abs(r['pearson']))
dump('correlations.json',dict(n_submissions=6,n_features=len(keys),permutations=len(permutations),features=correlations))
dump('validation.json',results)
dump('features.json',features)
with (OUT/'heldout_predictions.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
# Existing two-variable forecast: coefficients, classical conditional interval, jackknife sensitivity.
ks=specs['rate_plus_crop_entry'][0]
X=np.array([[1]+[features[r][k] for k in ks] for r in NAMES]); b=np.linalg.lstsq(X,Y,rcond=None)[0]
s2=np.sum((Y-X@b)**2)/(len(Y)-X.shape[1]); inv=np.linalg.inv(X.T@X)
forecast={}
for rec in ['v9','v11','v13','v13nc']:
    x=np.array([1]+[features[rec][k] for k in ks]); pred=x@b
    half=t.ppf(.975,3)*np.sqrt(s2*(1+x@inv@x))
    jack=[fit_predict(np.delete(X[:,1:],i,0),np.delete(Y,i),x[None,1:])[0] for i in range(6)]
    forecast[rec]=dict(prediction=pred,conditional_95_interval=[pred-half,pred+half],leave_one_out_fit_range=[min(jack),max(jack)])
coefficient_jackknife={NAMES[i]:dict(zip(['intercept']+ks,np.linalg.lstsq(np.delete(X,i,0),np.delete(Y,i),rcond=None)[0])) for i in range(6)}
dump('conditional_forecasts.json',dict(coefficient_jackknife=coefficient_jackknife,coefficients=dict(zip(['intercept']+ks,b)),warning='Retrospective selected model; intervals exclude feature selection, proxy mismatch and domain shift. Not validated leaderboard coverage.',forecasts=forecast))
# Descriptive early fixed-split cohort: do not combine with CV robustness proxies.
with (ROOT/'stage2/review_20260927/leaderboard_pairs.csv').open() as f: early=[r for r in csv.DictReader(f) if r['fixed_all']]
ey=np.array([float(r['leaderboard']) for r in early]); er={}
for metric in ['fixed_all','fixed_nexar']:
    ex=np.array([[float(r[metric])] for r in early]); ep=cv(ex,ey)
    er[metric]=dict(n=len(early),pearson=float(np.corrcoef(ex[:,0],ey)[0,1]),loo=stats(ey,ep))
er['mean_baseline']=stats(ey,cv(np.empty((len(ey),0)),ey))
dump('early_fixed_split.json',er)
# Standalone SVG scatter plot; no plotting dependency required.
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="980" height="410" viewBox="0 0 980 410"><rect width="980" height="410" fill="white"/>']
for panel,method in enumerate(['third_rate','rate_plus_crop_entry']):
    ox=65+panel*485
    def xy(a,b):return ox+(a-.49)/.15*360,345-(b-.49)/.15*280
    svg.append(f'<text x="{ox}" y="25" font-size="17">{method}</text>')
    for tick in [.50,.55,.60]:
        px,py=xy(tick,tick)
        svg.append(f'<path d="M {px} 65 V 345 M {ox} {py} H {ox+360}" stroke="#ddd"/><text x="{px-15}" y="366">{tick:.2f}</text><text x="{ox-40}" y="{py+5}">{tick:.2f}</text>')
    svg.append(f'<path d="M {ox} 65 V 345 H {ox+360} M {ox} 345 L {ox+360} 65" fill="none" stroke="#888"/>')
    for rec,pred in results[method]['loo_predictions'].items():
        px,py=xy(Y[NAMES.index(rec)],pred)
        svg.append(f'<circle cx="{px}" cy="{py}" r="5" fill="#2563eb"/><text x="{px+7}" y="{py-5}">{rec}</text>')
    svg.append(f'<text x="{ox+50}" y="395">Actual Stage 2 leaderboard (y: held-out prediction)</text>')
svg.append('</svg>');(OUT/'heldout_predictions.svg').write_text(''.join(svg))
print('Integrity max feature difference',maxdiff,'clean',len(reference_ids-excl))
for k,v in results.items():print(k,v['loo'])
print('Best correlation',correlations[0])
print('Forecast',forecast)
print('Two-variable blocks',results['rate_plus_crop_entry']['blocks'])
print('Two-variable forward',results['rate_plus_crop_entry']['forward'])
