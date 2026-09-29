"""Isolated CPU-only inference screen; run as module from repository root."""
import hashlib
import json
import time
from pathlib import Path
from functools import lru_cache

import numpy as np
from stage2.aux_signal_experiments import ensemble_test as ET
from stage2.phase_study import analyze as A
from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments import decode_motion as D

OUT = Path(__file__).resolve().parent
MANIFEST = json.loads((OUT / 'manifest.json').read_text())
AUDIT = C.REPO / 'stage2/generalization/results/dup_audit.json'
audit = json.loads(AUDIT.read_text())
EXCL = {t[0] for t in audit['lab_vs_lab'] if t[2] >= .5 and t[4] != t[5]} | {t[1] for t in audit['unl_vs_lab'] if t[2] >= .5}
original_shift = D.shift_scores

@lru_cache(maxsize=512)
def cached_shift(sid, frames):
    return original_shift({'sample_id': sid, 'frames': frames})

D.shift_scores = lambda p: cached_shift(p['sample_id'], tuple(p['frames']))

def pool(items, mode, beta=.5, snap=True):
    p = {k: v for k, v in items[0].items() if not k.endswith('_logits') and k != 'phase_logp'}
    for event in ['entry', 'collision']:
        logp = np.array([ET.PE.lp(x[event + '_logits']) for x in items])
        if mode == 'geometric':
            scores = logp.mean(0)
        else:
            probs = np.exp(logp)
            scores = np.log((probs.mean(0) if mode == 'arithmetic' else np.median(probs, axis=0)) + 1e-12)
        p[event + '_logp'] = ET.PE.lp(scores)
    p['entry_side'] = int(np.mean([x['side_prob_right'] for x in items]) >= .5)
    p['evasion_space'] = int(np.mean([x['evasion_prob'] for x in items]) >= .5)
    return D.decode(p, None, 0, 0, beta, False, snap)

def paired(a, b):
    aa, bb = A.clip_arrays(a), A.clip_arrays(b)
    assert aa['ids'] == bb['ids']
    sources = {p['sample_id']: C.source(p) for p in a}
    src = np.array([sources[s] for s in aa['ids']])
    rng = np.random.default_rng(20260927)
    idx = np.concatenate([rng.choice(np.flatnonzero(src == s), (2000, int((src == s).sum())), replace=True) for s in sorted(set(src))], axis=1)
    d = A.score_idx(aa, idx) - A.score_idx(bb, idx)
    return {'delta': C.metrics(a)['score'] - C.metrics(b)['score'], 'ci95': np.quantile(d, [.025, .975]).tolist(), 'method': '2000 source-stratified paired clip bootstraps; no multiple-search correction'}

def main():
    t0 = time.monotonic()
    outputs = {}; provenance = []
    subsets = [(0, 1, 2), (0, 1), (0, 2), (1, 2)]
    for recipe, families in MANIFEST['recipes'].items():
        for k in range(5):
            data = {}
            for family in families:
                for seed in range(3):
                    path = ET.PE.root(family) / family / 'cv' / f'fold{k}_seed{seed}' / 'predictions.json'
                    raw = path.read_bytes()
                    provenance.append({'path': str(path.relative_to(C.REPO)), 'sha256': hashlib.sha256(raw).hexdigest()})
                    data[family, seed] = {p['sample_id']: p for p in json.loads(raw)}
            ids = sorted(data[families[0], 0])
            assert all(sorted(d) == ids for d in data.values())
            for sid in ids:
                frames = data[families[0], 0][sid]['frames']
                assert all(d[sid]['frames'] == frames for d in data.values())
                for seeds in subsets:
                    items = [data[f, seed][sid] for f in families for seed in seeds]
                    for mode in MANIFEST['hypotheses']:
                        key = recipe + '/' + mode + '/' + ''.join(map(str, seeds))
                        p = pool(items, mode); p['fold'] = k
                        outputs.setdefault(key, []).append(p)
            print(recipe, 'fold', k, 'complete', flush=True)
    results = {}
    for key, ps in outputs.items():
        clean = [p for p in ps if p['sample_id'] not in EXCL]
        reference = outputs['v8/arithmetic/' + key.rsplit('/', 1)[1]]
        refclean = [p for p in reference if p['sample_id'] not in EXCL]
        results[key] = {'original': C.breakdown(ps), 'clean': C.breakdown(clean), 'paired_clean_vs_v8': paired(clean, refclean)}
        if key.endswith('/012'):
            print(key, 'clean', round(results[key]['clean']['overall']['score'],6), results[key]['paired_clean_vs_v8'], flush=True)
    artifact = {'manifest': MANIFEST, 'audit_sha256': hashlib.sha256(AUDIT.read_bytes()).hexdigest(), 'excluded_ids': sorted(EXCL), 'input_files': provenance, 'elapsed_seconds': time.monotonic()-t0, 'results': results}
    C.dump(OUT / 'inference_results.json', artifact)
    C.dump(OUT / 'decoded_predictions.json', outputs)
    print('Finished in', artifact['elapsed_seconds'], 'seconds', flush=True)

if __name__ == '__main__':
    main()
