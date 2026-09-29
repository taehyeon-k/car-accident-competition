"""Paired group-bootstrap summary of fixed additions to the v8 CV proxy."""
import argparse
import json
from pathlib import Path

import numpy as np

from stage2.phase_study import analyze as A

OUT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', choices=('vitb', 'consistency', 'objappearance', 'v13blend'), default='vitb')
    args = parser.parse_args()
    candidate = {'vitb': 'v8_plus_vitb', 'consistency': 'v8_plus_consistency',
                 'objappearance': 'v8_plus_objappearance', 'v13blend': 'v8_v13_equal'}[args.study]
    data = json.loads((OUT / f'clean_{args.study}_blend_results.json').read_text())
    group_by_id = json.loads((OUT / 'group_folds/metadata.json').read_text())['group_by_id']
    summary = {'protocol': data['protocol'], 'conditions': {},
               'limitations': 'Fixed equal-weight CV screen; descriptive group bootstrap conditional on earlier model search; not a leaderboard forecast.'}
    for rate, entries in sorted(data['conditions'].items()):
        arr = {name: A.clip_arrays(value['predictions']) for name, value in entries.items()}
        ids = arr['v8_plain']['ids']
        assert len(ids) == 284 and len(set(ids)) == 284
        assert all(value['ids'] == ids for value in arr.values())
        grouped = {}
        for i, sid in enumerate(ids):
            grouped.setdefault(group_by_id[sid], []).append(i)
        groups = list(grouped.values())
        rng = np.random.default_rng(20260927 + int(rate[1:]))
        indices = [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])[None]
                   for _ in range(2000)]
        result = {'n_clips': len(ids), 'n_groups': len(groups), 'recipes': {}, 'comparisons': {}}
        for name, value in entries.items():
            m = value['metrics']
            result['recipes'][name] = {s: m[s]['score'] for s in
                                       ('overall', 'source:AIHUB', 'source:CCD', 'source:MMAU',
                                        'source:NEXAR', 'bin:>1000')}
        comparisons = ([('v13_plain', 'v8_plain'), (candidate, 'v8_plain'),
                        (candidate, 'v13_plain')] if args.study == 'v13blend' else
                       [('v8_plus_vits', 'v8_plain'), (candidate, 'v8_plain'),
                        (candidate, 'v8_plus_vits')])
        for candidate_, control in comparisons:
            deltas = [float(A.score_idx(arr[candidate_], idx)[0] -
                            A.score_idx(arr[control], idx)[0]) for idx in indices]
            key = f'{candidate_}_minus_{control}'
            result['comparisons'][key] = {
                'delta': result['recipes'][candidate_]['overall'] - result['recipes'][control]['overall'],
                'ci95': np.quantile(deltas, [.025, .975]).tolist(),
                'nexar_delta': result['recipes'][candidate_]['source:NEXAR'] -
                               result['recipes'][control]['source:NEXAR'],
                'long_delta': result['recipes'][candidate_]['bin:>1000'] -
                              result['recipes'][control]['bin:>1000'],
            }
        summary['conditions'][rate] = result
        print(rate, {k: (round(v['delta'], 6), [round(x, 6) for x in v['ci95']])
                     for k, v in result['comparisons'].items()}, flush=True)
    (OUT / f'clean_{args.study}_blend_scorecard.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
