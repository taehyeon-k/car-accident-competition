"""Matched baseline/transport evaluation with complete-fold coverage checks."""
import json
from pathlib import Path
import numpy as np
from stage2.long_context_v2_experiments import common as C
from stage2.phase_study import analyze as A

OUT = Path(__file__).resolve().parent


def main():
    meta = json.loads((OUT/'group_folds/metadata.json').read_text())
    completed = {}
    for fold in range(5):
        paths = {arm: OUT/f'training/{arm}/cv/fold{fold}_seed0' for arm in ('base', 'transport')}
        if not all((p/'rate_eval.json').exists() for p in paths.values()):
            continue
        expected = {r['sample_id'] for r in C.rows(str(OUT/f'group_folds/fold{fold}_val.jsonl'))}
        pair = {}
        for arm, folder in paths.items():
            data = json.loads((folder/'rate_eval.json').read_text())
            config = json.loads((folder/'config.json').read_text())
            assert data['native_parity_mismatches'] == []
            assert config['seed'] == 0
            for split in ('train', 'val'):
                assert Path(config[split+'_split']).resolve() == OUT/f'group_folds/fold{fold}_{split}.jsonl'
            if arm == 'transport':
                assert config['target_policy'] == 'native_index_gaussian_mass_nearest_augmented_frame'
            for rate in ('k1', 'k2', 'k3'):
                ids = [p['sample_id'] for p in data[rate]['predictions']]
                assert len(ids) == len(set(ids)) and set(ids) == expected
            pair[arm] = data
        completed[fold] = pair
    result = {'matched_folds': sorted(completed), 'seed': 0, 'all_five_folds_complete': len(completed) == 5,
              'limitations': 'Selected checkpoints; one seed; known duplicate grouping only; no search correction or calibrated leaderboard forecast.', 'rates': {}}
    lines = ['# Target-transport confirmation', '', f'Matched folds: {sorted(completed)}; seed0. '+
             ('All five folds complete.' if len(completed) == 5 else '**Incomplete cohort.**'), '',
             '| Rate | Base | Transport | Delta | 95% paired group interval |', '|---|---:|---:|---:|---|']
    for rate in ('k1', 'k2', 'k3'):
        if not completed:
            break
        ps = {arm: [p for f in sorted(completed) for p in completed[f][arm][rate]['predictions']]
              for arm in ('base', 'transport')}
        arrays = {arm: A.clip_arrays(p) for arm, p in ps.items()}
        ids = arrays['base']['ids']
        assert arrays['transport']['ids'] == ids and len(set(ids)) == len(ids)
        groups = {}
        for i, sid in enumerate(ids):
            groups.setdefault(meta['group_by_id'][sid], []).append(i)
        groups = list(groups.values())
        rng = np.random.default_rng(20260927)
        deltas = []
        for _ in range(2000):
            idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])[None]
            deltas.append(float(A.score_idx(arrays['transport'], idx)[0] - A.score_idx(arrays['base'], idx)[0]))
        metrics = {arm: C.breakdown(p) for arm, p in ps.items()}
        base, candidate = [metrics[a]['overall']['score'] for a in ('base', 'transport')]
        ci = np.quantile(deltas, [.025, .975]).tolist()
        result['rates'][rate] = {'breakdown': metrics, 'delta': candidate-base, 'ci95': ci,
                                 'groups': len(groups), 'bootstrap_samples': 2000}
        lines.append(f'| {rate} | {base:.6f} | {candidate:.6f} | {candidate-base:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |')
    lines += ['', 'Pooled clip predictions with recomputed macro-F1. Source breakdowns and event-error tails are in transport_aggregate.json. No automatic candidate promotion.']
    C.dump(OUT/'transport_aggregate.json', result)
    (OUT/'TRANSPORT_SCORECARD.md').write_text('\n'.join(lines)+'\n')
    print('Matched transport folds:', sorted(completed))


if __name__ == '__main__':
    main()
