"""Score the isolated GPU seed3 only when both arms have all grouped folds."""
import json
from pathlib import Path

import numpy as np

from stage2.long_context_v2_experiments import common as C
from stage2.phase_study import analyze as A

OUT = Path(__file__).resolve().parent
ARMS = ('gpu_base', 'gpu_aux_sharp')


def main():
    group = json.loads((OUT / 'group_folds/metadata.json').read_text())['group_by_id']
    matched = {}
    fold_rows = []
    for fold in range(5):
        pair = {}
        expected = {r['sample_id'] for r in C.rows(str(OUT / f'group_folds/fold{fold}_val.jsonl'))}
        for arm in ARMS:
            folder = OUT / 'training' / arm / 'cv' / f'fold{fold}_seed3'
            if not (folder / 'rate_eval.json').exists():
                break
            d = json.loads((folder / 'rate_eval.json').read_text())
            cfg = json.loads((folder / 'config.json').read_text())
            mismatches = d['native_parity_mismatches']
            assert len(mismatches) <= 2 and all(k in ('entry_frame', 'collision_frame')
                   and abs(a-b) == 1 for _, k, a, b in mismatches)
            assert cfg['seed'] == 3 and cfg['training_device'] == 'cuda:0'
            assert Path(cfg['train_split']).resolve() == OUT / f'group_folds/fold{fold}_train.jsonl'
            assert Path(cfg['val_split']).resolve() == OUT / f'group_folds/fold{fold}_val.jsonl'
            for rate in ('k1', 'k2', 'k3'):
                ids = [p['sample_id'] for p in d[rate]['predictions']]
                assert len(ids) == len(set(ids)) and set(ids) == expected
            pair[arm] = d
        if len(pair) != 2:
            continue
        matched[fold] = pair
        for rate in ('k1', 'k2', 'k3'):
            b = pair[ARMS[0]][rate]['metrics']['overall']['score']
            a = pair[ARMS[1]][rate]['metrics']['overall']['score']
            fold_rows.append({'fold': fold, 'rate': rate, 'base': b, 'sharp': a, 'delta': a-b})
    result = {'seed': 3, 'device': 'cuda:0', 'matched_folds': sorted(matched),
              'complete_five_folds': len(matched) == 5, 'fold_rows': fold_rows,
              'native_parity_mismatches': {f'{arm}/fold{fold}': matched[fold][arm]['native_parity_mismatches']
                                            for fold in matched for arm in ARMS},
              'limitations': 'Different seed and device from CPU runs; no leaderboard calibration. Bootstrap is descriptive and conditional on checkpoint selection.'}
    if len(matched) == 5:
        rates = {}
        for rate in ('k1', 'k2', 'k3'):
            ps = {arm: [p for fold in range(5) for p in matched[fold][arm][rate]['predictions']]
                  for arm in ARMS}
            arrays = {arm: A.clip_arrays(p) for arm, p in ps.items()}
            ids = arrays[ARMS[0]]['ids']
            assert len(ids) == 349 and len(set(ids)) == 349 and arrays[ARMS[1]]['ids'] == ids
            groups = {}
            for i, sid in enumerate(ids):
                groups.setdefault(group[sid], []).append(i)
            groups = list(groups.values())
            rng = np.random.default_rng(20260930)
            deltas = []
            for _ in range(2000):
                idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])[None]
                deltas.append(float(A.score_idx(arrays[ARMS[1]], idx)[0] -
                                    A.score_idx(arrays[ARMS[0]], idx)[0]))
            metrics = {arm: C.breakdown(preds) for arm, preds in ps.items()}
            base = metrics[ARMS[0]]['overall']['score']
            sharp = metrics[ARMS[1]]['overall']['score']
            rates[rate] = {'base': metrics[ARMS[0]], 'sharp': metrics[ARMS[1]],
                           'delta': sharp-base,
                           'ci95': np.quantile(deltas, [.025, .975]).tolist()}
        result['rates'] = rates
    C.dump(OUT / 'gpu_seed3_aggregate.json', result)
    print('Matched folds:', sorted(matched), 'complete:', result['complete_five_folds'])
    if result['complete_five_folds']:
        print({r: round(v['delta'], 6) for r, v in result['rates'].items()})


if __name__ == '__main__':
    main()
