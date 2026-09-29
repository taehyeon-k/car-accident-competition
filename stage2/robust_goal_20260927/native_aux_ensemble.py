"""Fixed equal-weight native OOF ensembles from saved logits; no model inference."""
import json
from pathlib import Path
import numpy as np
from stage2.long_context_v2_experiments import common as C

OUT = Path(__file__).resolve().parent
ARMS = ('base', 'aux_sharp', 'aux_broad')
PAIRS = (('aux_sharp', 'aux_broad'), ('base', 'aux_sharp'), ('base', 'aux_broad'))


def probs(x):
    x = np.asarray(x, dtype=np.float64)
    x -= x.max()
    x = np.exp(x)
    return x/x.sum()


def decode(entry, collision):
    # Same prefix-maximum ordered pair as spotting_experiments.objective.constrained_anchors.
    best = -np.inf
    best_i = 0
    best_pair = (-np.inf, -1)
    for j, value in enumerate(collision):
        if entry[j] > best:
            best = entry[j]
            best_i = j
        pair = (best + value, j)
        if pair[0] > best_pair[0]:
            best_pair = (pair[0], best_i)
            collision_j = j
    return best_pair[1], collision_j


def main():
    scores = {name: [] for name in ARMS + tuple('+'.join(p) for p in PAIRS)}
    for fold in range(5):
        data = {arm: json.loads((OUT/f'training/{arm}/cv/fold{fold}_seed0/predictions.json').read_text())
                for arm in ARMS}
        n = len(data['base'])
        assert all(len(v) == n for v in data.values())
        for rows in zip(*(data[arm] for arm in ARMS)):
            refs = dict(zip(ARMS, rows))
            assert len({p['sample_id'] for p in rows}) == 1
            assert all(p['frames'] == rows[0]['frames'] for p in rows)
            for arm, p in refs.items():
                ei, ci = decode(np.asarray(p['entry_logits'], float), np.asarray(p['collision_logits'], float))
                assert (p['frames'][ei], p['frames'][ci]) == (p['entry_frame'], p['collision_frame']), (arm,p['sample_id'])
                assert int(p['side_prob_right'] >= .5) == p['entry_side']
                assert int(p['evasion_prob'] >= .5) == p['evasion_space']
                scores[arm].append({k: p[k] for k in ('sample_id','source_id','entry_frame','collision_frame','entry_side','evasion_space',
                                                         'entry_gt','collision_gt','entry_side_gt','evasion_gt','num_available_frames')})
            for a, b in PAIRS:
                x, y = refs[a], refs[b]
                entry = .5*(probs(x['entry_logits'])+probs(y['entry_logits']))
                collision = .5*(probs(x['collision_logits'])+probs(y['collision_logits']))
                ei, ci = decode(np.log(entry), np.log(collision))
                q = dict(scores[a][-1])
                q['entry_frame'] = x['frames'][ei]
                q['collision_frame'] = x['frames'][ci]
                q['entry_side'] = int((x['side_prob_right']+y['side_prob_right'])/2 >= .5)
                q['evasion_space'] = int((x['evasion_prob']+y['evasion_prob'])/2 >= .5)
                scores[a+'+'+b].append(q)
    result = {'protocol': 'Equal two-member probability average at native rate, grouped five-fold OOF seed0. Saved native logits only; no new inference or hyperparameter tuning.',
              'limitations': 'No half/third-rate logits, no seed ensemble, and no held-out leaderboard evidence. Exploratory recipe screen only.',
              'results': {name: C.breakdown(preds) for name, preds in scores.items()}}
    C.dump(OUT/'native_aux_ensemble_results.json', result)
    for name, metrics in result['results'].items():
        print(name, metrics['overall']['score'], metrics.get('source:NEXAR',{}).get('score'))


if __name__ == '__main__':
    main()
