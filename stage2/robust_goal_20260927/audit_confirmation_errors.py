"""Attribute matched-fold differences without fitting rules or loading models."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main():
    aggregate = json.loads((OUT / 'confirmation_aggregate.json').read_text())
    folds = aggregate['matched_folds']
    fps = {r['sample_id']: float(r['native_fps']) for r in
           map(json.loads, Path('/workspace/data/stage2/manifests/all.jsonl').read_text().splitlines())}
    result = {'folds': folds, 'seed': 0,
              'scope': 'Descriptive matched partial-CV errors; FPS used only for offline scoring.',
              'rates': {}}
    for rate, scores in aggregate['rates'].items():
        predictions = {}
        for arm in ('base', 'aux_sharp', 'aux_broad'):
            predictions[arm] = {}
            for fold in folds:
                path = OUT / 'training' / arm / f'cv/fold{fold}_seed0/rate_eval.json'
                data = json.loads(path.read_text())
                assert data['native_parity_mismatches'] == []
                for p in data[rate]['predictions']:
                    assert p['sample_id'] not in predictions[arm]
                    predictions[arm][p['sample_id']] = p
        assert all(set(p) == set(predictions['base']) for p in predictions.values())
        sources = [k for k in scores['breakdown']['base'] if k.startswith('source:') and k != 'source:non-NEXAR']
        rate_result = {}
        for source in sources:
            base_metrics = scores['breakdown']['base'][source]
            source_result = {}
            for arm in ('aux_sharp', 'aux_broad'):
                metrics = scores['breakdown'][arm][source]
                contributions = {key: weight * (metrics[key] - base_metrics[key]) for key, weight in
                                 [('entry_acc', .35), ('collision_acc', .35), ('side_f1', .15), ('evasion_f1', .15)]}
                assert abs(sum(contributions.values()) - (metrics['score'] - base_metrics['score'])) < 1e-10
                transitions = {}
                for event in ('entry', 'collision'):
                    buckets = {'both_hit': [], 'recovered': [], 'regressed': [], 'both_miss': []}
                    for sid, base in predictions['base'].items():
                        if base['source_id'].split(':')[0] != source.split(':')[1]:
                            continue
                        candidate = predictions[arm][sid]
                        assert candidate[event + '_gt'] == base[event + '_gt']
                        errors = [(p[event + '_frame'] - p[event + '_gt']) / fps[sid] for p in (base, candidate)]
                        hits = [abs(e) <= .300001 for e in errors]
                        bucket = ('both_hit' if hits[1] else 'regressed') if hits[0] else ('recovered' if hits[1] else 'both_miss')
                        buckets[bucket].append({'sample_id': sid, 'base_signed_error_seconds': errors[0],
                                                'candidate_signed_error_seconds': errors[1]})
                    transitions[event] = {'counts': {k: len(v) for k, v in buckets.items()}, 'clips': buckets}
                source_result[arm] = {'n': metrics['n'], 'weighted_score_contributions': contributions,
                                      'transitions': transitions}
            rate_result[source] = source_result
        result['rates'][rate] = rate_result
    (OUT / 'confirmation_error_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    nexar = result['rates']['k3']['source:NEXAR']['aux_broad']
    print(json.dumps({'folds': folds, 'third_rate_nexar': {
        'contributions': nexar['weighted_score_contributions'],
        'transitions': {e: d['counts'] for e, d in nexar['transitions'].items()}}}, indent=2))


if __name__ == '__main__':
    main()
