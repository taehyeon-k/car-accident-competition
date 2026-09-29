"""Evaluate a frozen ViT-B head as a fixed equal-weight addition to v8.

Runs CPU-only on the established duplicate-clean 284-clip CV cohort. The
same-backbone E4_sb addition is the capacity/recipe control for E4_sbB.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.spotting_experiments.objective import constrained_anchors
from stage2.generalization import robust_eval as R
from stage2.generalization import robustness_profile as P
from stage2.generalization.robustness_analyze import excluded
from .resource_guard import resource_guard

OUT = Path(__file__).resolve().parent
FAMILIES = {
    'vitb': ('E4_sa', 'E2_sa', 'XN4_sa', 'E4_sb', 'E4_sbB'),
    'consistency': ('E4_sa', 'E2_sa', 'XN4_sa', 'E4_sb', 'E4_sbu'),
    'objappearance': ('E4_sa', 'E2_sa', 'XN4_sa', 'E4_sb', 'E4_sbOA'),
    'v13blend': ('E4_sa', 'E2_sa', 'XN4_sa', 'E4_sbuC', 'E2_sbuC', 'XN4_sbuC'),
}


def add_object_appearance(it, stride):
    from stage2.aux_signal_experiments.train import segments
    sid = it['sid']
    if stride == 1:
        kept = np.arange(it['n_native'])
    else:
        kept = np.load(R.STRIDE / f'k{stride}' / f'{sid}.npz')['kept']
    pos = np.searchsorted(kept, it['abs'])
    assert np.array_equal(kept[pos], it['abs'])
    obj = np.load(C.REPO / f'stage2/objtrack/cache_objapp/k{stride}/{sid}.npy')
    assert len(obj) == len(kept) and obj.shape[1] == 120
    motion = np.concatenate([it['both'].numpy(), segments(obj, pos)], axis=1)
    it['both_objapp'] = torch.from_numpy(np.ascontiguousarray(motion))
    return it


@torch.inference_mode()
def predict_all(members, it, recipes, device):
    v = torch.ones(1, len(it['frames']), dtype=torch.bool, device=device)
    x = it['x'][None].to(device)
    xs = {}
    outputs = {}
    for family, models in members.items():
        outputs[family] = []
        for model, kind, features_dir in models:
            if features_dir:
                if features_dir not in xs:
                    xs[features_dir] = torch.from_numpy(np.ascontiguousarray(
                        np.load(Path(features_dir) / f"{it['sid']}.npy", mmap_mode='r')[it['abs']]))[None].to(device)
                xm = xs[features_dir]
            else:
                xm = x
            outputs[family].append(model(xm, v, motion=it[kind][None].to(device)) if kind else model(xm, v))

    row = it['row']
    frames = it['frames']
    preds = {}
    for name, families in recipes.items():
        oo = [o for family in families for o in outputs[family]]
        pe = torch.stack([o['entry_logits'].float().softmax(-1) for o in oo]).mean(0)
        pc = torch.stack([o['collision_logits'].float().softmax(-1) for o in oo]).mean(0)
        side = float(torch.stack([o['side_logits'].float().softmax(-1)[0, 1] for o in oo]).mean())
        evasion = float(torch.stack([o['evasion_logits'].float().sigmoid().reshape(-1)[0] for o in oo]).mean())
        ei, ci = constrained_anchors(pe.log(), pc.log())
        preds[name] = {
            'sample_id': it['sid'], 'source_id': row['source_id'],
            'entry_frame': int(frames[int(ei[0])]),
            'collision_frame': int(frames[int(ci[0])]),
            'entry_side': int(side >= .5), 'evasion_space': int(evasion >= .5),
            'entry_gt': int(row['entry_frame']), 'collision_gt': int(row['collision_frame']),
            'entry_side_gt': int(row['entry_side'] == 'RIGHT'),
            'evasion_gt': int(row['evasion_space']),
            'num_available_frames': it['n_native'],
        }
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strides', type=int, nargs='+', default=[3], choices=[1, 2, 3])
    parser.add_argument('--study', choices=FAMILIES, default='vitb')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; GPU screen will not fall back to CPU')
        free, _ = torch.cuda.mem_get_info(0)
        if free < 16 * 1024**3:
            raise RuntimeError(f'GPU admission refused: {free / 1024**3:.1f} GiB free')
        torch.cuda.set_per_process_memory_fraction(0.15, 0)
        torch.backends.cuda.matmul.allow_tf32 = False
    exclusions = excluded()
    families = FAMILIES[args.study]
    candidate = {'vitb': 'v8_plus_vitb', 'consistency': 'v8_plus_consistency',
                 'objappearance': 'v8_plus_objappearance', 'v13blend': 'v8_v13_equal'}[args.study]
    if args.study == 'v13blend':
        recipes = {'v8_plain': families[:3], 'v13_plain': families[3:],
                   candidate: families}
    else:
        recipes = {'v8_plain': families[:3], 'v8_plus_vits': families[:4],
                   candidate: families[:3] + families[4:]}
    dest = OUT / f'clean_{args.study}_blend_results.json'
    result = json.loads(dest.read_text()) if dest.exists() else {
        'protocol': f'284 duplicate-clean OOF clips; seeds0-2; fixed equal model weights; {args.device} plain decoder; no fitted recipe weights',
        'recipes': {k: list(v) for k, v in recipes.items()},
        'conditions': {},
    }
    for stride in args.strides:
        key = f'k{stride}'
        if key in result['conditions']:
            print(key, 'already evaluated', flush=True)
            continue
        outputs = {r: [] for r in recipes}
        for fold in range(5):
            resource_guard(14.0)
            if args.device == 'cuda':
                free, _ = torch.cuda.mem_get_info(0)
                if free < 12 * 1024**3:
                    raise RuntimeError(f'GPU fold admission refused: {free / 1024**3:.1f} GiB free')
            members = {family: [P.load_member(family, fold, seed, device)
                                for seed in (0, 1, 2)] for family in families}
            if args.study == 'objappearance':
                members['E4_sbOA'] = [(model, 'both_objapp', features_dir)
                                      for model, _, features_dir in members['E4_sbOA']]
            for row in C.rows(str(R.FOLDS / f'fold{fold}_val.jsonl')):
                if row['sample_id'] in exclusions:
                    continue
                resource_guard(14.0)
                item = R.item(row, stride)
                if args.study == 'objappearance':
                    item = add_object_appearance(item, stride)
                pred = predict_all(members, item, recipes, device)
                for recipe in recipes:
                    outputs[recipe].append(pred[recipe])
            print(key, 'fold', fold, 'complete', flush=True)
            del members
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        ids = [p['sample_id'] for p in outputs['v8_plain']]
        assert len(ids) == 284 and len(ids) == len(set(ids))
        assert all([p['sample_id'] for p in outputs[r]] == ids for r in recipes)
        result['conditions'][key] = {
            recipe: {'metrics': C.breakdown(preds), 'predictions': preds}
            for recipe, preds in outputs.items()
        }
        temp = dest.with_suffix('.tmp')
        temp.write_text(json.dumps(result) + '\n')
        temp.replace(dest)
        print(key, {r: round(result['conditions'][key][r]['metrics']['overall']['score'], 6)
                    for r in recipes}, flush=True)


if __name__ == '__main__':
    main()
