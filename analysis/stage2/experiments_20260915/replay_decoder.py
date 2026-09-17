"""Evaluate prespecified decoders on saved baseline logits; no model execution."""
import json
import math
from pathlib import Path

import torch
from stage2.utils.joint_losses import constrained_decode

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
torch.set_num_threads(2)


def macro_f1(rows, name):
    matrix = torch.zeros(2, 2)
    for row in rows:
        matrix[row[name + '_true'], row[name + '_pred']] += 1
    return float((2 * matrix.diag() / (matrix.sum(0) + matrix.sum(1)).clamp_min(1)).mean())


def metrics(rows):
    e = sum(abs(r['entry_error_s']) <= .300001 for r in rows) / len(rows)
    c = sum(abs(r['collision_error_s']) <= .300001 for r in rows) / len(rows)
    s, v = macro_f1(rows, 'side'), macro_f1(rows, 'evasion')
    return dict(n=len(rows), entry=e, collision=c, side=s, evasion=v,
                score=.35 * e + .35 * c + .15 * s + .15 * v)


def main():
    samples = torch.load(REPO / 'runs/joint_online_lora/val_logits_best.pt',
                         map_location='cpu', weights_only=True)
    original = {r['sample_id']: r for r in json.loads(
        (REPO / 'runs/joint_online_lora/val_errors_best.json').read_text())}
    result = {}
    # Include float32 as a numerical check; keep the baseline's bf16 decision rule
    # for the span comparisons. Mass decoding is a separate proposed experiment.
    for mode in ['baseline_bf16', 'baseline_fp32', 'span200_bf16',
                 'span5sec_bf16', 'span6sec_bf16', 'span200_fp32', 'span5sec_fp32',
                 'tolerance_mass_unlimited', 'tolerance_mass_200frames',
                 'tolerance_mass_5sec', 'fixed_radius3_mass_200frames']:
        rows = []
        for sample in samples:
            fps = sample['fps']
            span = (200 if mode in ['span200_bf16', 'span200_fp32',
                                   'tolerance_mass_200frames', 'fixed_radius3_mass_200frames'] else
                    math.floor(5 * fps + 1e-6) if mode in ['span5sec_bf16', 'span5sec_fp32', 'tolerance_mass_5sec'] else
                    math.floor(6 * fps + 1e-6) if mode == 'span6sec_bf16' else None)
            entry, collision = sample['entry_logits'][None], sample['collision_logits'][None]
            if 'bf16' in mode:
                entry, collision = entry.bfloat16(), collision.bfloat16()
            if 'mass' in mode:
                radius = 3 if mode == 'fixed_radius3_mass_200frames' else math.floor(.3 * fps + 1e-6)
                kernel = torch.ones(1, 1, 2 * radius + 1)
                entry = .35 * torch.nn.functional.conv1d(
                    entry.float().softmax(-1)[:, None], kernel, padding=radius)[:, 0]
                collision = .35 * torch.nn.functional.conv1d(
                    collision.float().softmax(-1)[:, None], kernel, padding=radius)[:, 0]
            e, c = constrained_decode(entry, collision, span)
            e, c = int(e), int(c)
            row = {k: sample[k] for k in ['sample_id', 'source', 'fps', 'entry_true',
                   'collision_true', 'side_true', 'side_pred', 'evasion_true', 'evasion_pred']}
            row.update(entry_pred=e, collision_pred=c,
                       entry_error_s=(e - sample['entry_true']) / fps,
                       collision_error_s=(c - sample['collision_true']) / fps)
            rows.append(row)
        baseline = original
        changed = [r for r in rows if (r['entry_pred'], r['collision_pred']) !=
                   (baseline[r['sample_id']]['entry_pred'], baseline[r['sample_id']]['collision_pred'])]
        transition = {}
        for event in ['entry', 'collision']:
            transition[event] = {
                'corrected': sum(abs(r[event + '_error_s']) <= .300001 and
                                 abs(baseline[r['sample_id']][event + '_err_s']) > .300001 for r in rows),
                'broken': sum(abs(r[event + '_error_s']) > .300001 and
                              abs(baseline[r['sample_id']][event + '_err_s']) <= .300001 for r in rows),
            }
        result[mode] = {'metrics': metrics(rows),
                        'by_source': {s: metrics([r for r in rows if r['source'] == s])
                                      for s in ['AIHUB', 'CCD', 'NEXAR']},
                        'changed_pairs': len(changed), 'transitions': transition,
                        'changed_rows': changed, 'rows': rows}
        print(mode, result[mode]['metrics'], 'changed', len(changed), transition)
    fp32_reference = {r['sample_id']: r for r in result['baseline_fp32']['rows']}
    for values in result.values():
        values['transitions_vs_fp32'] = {}
        for event in ['entry', 'collision']:
            values['transitions_vs_fp32'][event] = {
                'corrected': sum(abs(r[event + '_error_s']) <= .300001 and
                                 abs(fp32_reference[r['sample_id']][event + '_error_s']) > .300001
                                 for r in values['rows']),
                'broken': sum(abs(r[event + '_error_s']) > .300001 and
                              abs(fp32_reference[r['sample_id']][event + '_error_s']) <= .300001
                              for r in values['rows']),
            }
    (OUT / 'decoder_replay.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
