"""Insert measured C0/T1/T2 results into the Stage 3 transfer report."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/workspace/runs/stage3_tcp_transfer')
REPORT = Path('/workspace/car-accident/reports/stage3_tcp_openpilot_transfer.md')
RUNS = ('C0_control', 'T1_frozen_tcp', 'T2_tcp_projection')
START = '<!-- MEASURED_RESULTS_START -->'
END = '<!-- MEASURED_RESULTS_END -->'


def score(value: float) -> str:
    return f'{value:.4f}'


def main() -> None:
    records = {name: json.loads((ROOT / name / 'metrics.json').read_text()) for name in RUNS}
    lines = [
        '## Measured time-boxed screening results',
        '',
        'All rows use the same 466 BATON validation clips, seed 42, and ±0.4 m/s² / 7° decoder.',
        '',
        '| ID | Epochs | Competition | Acceleration Macro-F1 | Steering Macro-F1 | Accel MAE (m/s²) | Steering MAE (°) | Speed MAE (m/s) | Best epoch | Train time (h) | Peak train VRAM (GiB) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name, record in records.items():
        val = record['val']
        lines.append('| ' + ' | '.join((
            name, str(len((ROOT / name / 'history.jsonl').read_text().splitlines())),
            score(val['competition_score']), score(val['acceleration_macro_f1']),
            score(val['steering_macro_f1']), score(val['acceleration_mae']),
            score(val['steering_angle_mae']), score(val['speed_mae']),
            str(record['best_epoch']), f"{record['train_seconds'] / 3600:.2f}",
            f"{record['train_peak_vram_mb'] / 1024:.2f}",
        )) + ' |')
    lines += ['', '| ID | ACCELERATING F1 | DECELERATING F1 | CONSTANT F1 | STOPPED F1 | LEFT F1 | RIGHT F1 | STRAIGHT F1 | Trainable params | Best checkpoint size (MiB) | Cached validation ms/frame |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, record in records.items():
        val = record['val']
        checkpoint_size = (ROOT / name / 'best.pt').stat().st_size / 2**20
        lines.append('| ' + ' | '.join((
            name, score(val['acceleration_f1_accelerating']), score(val['acceleration_f1_decelerating']),
            score(val['acceleration_f1_constant']), score(val['acceleration_f1_stopped']),
            score(val['steering_f1_left']), score(val['steering_f1_right']),
            score(val['steering_f1_straight']), str(record['trainable_params']),
            f'{checkpoint_size:.1f}', f"{record['eval_ms_per_frame']:.3f}",
        )) + ' |')
    lines += ['', 'Cached validation ms/frame excludes SEA-RAFT preprocessing for C0 and TCP RGB encoding for T1/T2. It is not an end-to-end inference latency.', '']
    for name, record in records.items():
        lines += [f'### {name} confusion matrices', '',
                  'Acceleration, rows = truth, columns = prediction:', '',
                  '```text',
                  '  ' + ' '.join(record['confusion_acceleration']['labels']),
                  *(str(row) for row in record['confusion_acceleration']['matrix']),
                  '```', '',
                  'Steering, rows = truth, columns = prediction; ground-truth STOPPED excluded:', '',
                  '```text',
                  '  ' + ' '.join(record['confusion_steering']['labels']),
                  *(str(row) for row in record['confusion_steering']['matrix']),
                  '```', '',
                  f"Acceleration-vs-speed-derivative validation loss: {record['val'].get('loss_accel_speed_consistency', float('nan')):.4f}.", '']
    base = records['C0_control']['val']['competition_score']
    frozen = records['T1_frozen_tcp']['val']['competition_score']
    tuned = records['T2_tcp_projection']['val']['competition_score']
    lines += ['## Screening interpretation', '',
              f'T1 minus C0: {frozen - base:+.4f} competition score. T2 minus T1: {tuned - frozen:+.4f}.',
              'These are single-seed BATON-validation margins, and the earlier archived V3 leaderboard result shows that BATON validation can overestimate DACON transfer. T2 changes the TCP projection and warm-starts T1 temporal/head weights, so T2 vs T1 is an additional-training comparison, not a from-scratch equal-budget comparison.',
              'Openpilot D1/D2 remain unrun because the uncalibrated teacher probe failed on moving BATON footage. A robust two-view teacher input and output-quality audit are prerequisites. No conclusion about the benefit of valid openpilot distillation can be drawn.',
              'The next experiment should confirm any promising TCP margin on a route-group-disjoint split and measure full RGB inference latency; only then consider a 100-epoch schedule.', '']
    replacement = START + '\n' + '\n'.join(lines) + '\n' + END
    report = REPORT.read_text()
    before, sep, after = report.partition(START)
    if not sep or END not in after:
        raise ValueError('Report measured-results markers are missing')
    REPORT.write_text(before + replacement + after.split(END, 1)[1])
    print(f'Updated {REPORT}')


if __name__ == '__main__':
    main()
