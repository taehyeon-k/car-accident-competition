"""Continue the best completed time-boxed screen through epoch 100.

The original run folders remain untouched. The copied optimizer state resumes
training under a new 100-epoch cosine schedule.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path('/workspace/runs/stage3_tcp_transfer')
CANDIDATES = ('C0_control', 'T1_frozen_tcp', 'T2_tcp_projection')
NAME = 'X1_best_to_100ep'
DEADLINE = datetime(2026, 9, 28, 23, 0, tzinfo=timezone.utc)


def main() -> None:
    target = ROOT / NAME
    # The original deadline prevented a new extension from starting after the
    # requested cutoff. An already-created extension may be resumed later.
    if datetime.now(timezone.utc) >= DEADLINE and not (target / 'last.pt').is_file():
        print('Deadline reached; extension will not start.', flush=True)
        return
    candidates = []
    for name in CANDIDATES:
        folder = ROOT / name
        metrics = folder / 'metrics.json'
        if not metrics.is_file():
            raise FileNotFoundError(f'Screen is incomplete: {metrics}')
        result = json.loads(metrics.read_text())
        candidates.append((float(result['val']['competition_score']), name))
    score, source_name = max(candidates)
    source = ROOT / source_name
    target.mkdir(exist_ok=True)
    marker = target / 'source.json'
    if not (target / 'last.pt').is_file():
        if any(target.iterdir()):
            raise RuntimeError(f'Incomplete extension setup in {target}; inspect before retrying')
        cfg = yaml.safe_load((source / 'config.yaml').read_text())
        cfg['output_dir'] = str(target)
        cfg['optimization']['epochs'] = 100
        cfg['data']['num_workers'] = 0 if source_name == 'C0_control' else 1
        if cfg['data']['num_workers']:
            cfg['data']['prefetch_factor'] = 1
        # The screen already established the short-schedule curve. Spend the
        # deadline-limited continuation on updates, with periodic and final
        # authoritative validation rather than a full pass after every epoch.
        cfg.setdefault('logging', {})['val_every'] = 5
        cfg.setdefault('evaluation', {})['skip_train_full'] = True
        for filename in ('last.pt', 'best.pt', 'history.jsonl', 'environment.json'):
            shutil.copy2(source / filename, target / filename)
        (target / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        source_epochs = len((source / 'history.jsonl').read_text().splitlines())
        marker.write_text(json.dumps({'source': source_name, 'source_score': score,
                                      'source_epochs': source_epochs,
                                      'target_epochs': 100}, indent=2) + '\n')
    else:
        source_name = json.loads(marker.read_text())['source']
    if (target / 'metrics.json').is_file():
        print('Extension already complete.', flush=True)
        return
    module = ('stage3.experiments.run' if source_name == 'C0_control'
              else 'stage3.experiments.tcp_transfer.run_tcp')
    print(f'Continuing {source_name} in {NAME} through epoch 100', flush=True)
    os.execv(sys.executable, [sys.executable, '-m', module, '--name', NAME,
                              '--root', str(ROOT), '--resume'])


if __name__ == '__main__':
    main()
