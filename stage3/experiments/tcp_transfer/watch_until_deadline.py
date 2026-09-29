"""Keep the Stage 3 experiment queue alive until 08:00 KST, 29 Sep 2026.

Supervisor runs this independently of the interactive shell. It only manages the
three Stage 3 TCP experiment services and never touches unrelated training jobs.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/workspace/runs/stage3_tcp_transfer')
DEADLINE = datetime(2026, 9, 28, 23, 0, tzinfo=timezone.utc)
RUNS = (
    ('stage3_tcp_control', 'C0_control'),
    ('stage3_tcp_t1', 'T1_frozen_tcp'),
    ('stage3_tcp_t2', 'T2_tcp_projection'),
    ('stage3_tcp_ensemble', 'E1_weighted_vote'),
    ('stage3_tcp_extension', 'X1_best_to_100ep'),
)


def supervisor_status(name: str) -> str:
    result = subprocess.run(['supervisorctl', 'status', name], capture_output=True, text=True)
    return result.stdout.split()[1] if result.returncode == 0 and len(result.stdout.split()) > 1 else 'UNKNOWN'


def snapshot() -> dict:
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'deadline_utc': DEADLINE.isoformat(), 'target_score': 0.9, 'runs': {}}
    for service, run in RUNS:
        folder = ROOT / run
        row = {'service': service, 'state': supervisor_status(service)}
        metrics = folder / 'metrics.json'
        history = folder / 'history.jsonl'
        if metrics.is_file():
            try:
                final = json.loads(metrics.read_text())
                row.update(completed=True, score=final['val']['competition_score'])
                if final.get('best_epoch') is not None:
                    row['best_epoch'] = final['best_epoch']
            except (OSError, ValueError, KeyError):
                pass
        if history.is_file():
            try:
                epochs = [json.loads(line) for line in history.read_text().splitlines() if line]
                if epochs:
                    best = max(epochs, key=lambda x: x.get('val/competition_score', 0))
                    row.update(epochs_finished=len(epochs),
                               best_interim_score=best.get('val/competition_score', 0),
                               best_interim_epoch=best.get('epoch'))
            except (OSError, ValueError):
                pass
        result['runs'][run] = row
    result['best_verified_score'] = max((row.get('score', row.get('best_interim_score', 0))
                                         for row in result['runs'].values()), default=0)
    result['target_reached'] = result['best_verified_score'] >= 0.9
    return result


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    restarted: dict[str, float] = {}
    while True:
        now = datetime.now(timezone.utc)
        status = snapshot()
        temp = ROOT / 'goal_status.json.tmp'
        temp.write_text(json.dumps(status, indent=2) + '\n')
        temp.replace(ROOT / 'goal_status.json')
        if now >= DEADLINE:
            for service, _ in RUNS:
                if supervisor_status(service) == 'RUNNING':
                    subprocess.run(['supervisorctl', 'stop', service], check=False)
            status = snapshot()
            status['deadline_reached'] = True
            (ROOT / 'goal_status.json').write_text(json.dumps(status, indent=2) + '\n')
            print(json.dumps({'event': 'deadline', 'best_verified_score': status['best_verified_score']}), flush=True)
            return
        for service, run in RUNS:
            row = status['runs'][run]
            if row.get('completed') or row['state'] in {'RUNNING', 'STARTING'}:
                continue
            if time.monotonic() - restarted.get(service, -1e9) < 180:
                continue
            restarted[service] = time.monotonic()
            action = subprocess.run(['supervisorctl', 'start', service], capture_output=True, text=True)
            print(json.dumps({'event': 'restart', 'service': service,
                              'returncode': action.returncode, 'message': action.stdout.strip() or action.stderr.strip()}), flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
