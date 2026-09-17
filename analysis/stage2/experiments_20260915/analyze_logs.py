"""Extract local W&B records, including complete records from an active run."""
import collections
import csv
import datetime
import json
import struct
import zlib
from pathlib import Path

from wandb.proto.wandb_internal_pb2 import Record

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
RUNS = {
    'baseline': ('joint_online_lora', 'run-20260914_164354-kqr8dczz', 'kqr8dczz'),
    'span_lr': ('joint_online_lora_span_lr', 'run-20260915_010948-irfg2z4m', 'irfg2z4m'),
    'span_lr_dr': ('joint_online_lora_span_lr_dr', 'run-20260915_042253-15ov22sn', '15ov22sn'),
}


def records(raw):
    assert struct.unpack('<4sHB', raw[:7]) == (b':W&B', 0xBEE1, 0)
    pos, fragment = 7, b''
    while pos < len(raw):
        remaining = 32768 - pos % 32768
        if remaining < 7:
            padding = raw[pos:pos + remaining]
            if len(padding) < remaining:
                return
            assert padding == bytes(remaining)
            pos += remaining
            continue
        if pos + 7 > len(raw):
            return
        checksum, size, kind = struct.unpack_from('<IHB', raw, pos)
        pos += 7
        if pos + size > len(raw):
            return  # Do not parse an unfinished record from a running writer.
        payload = raw[pos:pos + size]
        pos += size
        assert kind in (1, 2, 3, 4), (pos, kind)
        assert zlib.crc32(bytes([kind]) + payload) & 0xffffffff == checksum
        if kind in (1, 2):
            assert not fragment
            fragment = payload
        else:
            fragment += payload
        if kind in (1, 4):
            record = Record()
            record.ParseFromString(fragment)
            fragment = b''
            yield record


def items(values):
    return {item.key or '/'.join(item.nested_key): json.loads(item.value_json)
            for item in values}


def main():
    summary = {}
    for name, (directory, run, run_id) in RUNS.items():
        path = REPO / 'runs' / directory / 'wandb' / run / f'run-{run_id}.wandb'
        raw = path.read_bytes()
        counts = collections.Counter()
        history, stats, config = [], [], {}
        for record in records(raw):
            kind = record.WhichOneof('record_type')
            counts[kind] += 1
            if kind == 'history':
                history.append(items(record.history.item))
            elif kind == 'stats':
                stats.append(items(record.stats.item))
            elif kind == 'config':
                config.update(items(record.config.update))
        epochs = [r for r in history if 'val/competition_score' in r]
        (OUT / f'{name}_history.json').write_text(json.dumps(history, indent=2) + '\n')
        keys = sorted(set().union(*(r.keys() for r in epochs)))
        with (OUT / f'{name}_epochs.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(epochs)
        best = max(epochs, key=lambda r: r['val/competition_score'])
        summary[name] = {
            'run_id': run_id, 'snapshot_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'log_bytes': len(raw), 'record_counts': dict(counts), 'finished': bool(counts['exit']),
            'epochs': len(epochs), 'config_record': config, 'best': best, 'last': epochs[-1],
            'gpu_keys': sorted({k for r in stats for k in r if 'gpu' in k.lower()}),
        }
        print(name, 'finished=', bool(counts['exit']), 'epochs=', len(epochs))
        for row in epochs:
            print('epoch', row.get('train_config/epoch'),
                  'train', round(row['train/competition_score'], 4),
                  'val', round(row['val/competition_score'], 4),
                  'entry', row['val/acc_entry_0.3s'],
                  'collision', row['val/acc_collision_0.3s'],
                  'side', round(row['val/f1_entry_side_macro'], 4),
                  'evasion', round(row['val/f1_evasion_space_macro'], 4))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
