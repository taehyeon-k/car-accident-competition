"""Read local W&B records without starting a run or contacting W&B."""
import collections
import json
import struct
import zlib
from pathlib import Path
from wandb.proto.wandb_internal_pb2 import Record

out = Path(__file__).resolve().parent
path = out.parents[1] / 'runs/joint_online_lora/wandb/run-20260914_164354-kqr8dczz/run-kqr8dczz.wandb'
raw = path.read_bytes()
assert struct.unpack('<4sHB', raw[:7]) == (b':W&B', 0xBEE1, 0)
pos = 7
fragment = b''
history = []
counts = collections.Counter()
while pos < len(raw):
    remaining = 32768 - pos % 32768
    if remaining < 7:
        assert raw[pos:pos+remaining] == bytes(remaining)
        pos += remaining
        continue
    checksum, size, kind = struct.unpack_from('<IHB', raw, pos)
    pos += 7
    payload = raw[pos:pos+size]
    pos += size
    assert zlib.crc32(bytes([kind]) + payload) & 0xffffffff == checksum
    if kind in (1, 2):
        assert not fragment
        fragment = payload
    else:
        assert kind in (3, 4)
        fragment += payload
    if kind not in (1, 4):
        continue
    record = Record()
    record.ParseFromString(fragment)
    fragment = b''
    record_type = record.WhichOneof('record_type')
    counts[record_type] += 1
    if record_type == 'history':
        history.append({item.key or '/'.join(item.nested_key): json.loads(item.value_json)
                        for item in record.history.item})
assert not fragment
(out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
print(dict(counts))
for row in history:
    if 'val/competition_score' in row:
        print({k: round(v,4) if isinstance(v,float) else v for k,v in row.items()
               if k in ['_step','train_config/epoch','val/competition_score','train/competition_score','val/acc_entry_0.3s','val/acc_collision_0.3s','val/f1_entry_side_macro','val/f1_evasion_space_macro','val/loss']})
