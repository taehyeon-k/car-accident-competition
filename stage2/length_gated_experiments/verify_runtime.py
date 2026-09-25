"""Check the feature-level hybrid against the saved 70-video routing result."""
import json
from pathlib import Path
import torch
from .runtime import Hybrid
from stage2.long_context_v2_experiments import common as C

ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.12)
    model=Hybrid()
    saved={p['sample_id']:p for p in json.loads((ROOT/'predictions_all_seeds.json').read_text())}
    assert len(saved)==70
    for row in C.rows('val'):
        item=C.make_item(row,'adaptive')
        actual=model.predict(item['x'],item['frame_numbers'].numpy(),item['num_available_frames'])
        expected=saved[row['sample_id']]
        for key,value in actual.items():assert value==expected[key],(row['sample_id'],key,value,expected[key])
    print('Feature-level hybrid prediction parity: 70/70')

if __name__=='__main__':main()
