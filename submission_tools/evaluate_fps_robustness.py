"""Held-out cached-feature stress test at different temporal sampling densities."""
import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission_tools'))
from train_fps_stage2 import Features, read, evaluate, TemporalProbe


def main():
    torch.set_num_threads(2)
    rows=read('/workspace/data/stage2/manifests/val.jsonl')
    state=torch.load(ROOT/'submission_tools/fps_stage2/selection.pt',map_location='cpu',weights_only=False)
    model=TemporalProbe().cuda().eval(); model.load_state_dict(state['model'])
    results={}
    for stride in (1,2,3,4):
        data=Features(rows)
        data.arrays=[(a[::stride],idx[::stride]) for a,idx in data.arrays]
        results[f'cached_stride_{stride}']=evaluate(model,DataLoader(data,batch_size=8))
    out=ROOT/'submission_tools/fps_robustness_results.json'; out.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
