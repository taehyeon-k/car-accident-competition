"""Measure isolated inference costs after ablation training has completed."""
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from stage2.geometry_pretrain.ablate_normalized_probe import OUT, ROOT, Probe, read


def main():
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('stage2_benchmark_runtime',ROOT/'submission/model/stage2/runtime.py')
    runtime=importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    from dinov3.hub.backbones import dinov3_vits16
    from sampling import normalized_indices
    backbone=dinov3_vits16(pretrained=False)
    backbone.load_state_dict(torch.load(ROOT/'submission/model/stage2/backbone.pth',map_location='cpu',weights_only=True))
    backbone.cuda().eval()
    row=max(read('val'),key=lambda r:r['num_frames'])
    pairs=runtime.indexed_frames(Path(row['frames_dir']))
    results={}
    with torch.inference_mode():
        for count in (64,128,256):
            positions=normalized_indices([i for i,_ in pairs],count)
            unique,inverse=np.unique(positions,return_inverse=True)
            selected=[pairs[i] for i in unique]
            runtime.encode(backbone,selected[:16],torch.device('cuda'),16,2)
            measurements=[]; torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                torch.cuda.synchronize(); start=time.perf_counter()
                features=runtime.encode(backbone,selected,torch.device('cuda'),16,2)
                features=features[:,torch.as_tensor(inverse,device='cuda')]
                torch.cuda.synchronize(); measurements.append(time.perf_counter()-start)
            heads={}
            for temporal in (True,False):
                model=Probe(temporal).cuda().eval(); valid=torch.ones((1,count),device='cuda',dtype=torch.bool)
                for _ in range(5): model(features,valid)
                torch.cuda.synchronize(); start=time.perf_counter()
                for _ in range(100): model(features,valid)
                torch.cuda.synchronize()
                heads['tcn' if temporal else 'no_tcn']=1000*(time.perf_counter()-start)/100
                del model
            results[str(count)]={'unique_frames':len(unique),'encode_seconds_median':float(np.median(measurements)),
                                 'head_milliseconds':heads,'peak_gpu_MiB':torch.cuda.max_memory_allocated()/2**20}
    result={'video_id':row['sample_id'],'input_frames':len(pairs),'batch_size':16,'cpu_threads':2,
            'includes':'JPEG loading, letterbox, DINOv3-S, pooling; excludes initial model loading',
            'results':results}
    (OUT/'latency.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__=='__main__':main()
