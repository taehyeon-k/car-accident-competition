"""Sampling invariance and real CUDA end-to-end checks for the normalized-clip model."""
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'submission_tools/fps_stage2'
sys.path.insert(0,str(PACKAGE))
from sampling import normalized_indices


def main():
    torch.set_num_threads(2)
    for length in (1,2,17,128,500,10000):
        values=np.arange(length)
        positions=normalized_indices(values)
        assert len(positions)==128 and positions.min()>=0 and positions.max()<length
        assert (np.diff(positions)>=0).all()
        assert np.array_equal(positions,normalized_indices(values*7+100))
        assert positions[0]==0 and positions[-1]==length-1
    spec=importlib.util.spec_from_file_location('fps_stage2_runtime',PACKAGE/'runtime.py')
    runtime=importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    record=json.loads(Path('/workspace/data/stage2/manifests/val.jsonl').read_text().splitlines()[0])
    def blocked(*a,**kw): raise AssertionError('Network access attempted')
    socket.socket.connect=blocked; socket.create_connection=blocked
    with tempfile.TemporaryDirectory(prefix='fps-stage2-') as tmp:
        root=Path(tmp); images=root/'stage2/images'
        for name in ('original','reindexed','single'): (images/name).mkdir(parents=True)
        for p in sorted(Path(record['frames_dir']).glob('*.jpg')):
            i=int(p.stem)
            shutil.copy2(p,images/'original'/f'frame_{i:06d}.jpg')
            shutil.copy2(p,images/'reindexed'/f'frame_{i*7+100:06d}.jpg')
        shutil.copy2(next(Path(record['frames_dir']).glob('*.jpg')),images/'single/frame_000042.jpg')
        # Different and even invalid metadata cannot change FPS-independent behavior.
        (images/'original/metadata.json').write_text('{"fps": 15}')
        (images/'reindexed/metadata.json').write_text('{"fps": 240}')
        (images/'single/metadata.json').write_text('not valid json')
        start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        df=runtime.predict(root,PACKAGE).set_index('ID')
        a,b=df.loc['original'],df.loc['reindexed']
        assert b.entry_frame==a.entry_frame*7+100 and b.collision_frame==a.collision_frame*7+100
        assert a.entry_side==b.entry_side and a.evasion_space==b.evasion_space
        assert df.loc['single'].entry_frame==df.loc['single'].collision_frame==42
        assert (df.entry_frame<=df.collision_frame).all()
        assert set(df.entry_side)<={'LEFT','RIGHT'} and set(df.evasion_space)<={0,1}
        results={'sampling_lengths':[1,2,17,128,500,10000],'affine_frame_index_invariance':True,
                 'fps_metadata_independence':True,'single_frame':True,'network_blocked':True,
                 'seconds':time.perf_counter()-start,'peak_gpu_MiB':torch.cuda.max_memory_allocated()/2**20,
                 'predictions':df.reset_index().to_dict('records')}
        (ROOT/'submission_tools/fps_stage2_test_results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results,indent=2))


if __name__=='__main__': main()
