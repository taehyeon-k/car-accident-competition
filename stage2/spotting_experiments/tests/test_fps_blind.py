import numpy as np
import torch
from stage2.spotting_experiments.sampling import normalized_indices
from stage2.spotting_experiments.objective import decode


def test_sampling_affine_invariant():
    frames=np.array([0,2,5,9,15,22,30]); a=normalized_indices(frames,16); b=normalized_indices(frames*7+101,16)
    assert np.array_equal(a,b)


def test_offset_decode_returns_ordered_original_frames():
    outputs={"entry_logits":torch.tensor([[0.,3.,0.]]),"collision_logits":torch.tensor([[0.,0.,4.]]),
             "entry_offsets":torch.zeros(1,3),"collision_offsets":torch.zeros(1,3)}
    batch={"normalized_positions":torch.tensor([[0.,.5,1.]]),"frame_numbers":torch.tensor([[10,20,30]]),"time_valid":torch.ones(1,3,dtype=torch.bool)}
    e,c=decode(outputs,batch); assert (int(e),int(c))==(20,30)
