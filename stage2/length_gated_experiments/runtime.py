"""Feature-level hybrid: old P2 for short clips, NEXAR specialist for long clips.

All inference inputs are frozen DINO features and original frame numbers. No
manifest, source ID, FPS, duration, timestamp, or annotation is consulted.
"""
from pathlib import Path
import torch
from stage2.long_context_v2_experiments import common as C
from stage2.spotting_experiments.objective import constrained_anchors
from stage2.nexar65_experiments.runtime import NexarSpecialist

THRESHOLD_FRAMES=500

class Hybrid:
    def __init__(self, specialist='candidate_all_seeds', device='cuda'):
        self.device=torch.device(device)
        self.p2,_=C.load_named('P2_ens4',self.device)
        candidate=Path(__file__).resolve().parents[1]/'nexar65_experiments'/specialist
        self.specialist=NexarSpecialist(candidate,self.device)

    @torch.inference_mode()
    def predict(self,features,frame_numbers,total_frames):
        # `frame_numbers` contains only the adaptively sampled 128–320 frames.
        # The route must use the original number of frames in the clip.
        if total_frames>THRESHOLD_FRAMES:
            return self.specialist.predict(features,frame_numbers)
        x=torch.as_tensor(features,device=self.device)[None]
        valid=torch.ones(1,len(frame_numbers),dtype=torch.bool,device=self.device)
        out=self.p2(x,valid)
        entry,collision=constrained_anchors(out['entry_logits'],out['collision_logits'])
        return {'entry_frame':int(frame_numbers[int(entry)]),'collision_frame':int(frame_numbers[int(collision)]),
                'entry_side':int(out['side_logits'][0].argmax()),'evasion_space':int(out['evasion_logits'][0]>=0)}
