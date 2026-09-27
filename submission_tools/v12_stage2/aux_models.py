"""Inference-only AuxPyramid (stage2/aux_signal_experiments/model.py) for phase_rep=none members.

With phase_rep=none, PhasePyramid.forward is exactly LCPyramid.forward (coarse=none, phase=None), so the members are an
LCPyramid plus a motion projection of arbitrary input width (28 global + 72 residual = 100 for motion=both). The boundary head
(bnd2) is training-only; it is instantiated so the checkpoint loads strictly, but its output is never used.
"""
from torch import nn

from lc_models import LCPyramid


class AuxPyramid(LCPyramid):
    def __init__(self, motion_dim, boundary='none', feat_dim=384):
        super().__init__(phase=0, motion=False, feat_dim=feat_dim)
        hidden = self.event.in_features
        if boundary == 'bnd1': self.bnd = nn.Linear(hidden, 2)
        elif boundary == 'bnd2': self.bnd = nn.Linear(2 * hidden, 2)
        elif boundary == 'bnd3': self.bnd = nn.Conv1d(hidden, 2, 3, padding=1)
        if motion_dim:
            self.uses_motion = True
            self.motion_proj = nn.Sequential(nn.LayerNorm(motion_dim), nn.Linear(motion_dim, hidden), nn.GELU(), nn.Dropout(0.35),
                                             nn.Linear(hidden, hidden))
