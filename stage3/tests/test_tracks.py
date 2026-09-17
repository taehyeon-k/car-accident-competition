import numpy as np

from stage3.geometry.tracks import advect_scalar


def test_forward_tracks_preserve_translated_values():
    values = np.zeros((8, 10), np.float32)
    values[3, 4] = 7
    flow = np.zeros((2, 8, 10), np.float32)
    flow[0] = 1
    tracked, valid = advect_scalar(values, [flow, flow])
    assert tracked[3, 6] == 7
    assert valid[3, 6]
