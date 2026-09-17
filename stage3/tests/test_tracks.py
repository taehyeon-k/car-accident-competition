import numpy as np

from stage3.geometry.tracks import advect_lagged_fields, advect_scalar


def test_forward_tracks_preserve_translated_values():
    values = np.zeros((8, 10), np.float32)
    values[3, 4] = 7
    flow = np.zeros((2, 8, 10), np.float32)
    flow[0] = 1
    tracked, valid = advect_scalar(values, [flow, flow])
    assert tracked[3, 6] == 7
    assert valid[3, 6]


def test_batched_lagged_tracking_matches_scalar_reference():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(7, 8, 10)).astype(np.float32)
    flows = rng.uniform(-0.4, 0.4, size=(7, 2, 8, 10)).astype(np.float32)
    tracked, valid = advect_lagged_fields(values, flows, lag=4, device="cpu", batch_size=2)
    assert not valid[:4].any()
    for current in range(4, len(values)):
        expected, expected_valid = advect_scalar(values[current - 4], list(flows[current - 3 : current + 1]))
        assert np.array_equal(valid[current], expected_valid)
        assert np.allclose(tracked[current], expected, atol=2e-5)
