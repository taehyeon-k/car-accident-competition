from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stage3.data.adapters.base import Signals
from stage3.data.cache import dequantize_motion, quantize_motion
from stage3.data.targets import make_targets
from stage3.data.timing import decode_dacon_stage3_video
from stage3.utils.checkpoint import load_checkpoint, save_checkpoint
from stage3.utils.config import load_config


def test_dacon_ignores_missing_duplicate_and_nonmonotonic_pts(monkeypatch):
    import av
    class Container:
        streams = SimpleNamespace(video=[object()])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def decode(self, stream):
            for i, pts in enumerate([0, 1, 1, None, -1]):
                yield SimpleNamespace(pts=pts, to_ndarray=lambda format, i=i: np.full((4, 4, 3), i, np.uint8))
    monkeypatch.setattr(av, 'open', lambda _: Container())
    result = decode_dacon_stage3_video('unused')
    assert len(result.frames) == 5
    assert [int(f[0, 0, 0]) for f in result.frames] == list(range(5))
    assert np.allclose(result.actual_times, np.arange(5) * .1)


def test_invalid_signals_and_large_gaps_remain_invalid_after_smoothing():
    t = np.r_[np.arange(10) * .1, np.arange(20, 30) * .1]
    valid = np.ones(len(t), bool)
    valid[4] = False
    a = np.zeros(len(t)); a[4] = 50
    signals = Signals(t, np.ones(len(t)), a, np.zeros(len(t)), valid=valid)
    grid = np.arange(30) * .1
    targets = make_targets(signals, grid, {})
    assert not targets['valid_accel'][4]
    assert not targets['valid_accel'][10:20].any()
    assert not targets['valid_accel_speed'][9:21].any()
    assert np.allclose(targets['a_long_s1'][targets['valid_accel']], 0)
    assert not targets['valid_speed'][4]


def test_crop_dequantization_is_identical_to_full_decode():
    rng = np.random.default_rng(42)
    q, offset, scale = quantize_motion(rng.normal(size=(17, 10, 8, 8)).astype(np.float32))
    cache = {'motion_q': q, 'motion_offset': offset, 'motion_scale': scale}
    assert torch.equal(dequantize_motion(cache, 3, 9), dequantize_motion(cache)[3:9])


def test_checkpoint_feature_version_and_configuration_are_verified(tmp_path):
    cfg = load_config('stage3/configs/smoke.workspace.yaml')
    path = tmp_path / 'model.pt'
    save_checkpoint(path, config=cfg, model={})
    state = load_checkpoint(path)
    legacy = dict(state); legacy.pop('feature_version')
    torch.save(legacy, path)
    with pytest.raises(ValueError, match='feature version'):
        load_checkpoint(path)
    state['config']['calibration']['hfov_prior_deg'] = 80
    torch.save(state, path)
    with pytest.raises(ValueError, match='feature version'):
        load_checkpoint(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_cnn_chunks_match_full_sequence():
    from stage3.model import Stage3MotionModel
    from stage3.tests.test_model import MODEL
    torch.manual_seed(5)
    model = Stage3MotionModel(MODEL).cuda().eval()
    motion = torch.randn(1, 37, 10, 96, 168)
    physics = torch.randn(1, 37, 20, device='cuda')
    with torch.inference_mode(), torch.backends.cudnn.flags(allow_tf32=False):
        full = model(motion.cuda(), physics)
        chunks = model(motion, physics, chunk_frames=8)
    for name in full:
        torch.testing.assert_close(chunks[name], full[name], atol=2e-5, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_geometry_matches_cpu_with_motion_and_stopped_frames():
    from stage3.geometry.pipeline import build_motion_features
    from stage3.geometry.rotation import rotational_design
    rng = np.random.default_rng(3)
    t, h, w = 12, 96, 168
    y, x = np.mgrid[:h, :w]
    flows = np.stack([np.stack((x - w/2, y - h/2)) * (.015 + i*.0001) for i in range(t)]).astype(np.float32)
    flows += np.einsum('hwkc,c->khw', rotational_design(h, w, w/2), [.001, -.002, .001])[None].astype(np.float32)
    flows += rng.normal(0, .02, flows.shape).astype(np.float32)
    flows[0] = 0
    confidence = np.ones((t, h, w), np.float32)
    confidence[5] = 0
    args = (flows, confidence, np.arange(t)*.1, {'focal_mode': 'prior', 'hfov_prior_deg': 90})
    reference = build_motion_features(*args, tracking_device='cpu')
    actual = build_motion_features(*args, tracking_device='cuda', tracking_batch_size=4)
    np.testing.assert_allclose(actual[0], reference[0], atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(actual[1], reference[1], atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(actual[2]['foe'], reference[2]['foe'], atol=1e-3, rtol=1e-4)
