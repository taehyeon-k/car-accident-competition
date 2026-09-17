import numpy as np
import pytest

from stage3.data.adapters.base import Signals
from stage3.data.targets import make_targets


CFG = {"smoothing_seconds": [0.5, 1.5], "require_direct_acceleration": True, "stopped_speed_mps": 0.15}


def test_direct_acceleration_is_primary_and_speed_derivative_is_separate():
    t = np.arange(0, 5, 0.01)
    signals = Signals(t, t**2, np.full_like(t, -1.25), np.sin(t) * 10)
    frame_t = np.arange(0, 4.9, 0.1)
    targets = make_targets(signals, frame_t, CFG)
    assert np.allclose(targets["a_long_s1"], -1.25)
    assert np.nanmedian(targets["a_dvdt_s1"]) > 3.0
    assert not np.allclose(targets["a_long_s1"], targets["a_dvdt_s1"])


def test_speed_smoothing_does_not_replace_direct_acceleration():
    t = np.arange(0, 5, 0.01)
    signals = Signals(t, np.sin(t * 9), np.cos(t), np.zeros_like(t))
    a = make_targets(signals, np.arange(0, 4.9, 0.1), CFG)
    altered = {**CFG, "smoothing_seconds": [0.9, 1.9]}
    b = make_targets(signals, np.arange(0, 4.9, 0.1), altered)
    assert np.corrcoef(a["a_long_s1"], np.cos(np.arange(0, 4.9, 0.1)))[0, 1] > 0.99
    assert not np.allclose(a["a_dvdt_s1"], b["a_dvdt_s1"])


def test_missing_direct_acceleration_is_explicit():
    t = np.arange(20) * 0.1
    with pytest.raises(ValueError, match="Direct longitudinal"):
        make_targets(Signals(t, t, None, t), t, CFG)
