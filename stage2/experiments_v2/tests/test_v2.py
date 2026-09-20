"""CPU checks for the v2 modules, including the FPS-blindness contract."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from stage2.experiments_v2.metrics import error_distribution, fpsblind_metrics
from stage2.experiments_v2.models import build_model
from stage2.experiments_v2.modules import MultiRateDifference
from stage2.experiments_v2.objective import (binned_distribution, consistency_loss, decode,
                                             state_labels)
from stage2.experiments_v2.sampling import apply_drop, apply_duplicate, apply_warp, build_view

PACKAGE = Path(__file__).resolve().parents[1]


def test_multi_rate_difference_clamps_at_start():
    module = MultiRateDifference(8, strides=(1, 2), dropout=0.0, gated=False).eval()
    h = torch.randn(2, 6, 8)
    with torch.no_grad():
        out = module(h)
    assert out.shape == h.shape
    # A constant sequence has zero differences at every rate, so the fused
    # residual must not depend on stride bookkeeping at the boundary.
    flat = torch.ones(1, 5, 8)
    with torch.no_grad():
        a = module(flat)
    assert torch.allclose(a[0, 0], a[0, -1], atol=1e-5)


def test_multi_rate_difference_rejects_bad_strides():
    with pytest.raises(ValueError):
        MultiRateDifference(8, strides=(0,), dropout=0.0)


def test_state_labels_three_regions():
    valid = torch.ones(1, 11, dtype=torch.bool)
    labels = state_labels(torch.tensor([5]), valid, radius=2)[0].tolist()
    assert labels == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]


def test_binned_distribution_is_normalized_and_cadence_invariant():
    # Same underlying event position, two different sampling densities.
    def view(count):
        positions = torch.linspace(0, 1, count)[None]
        valid = torch.ones(1, count, dtype=torch.bool)
        logits = -60.0 * (positions - 0.4).abs()
        return logits, valid, positions

    la, va, pa = view(64)
    lb, vb, pb = view(160)
    da = binned_distribution(la, va, pa, 32)
    db = binned_distribution(lb, vb, pb, 32)
    assert torch.allclose(da.sum(-1), torch.ones(1), atol=1e-5)
    assert int(da.argmax(-1)) == int(db.argmax(-1))
    agreement = consistency_loss({"entry_logits": la, "collision_logits": la},
                                 {"time_valid": va, "normalized_positions": pa},
                                 {"entry_logits": lb, "collision_logits": lb},
                                 {"time_valid": vb, "normalized_positions": pb}, 32)
    assert float(agreement) < 0.05


def test_consistency_loss_penalises_disagreement():
    positions = torch.linspace(0, 1, 64)[None]
    valid = torch.ones(1, 64, dtype=torch.bool)
    near = -60.0 * (positions - 0.4).abs()
    far = -60.0 * (positions - 0.9).abs()
    same = consistency_loss({"entry_logits": near, "collision_logits": near},
                            {"time_valid": valid, "normalized_positions": positions},
                            {"entry_logits": near, "collision_logits": near},
                            {"time_valid": valid, "normalized_positions": positions}, 32)
    different = consistency_loss({"entry_logits": near, "collision_logits": near},
                                 {"time_valid": valid, "normalized_positions": positions},
                                 {"entry_logits": far, "collision_logits": far},
                                 {"time_valid": valid, "normalized_positions": positions}, 32)
    assert float(different) > float(same) + 0.1


def test_sampling_preserves_order_and_endpoints():
    rng = np.random.default_rng(0)
    for _ in range(20):
        slots = build_view(192, 128, jitter=0.45, drop_probability=0.05, warp=0.2,
                           duplicate_probability=0.05, rng=rng)
        assert (np.diff(slots) >= 0).all()
        assert slots[0] == 0 and slots[-1] == 191
        assert len(slots) >= 2


def test_drop_and_duplicate_keep_endpoints():
    rng = np.random.default_rng(1)
    selected = np.arange(20)
    assert apply_drop(selected, 0.9, rng)[[0, -1]].tolist() == [0, 19]
    assert apply_duplicate(selected, 1.0, rng).tolist()[0] == 0
    assert (np.diff(apply_warp(selected, 0.3, rng)) >= 0).all()


def test_decode_returns_original_frame_numbers():
    frames = torch.tensor([[10, 20, 30, 40, 50]])
    view = {"frame_numbers": frames,
            "normalized_positions": torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]]),
            "time_valid": torch.ones(1, 5, dtype=torch.bool)}
    outputs = {"entry_logits": torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]]),
               "collision_logits": torch.tensor([[0.0, 0.0, 0.0, 6.0, 0.0]])}
    entry, collision = decode(outputs, view)
    assert int(entry[0]) == 20 and int(collision[0]) == 40
    assert int(entry[0]) <= int(collision[0])


def test_decode_enforces_entry_before_collision():
    view = {"frame_numbers": torch.tensor([[0, 10, 20, 30]]),
            "normalized_positions": torch.tensor([[0.0, 1 / 3, 2 / 3, 1.0]]),
            "time_valid": torch.ones(1, 4, dtype=torch.bool)}
    # Peak ENTRY after peak COLLISION; constrained decoding must still order them.
    outputs = {"entry_logits": torch.tensor([[0.0, 0.0, 0.0, 9.0]]),
               "collision_logits": torch.tensor([[0.0, 9.0, 0.0, 0.0]])}
    entry, collision = decode(outputs, view)
    assert int(entry[0]) <= int(collision[0])


def test_error_distribution_reports_requested_percentiles():
    keys = error_distribution([1, 2, 3, 4], "x")
    assert set(keys) == {"x_mean", "x_median", "x_p25", "x_p50", "x_p75", "x_p90"}


def test_fpsblind_metrics_never_needs_fps():
    predictions = [{"sample_id": "a", "entry_frame": 10, "collision_frame": 20,
                    "entry_gt": 10, "collision_gt": 21, "entry_side": 1, "evasion_space": 0,
                    "entry_side_gt": 1, "evasion_gt": 0, "num_available_frames": 101}]
    result = fpsblind_metrics(predictions)
    assert result["entry_abs_frames_mean"] == 0.0
    assert 0.0 <= result["fpsblind_selection_score"] <= 1.0


@pytest.mark.parametrize("config_name", [p.name for p in sorted((PACKAGE / "configs").glob("*.json"))])
def test_every_config_builds_and_runs_forward(config_name):
    cfg = json.loads((PACKAGE / "configs" / config_name).read_text())
    assert cfg["seed"] == 42, "the screen is fixed to seed 42"
    model = build_model(cfg).eval()
    x = torch.randn(2, 24, cfg["n_tokens"], cfg["feature_dim"], dtype=torch.float16)
    valid = torch.ones(2, 24, dtype=torch.bool)
    with torch.no_grad():
        out = model(x, valid)
    assert {"side_logits", "evasion_logits"} <= set(out)
    assert out["side_logits"].shape == (2, 2)


def executable_source(path):
    """Source with comments and string literals removed, so prose cannot match."""
    import io
    import tokenize
    pieces = []
    with path.open("rb") as handle:
        for tok in tokenize.tokenize(handle.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(tok.string)
    return " ".join(pieces)


def test_training_path_reads_no_fps_field():
    """Static contract: only metrics.py may read native_fps or any duration."""
    offenders = []
    for path in PACKAGE.glob("*.py"):
        if path.name == "metrics.py":
            continue
        code = executable_source(path)
        for token in ("native_fps", "duration", "seconds_per_frame", "fps"):
            # data.py legitimately reads the cache's own fps_blind guard keys.
            if path.name == "data.py" and token == "fps":
                continue
            if re.search(rf"\b{token}\b", code):
                offenders.append(f"{path.name}:{token}")
    assert not offenders, offenders
