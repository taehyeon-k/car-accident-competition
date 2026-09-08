import numpy as np
import torch
from stage2.data.sampling import (
    sort_frame_paths,
    build_coarse_bins,
    recover_region,
    sliding_windows,
)
from stage2.model.tracking import Detection, HungarianTracker
from stage2.model.geometry import tubelet_geometry
from stage2.model.model import CoarseModel, FineModel
from stage2.utils.losses import coarse_loss, fine_loss


def test_frame_ids_and_windows():
    paths = ["frame_000020.jpg", "frame_000003.jpg", "frame_000105.jpg"]
    _, ids = sort_frame_paths(paths)
    assert ids == [3, 20, 105]
    bins = build_coarse_bins(
        list(range(100)),
        list(range(100)),
        False,
    )
    assert bins.valid.sum() == 32
    region = recover_region(
        bins,
        15,
    )
    assert np.all(np.diff(region) == 1)
    long = np.arange(150)
    windows = sliding_windows(long)
    assert (
        windows[0][0] == 0
        and windows[-1][-1] == 149
        and all(len(x) == 64 for x in windows)
    )


def test_tracker_rejects_far_assignments():
    tracker = HungarianTracker(
        max_center_distance=0.1,
        max_match_cost=0.65,
    )
    tracks = tracker.track(
        [
            [
                Detection(
                    np.array([0, 0, 10, 10.0]),
                    0.9,
                    "car",
                )
            ],
            [
                Detection(
                    np.array([90, 90, 100, 100.0]),
                    0.9,
                    "car",
                )
            ],
        ],
        [(100, 100), (100, 100)],
    )
    assert len(tracks) == 2


def test_tubelet_contract():
    frame = torch.randn(
        2,
        32,
        12,
        9,
    )
    mask = torch.ones(
        2,
        32,
        12,
        dtype=torch.bool,
    )
    out, valid = tubelet_geometry(
        frame,
        mask,
    )
    assert out.shape == (2, 16, 12, 9) and valid.shape == (2, 16, 12)


def test_coarse_and_fine_forward_losses():
    b = 1
    coarse = CoarseModel().eval()
    out = coarse(
        torch.randn(
            b,
            16,
            24,
            24,
            768,
        ),
        torch.rand(
            b,
            16,
            12,
            4,
        )
        * 23,
        torch.randn(
            b,
            16,
            12,
            9,
        ),
        torch.ones(
            b,
            16,
            12,
            dtype=torch.bool,
        ),
        torch.ones(
            b,
            32,
            dtype=torch.bool,
        ),
    )
    assert out["entry_logits"].shape == (b, 32)
    loss, _ = coarse_loss(
        out,
        {
            "entry_bin": torch.tensor([3]),
            "collision_bin": torch.tensor([7]),
            "entry_side": torch.tensor([0]),
            "evasion": torch.tensor([1]),
        },
    )
    assert torch.isfinite(loss)
    fine = FineModel().eval()
    out = fine(
        torch.randn(
            b,
            64,
            384,
        ),
        torch.randn(
            b,
            64,
            24,
            24,
            384,
        ),
        torch.rand(
            b,
            64,
            12,
            4,
        )
        * 23,
        torch.randn(
            b,
            64,
            12,
            9,
        ),
        torch.ones(
            b,
            64,
            12,
            dtype=torch.bool,
        ),
        torch.ones(
            b,
            64,
            dtype=torch.bool,
        ),
    )
    assert out["collision_logits"].shape == (b, 64)
    loss, _ = fine_loss(
        out,
        {
            "event_type": torch.tensor([1]),
            "time_valid": torch.ones(
                b,
                64,
                dtype=torch.bool,
            ),
            "event_local_index": torch.tensor([31]),
        },
    )
    assert torch.isfinite(loss)
