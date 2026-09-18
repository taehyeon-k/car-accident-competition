from copy import deepcopy

from stage3.data.cache import cache_key


def test_target_change_does_not_change_motion_cache_key():
    cfg = {
        "flow": {"backend": "x", "version": "1", "working_size": [1, 1], "checkpoint_sha256": "a"},
        "calibration": {"focal_mode": "prior", "hfov_prior_deg": 90},
        "geometry": {"canonical_size": [96, 168]}, "targets": {"smoothing_seconds": [0.5, 1.5]},
    }
    changed = deepcopy(cfg)
    changed["targets"]["smoothing_seconds"] = [0.7, 2.0]
    assert cache_key(cfg) == cache_key(changed)


def test_training_rejects_stale_cache(tmp_path):
    import json
    import pytest
    import torch
    from stage3.data.dataset import CachedMotionDataset

    path = tmp_path / 'old.pt'
    torch.save({'schema': 2, 'signals': {}, 'cache_key': 'old-geometry'}, path)
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps({'cache_path': str(path)}) + '\n')
    dataset = CachedMotionDataset(str(manifest), expected_cache_key='new-geometry')
    with pytest.raises(ValueError, match='Stale motion cache'):
        dataset[0]
