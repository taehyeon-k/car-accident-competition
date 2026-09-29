import pytest
import torch

from stage3.utils.checkpoint import atomic_gzip_save, atomic_save, load_artifact


@pytest.mark.parametrize('compressed', [False, True])
@pytest.mark.parametrize('weights_only', [False, True])
def test_artifact_roundtrip_preserves_tensor_views_and_metadata(tmp_path, compressed, weights_only):
    tensor = torch.arange(120, dtype=torch.uint8).reshape(3, 4, 10)
    artifact = {
        'motion_q': tensor,
        'view': tensor[:, :, ::2],
        'signals': {'speed': torch.tensor([0.0, 1.5]), 'yaw': None},
        'cache_key': 'unchanged',
        'schema': 2,
    }
    path = tmp_path / 'cache.pt'
    (atomic_gzip_save if compressed else atomic_save)(artifact, path)
    result = load_artifact(path, weights_only=weights_only)
    assert result['cache_key'] == artifact['cache_key']
    assert result['schema'] == artifact['schema']
    assert result['signals']['yaw'] is None
    for key in ('motion_q', 'view'):
        assert torch.equal(result[key], artifact[key])
        assert result[key].dtype == artifact[key].dtype
        assert result[key].stride() == artifact[key].stride()
    assert torch.equal(result['signals']['speed'], artifact['signals']['speed'])
