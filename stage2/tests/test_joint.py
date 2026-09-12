import torch

from stage2.data.cache_joint_features import clip_positions, object_tensors
from stage2.data.joint import joint_collate
from stage2.model.joint import JointStage2Model
from stage2.model.tracking import Detection, Track
from stage2.utils.joint_losses import constrained_decode, joint_loss


def _item(length, global_length):
    return {
        "scene_features": torch.randn(length, 7, 768, dtype=torch.float16),
        "roi_features": torch.randn(length, 12, 768, dtype=torch.float16),
        "geometry": torch.randn(length, 12, 13),
        "object_valid": torch.rand(length, 12) > 0.3,
        "time_valid": torch.ones(length, dtype=torch.bool),
        "local_time": torch.linspace(0, 1, length),
        "frame_seconds": torch.arange(length) / 15,
        "global_features": torch.randn(global_length, 1024, dtype=torch.float16),
        "global_time": torch.linspace(0, 1, global_length),
        "global_valid": torch.ones(global_length, dtype=torch.bool),
        "entry_index": 2,
        "entry_supervised": True,
        "collision_index": length - 2,
        "entry_side": 1,
        "evasion": 0.0,
        "frame_ids": torch.arange(length),
        "sample_id": f"sample-{length}",
        "source_id": f"source-{length}",
    }


def test_joint_forward_backward_and_ordered_decode():
    batch = joint_collate([_item(19, 8), _item(13, 5)])
    model = JointStage2Model()
    outputs = model(batch)
    assert outputs["entry_logits"].shape == (2, 19)
    assert outputs["collision_logits"].shape == (2, 19)
    assert outputs["side_logits"].shape == (2, 2)
    assert outputs["evasion_logits"].shape == (2,)
    loss, parts = joint_loss(outputs, batch)
    assert torch.isfinite(loss) and set(parts) >= {"entry", "collision", "side"}
    loss.backward()
    entry, collision = constrained_decode(
        outputs["entry_logits"], outputs["collision_logits"]
    )
    assert torch.all(entry <= collision)


def test_sampling_policy_and_geometry_contract():
    clips = clip_positions(150)
    assert clips[0].tolist() == list(range(0, 61, 4))
    assert int(clips[-1][-1]) == 149
    detection = Detection(torch.tensor([10, 20, 30, 60]).numpy(), 0.9, "car", 2.0)
    track = Track(0, "car", {0: detection, 2: detection}, 2)
    boxes, geometry, valid = object_tensors([track], [(100, 100)] * 3, 15.0)
    assert boxes.shape == (3, 12, 4)
    assert geometry.shape == (3, 12, 13)
    assert valid[:, 0].tolist() == [True, False, True]
