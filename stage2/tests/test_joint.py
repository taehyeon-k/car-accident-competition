"""Joint architecture, data, gradient-flow and metric regression tests."""

import json
from contextlib import contextmanager
from pathlib import Path

import math

import numpy as np
import pytest
import torch
from torch import nn

from stage2.data.augment import (
    ClipPhotometric,
    augmentation_config,
    flip_boxes,
    flip_side,
    sample_photometric,
)
from stage2.data.joint import (
    JointFeatureDataset,
    crop_objects,
    joint_collate,
    joint_item,
    load_detections,
)
from stage2.data.joint_sampling import (
    choose_crop,
    clip_positions,
    temporal_probabilities,
)
from stage2.model.lora import LoRALinear, add_lora_to_last_blocks
from stage2.model.joint import (
    HybridTemporalBlock,
    JointStage2Model,
    MaskedDilatedConv,
    TemporalCrossAttention,
    head_config,
)
from stage2.model.joint_system import JointSystem
from stage2.model.joint_tracking import (
    GEOMETRY_CHANNELS,
    GEOMETRY_DIM,
    object_tensors,
    ego_lane_score,
    percentile_rank,
)
from stage2.model.tracking import Detection, Track
from stage2.utils.joint_losses import constrained_decode, joint_loss
from stage2.utils.joint_metrics import joint_metric_packet, JointMetricAccumulator

SETTINGS = head_config(None)
HIDDEN_DIM = SETTINGS["hidden_dim"]
ROI_DIM = SETTINGS["roi_dim"]
GEOMETRY_EMBEDDING = SETTINGS["geometry_embedding_dim"]
SCENE_TOKEN_COUNT = 17
OBJECT_SLOTS = 12


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _item(length):
    return {
        "scene_features": torch.randn(length, 17, 768, dtype=torch.float16),
        "roi_features": torch.randn(length, 12, 768, dtype=torch.float16),
        "geometry": torch.randn(length, 12, GEOMETRY_DIM),
        "object_valid": torch.rand(length, 12) > 0.3,
        "time_valid": torch.ones(length, dtype=torch.bool),
        "local_time": torch.linspace(0, 1, length),
        "frame_seconds": torch.arange(length) / 15,
        "entry_index": min(2, length - 1),
        "collision_index": max(0, length - 2),
        "entry_side": 1,
        "evasion": 0.0,
        "frame_ids": torch.arange(length),
        "sample_id": f"sample-{length}",
        "source_id": f"source-{length}",
    }


def head_batch(items):
    batch = joint_collate(items)
    longest = max(len(item["time_valid"]) for item in items)
    for name in ("scene_features", "roi_features"):
        first = items[0][name]
        padded = first.new_zeros((len(items), longest, *first.shape[1:]))
        for i, item in enumerate(items):
            padded[i, : len(item[name])] = item[name]
        batch[name] = padded
    batch.update(
        global_features=torch.randn(len(items), 8, 1024),
        global_time=torch.linspace(0, 1, 8).expand(len(items), -1),
        global_valid=torch.ones(len(items), 8, dtype=torch.bool),
    )
    return batch


def test_joint_forward_backward_and_ordered_decode():
    batch = head_batch([_item(19), _item(13)])
    model = JointStage2Model()
    outputs = model(batch)
    assert outputs["entry_logits"].shape == (2, 19)
    assert outputs["side_logits"].shape == (2, 2)
    assert outputs["evasion_logits"].shape == (2,)
    loss, parts = joint_loss(outputs, batch)
    assert torch.isfinite(loss)
    expected = (
        0.35 * parts["loss_entry"]
        + 0.35 * parts["loss_collision"]
        + 0.15 * parts["loss_entry_side"]
        + 0.15 * parts["loss_evasion_space"]
        + 0.05 * parts["loss_invalid_order"]
    )
    torch.testing.assert_close(loss.detach(), expected)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    entry, collision = constrained_decode(
        outputs["entry_logits"], outputs["collision_logits"]
    )
    assert torch.all(entry <= collision)
    assert torch.isneginf(outputs["entry_logits"][1, 13:]).all()


def test_attribute_losses_do_not_backpropagate_to_event_logits():
    batch = head_batch([_item(9)])
    batch["object_valid"].zero_()
    batch["scene_features"] = batch["scene_features"].float().requires_grad_()
    model = JointStage2Model()
    outputs = model(batch)
    outputs["entry_logits"].retain_grad()
    outputs["collision_logits"].retain_grad()
    (
        outputs["side_logits"].square().sum() + outputs["evasion_logits"].square().sum()
    ).backward()
    assert outputs["entry_logits"].grad is None
    assert outputs["collision_logits"].grad is None
    assert model.entry_head[0].weight.grad is None
    assert model.collision_head[0].weight.grad is None
    assert model.entry_spatial.attention.in_proj_weight.grad.abs().sum() > 0
    assert batch["scene_features"].grad.abs().sum() > 0


def test_padding_does_not_change_predictions():
    model = JointStage2Model().eval()
    short = _item(7)
    single = head_batch([short])
    mixed = head_batch([short, _item(15)])
    for name in ("global_features", "global_time", "global_valid"):
        mixed[name][0] = single[name][0]
    mixed["scene_features"][0, 7:] = float("nan")
    # Validity clearing happens before attention; padded raw scene NaNs must not leak.
    with torch.no_grad():
        one, many = model(single), model(mixed)
    for name in ("entry_logits", "collision_logits"):
        torch.testing.assert_close(
            one[name][0], many[name][0, :7], atol=1e-5, rtol=1e-4
        )
    for name in ("side_logits", "evasion_logits"):
        torch.testing.assert_close(one[name][0], many[name][0], atol=1e-5, rtol=1e-4)


def test_head_dimensions_are_narrow_and_consistent():
    model = JointStage2Model()
    # Scene tokens, V-JEPA projection and the object token all land on hidden_dim.
    assert model.scene_projection[0].out_features == HIDDEN_DIM
    assert model.scene_projection[1].normalized_shape == (HIDDEN_DIM,)
    assert model.scene_positions.shape == (SCENE_TOKEN_COUNT, HIDDEN_DIM)
    assert model.global_projection[0].in_features == 1024
    assert model.global_projection[0].out_features == HIDDEN_DIM
    # Appearance and geometry must add up exactly, not be reconciled later.
    assert model.roi_projection[0].out_features == ROI_DIM == 192
    assert model.geometry_projection[-1].normalized_shape == (GEOMETRY_EMBEDDING,)
    assert ROI_DIM + GEOMETRY_EMBEDDING == HIDDEN_DIM == 256
    # Exactly one spatial layer and one hybrid temporal block survive.
    assert len(model.spatial.encoder.layers) == 1
    hybrids = [m for m in model.temporal if isinstance(m, HybridTemporalBlock)]
    dilated = [m for m in model.temporal if isinstance(m, MaskedDilatedConv)]
    assert len(hybrids) == 1 and len(dilated) == 1 and len(model.temporal) == 2
    block = hybrids[0]
    assert (block.dim, block.heads, block.head_dim) == (HIDDEN_DIM, 4, 64)
    assert block.ffn[0].out_features == 768
    assert all(
        conv.in_channels == conv.out_channels == HIDDEN_DIM for conv in block.convs
    )
    assert block.conv_projection.in_features == 3 * HIDDEN_DIM
    assert dilated[0].pointwise.in_channels == HIDDEN_DIM
    # Fusion keeps four heads of 64.
    assert isinstance(model.fusion, TemporalCrossAttention)
    assert (model.fusion.dim, model.fusion.heads, model.fusion.head_dim) == (
        HIDDEN_DIM,
        4,
        64,
    )
    assert model.fusion.gate.in_features == 2 * HIDDEN_DIM
    # ENTRY and COLLISION keep separate spatial attention modules.
    assert model.entry_spatial is not model.collision_spatial
    for module in (model.entry_spatial, model.collision_spatial):
        assert module.attention.embed_dim == HIDDEN_DIM
        assert module.attention.num_heads == 4
    for head in (model.entry_head, model.collision_head):
        assert (head[0].in_features, head[0].out_features) == (HIDDEN_DIM, 64)
        assert head[-1].out_features == 1
    assert model.side_head[-1].out_features == 2
    assert model.evasion_head[-1].out_features == 1
    for head in (model.side_head, model.evasion_head):
        assert head[0].in_features == HIDDEN_DIM and head[0].out_features == 64
        assert isinstance(head[2], nn.Dropout) and head[2].p == pytest.approx(0.15)


def test_joint_head_parameter_budget_is_about_three_million():
    total = sum(p.numel() for p in JointStage2Model().parameters() if p.requires_grad)
    # The pre-narrowing head was 9,257,560 randomly initialised parameters.
    assert 2.5e6 < total < 3.5e6, total
    assert 9_257_560 / total > 2.5


def test_internal_tensor_widths_are_hidden_dim():
    """Scene, object and fused representations must all be hidden_dim wide."""
    model = JointStage2Model().eval()
    batch = head_batch([_item(9), _item(5)])
    captured = {}

    def spy(name):
        def hook(_module, _inputs, output):
            captured[name] = output

        return hook

    model.scene_projection.register_forward_hook(spy("scene"))
    model.roi_projection.register_forward_hook(spy("roi"))
    model.geometry_projection.register_forward_hook(spy("geometry"))
    model.global_projection.register_forward_hook(spy("global"))
    model.spatial.register_forward_hook(spy("spatial"))
    model.fusion.register_forward_hook(spy("fusion"))
    with torch.no_grad():
        outputs = model(batch)

    assert captured["scene"].shape[-1] == HIDDEN_DIM
    assert captured["roi"].shape[-1] == ROI_DIM
    assert captured["geometry"].shape[-1] == GEOMETRY_EMBEDDING
    assert captured["roi"].shape[-1] + captured["geometry"].shape[-1] == HIDDEN_DIM
    assert captured["global"].shape[-1] == HIDDEN_DIM
    # The spatial transformer sees 17 scene tokens + 12 objects per frame.
    assert captured["spatial"].shape[1:] == (
        SCENE_TOKEN_COUNT + OBJECT_SLOTS,
        HIDDEN_DIM,
    )
    assert captured["spatial"].shape[1] == 29
    assert captured["fusion"].shape[-1] == HIDDEN_DIM
    assert outputs["hidden"].shape[-1] == HIDDEN_DIM


def test_external_prediction_shapes_are_unchanged():
    """Narrowing must not alter the Stage 2 prediction or loss interface."""
    model = JointStage2Model().eval()
    batch = head_batch([_item(11), _item(7)])
    with torch.no_grad():
        outputs = model(batch)
    length = batch["time_valid"].shape[1]
    assert outputs["entry_logits"].shape == (2, length)
    assert outputs["collision_logits"].shape == (2, length)
    assert outputs["side_logits"].shape == (2, 2)
    assert outputs["evasion_logits"].shape == (2,)
    assert set(outputs) == {
        "entry_logits",
        "collision_logits",
        "side_logits",
        "evasion_logits",
        "hidden",
    }
    loss, parts = joint_loss(outputs, batch)
    assert torch.isfinite(loss)
    assert set(parts) == {
        "loss_entry",
        "loss_collision",
        "loss_entry_side",
        "loss_evasion_space",
        "loss_invalid_order",
    }


def test_head_dimensions_are_configurable():
    model = JointStage2Model(
        config={
            "hidden_dim": 128,
            "roi_dim": 96,
            "geometry_embedding_dim": 32,
            "spatial": {"layers": 1, "heads": 2, "ffn_dim": 256, "dropout": 0.0},
            "temporal": {
                "hybrid_blocks": 1,
                "heads": 2,
                "ffn_dim": 256,
                "dropout": 0.0,
            },
            "attribute_hidden_dim": 32,
            "attribute_dropout": 0.0,
        }
    )
    assert model.hidden_dim == 128
    assert model.scene_projection[0].out_features == 128
    assert model.roi_projection[0].out_features == 96
    smaller = sum(p.numel() for p in model.parameters() if p.requires_grad)
    default = sum(p.numel() for p in JointStage2Model().parameters() if p.requires_grad)
    assert smaller < default


def test_hybrid_block_locality_and_empty_rows():
    block = HybridTemporalBlock(radius=16).eval()
    x = torch.randn(2, 65, HIDDEN_DIM)
    valid = torch.ones(2, 65, dtype=torch.bool)
    valid[1] = False
    with torch.no_grad():
        baseline = block(x, valid)
        x[:, 40:] += 100
        changed = block(x, valid)
    torch.testing.assert_close(baseline[0, 10], changed[0, 10])
    assert baseline[1].count_nonzero() == 0
    assert torch.isfinite(baseline).all()


@pytest.mark.parametrize("length", [1, 2, 15, 60, 61, 62, 109, 150])
def test_sampling_policy(length):
    clips = clip_positions(length)
    assert all(
        len(clip) == 16 and int(clip.min()) >= 0 and int(clip.max()) < length
        for clip in clips
    )
    assert int(clips[-1][-1]) == length - 1
    if length >= 61:
        assert all(
            torch.equal(torch.diff(clip), torch.full((15,), 4)) for clip in clips
        )
    else:
        assert torch.equal(clips[0], torch.linspace(0, length - 1, 16).round().long())


def det(x=40, score=0.9):
    return Detection(np.array([x, 40, x + 20, 80], dtype=np.float32), score, "car")


def test_persistent_slots_and_deterministic_selection():
    tracks = [
        Track(10, "car", {0: det(), 2: det(42)}, 2),
        Track(20, "car", {1: det(5), 2: det(8)}, 2),
    ]
    first = object_tensors(tracks, [(100, 100)] * 3, return_track_ids=True)
    second = object_tensors(
        list(reversed(tracks)), [(100, 100)] * 3, return_track_ids=True
    )
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)
    boxes, geometry, valid, ids = first
    for track in tracks:
        slot = int((ids == track.id).nonzero()[0])
        assert valid[:, slot].tolist() == [t in track.observations for t in range(3)]
        for t, detection in track.observations.items():
            torch.testing.assert_close(boxes[t, slot], torch.from_numpy(detection.box))
    assert not geometry[~valid].count_nonzero()
    assert len(set(ids[ids >= 0].tolist())) == 2


def test_geometry_is_nine_bbox_channels_without_depth():
    assert GEOMETRY_DIM == 9
    assert GEOMETRY_CHANNELS == (
        "center_x",
        "bottom_y",
        "width",
        "height",
        "dx_per_frame",
        "dy_per_frame",
        "log_area_growth_per_frame",
        "detection_confidence",
        "track_continuity",
    )
    # Two observations three frames apart: motion and growth are per frame.
    first = Detection(np.array([40, 40, 60, 80], dtype=np.float32), 0.8, "car")
    second = Detection(np.array([46, 46, 76, 96], dtype=np.float32), 0.6, "car")
    track = Track(1, "car", {0: first, 3: second}, 3)
    _, geometry, valid, _ = object_tensors(
        [track], [(100, 100)] * 4, return_track_ids=True
    )
    assert geometry.shape == (4, 12, GEOMETRY_DIM)
    channel = {name: i for i, name in enumerate(GEOMETRY_CHANNELS)}

    start = geometry[0, 0]
    assert start[channel["center_x"]] == pytest.approx(0.5)
    assert start[channel["bottom_y"]] == pytest.approx(0.8)
    assert start[channel["width"]] == pytest.approx(0.2)
    assert start[channel["height"]] == pytest.approx(0.4)
    assert start[channel["detection_confidence"]] == pytest.approx(0.8)
    # A first observation has no previous frame, so all motion channels are zero.
    assert start[channel["dx_per_frame"]] == 0
    assert start[channel["dy_per_frame"]] == 0
    assert start[channel["log_area_growth_per_frame"]] == 0

    later = geometry[3, 0]
    assert later[channel["dx_per_frame"]] == pytest.approx((0.61 - 0.5) / 3, abs=1e-6)
    assert later[channel["dy_per_frame"]] == pytest.approx((0.71 - 0.6) / 3, abs=1e-6)
    growth = math.log(0.3 * 0.5 + 1e-8) - math.log(0.2 * 0.4 + 1e-8)
    assert later[channel["log_area_growth_per_frame"]] == pytest.approx(
        growth / 3, abs=1e-6
    )
    assert later[channel["track_continuity"]] == pytest.approx(0.4)
    # Frames with no observation stay zero-filled and masked.
    assert valid[:, 0].tolist() == [True, False, False, True]
    assert not geometry[1:3, 0].count_nonzero()


def test_ranking_lane_and_missing_history():
    assert ego_lane_score(0.5, 0.2) == 0
    assert ego_lane_score(0.5, 1) == 1
    assert ego_lane_score(0.9, 1) == 0
    assert percentile_rank(0, [0, 0, 0]) == 0
    tracks = [Track(i, "car", {0: det(0, 0.99)}, 0) for i in range(13)]
    tracks.append(Track(99, "car", {0: det(40, 0.21)}, 0))
    _, _, _, ids = object_tensors(tracks, [(100, 100)], return_track_ids=True)
    assert ids[0] == 99  # Lane relevance wins even with low confidence and no history.
    assert len(ids) == 12


class DummyDino(nn.Module):
    """Position-dependent patch tokens so ROI pooling is verifiable."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.blocks = nn.ModuleList()
        self.seen = []

    def forward_features(self, images):
        self.seen.append(images.detach().clone())
        batch = len(images)
        dense = (
            torch.arange(576, dtype=torch.float32)
            .reshape(1, 576, 1)
            .expand(batch, -1, 768)
        )
        return {
            "x_norm_clstoken": torch.zeros(batch, 768),
            "x_norm_patchtokens": dense * self.scale,
        }


class DummyVjepa(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = []
        self.seen = []
        self.blocks = nn.ModuleList()

    def forward(self, video):
        self.calls.append(tuple(video.shape))
        self.seen.append(video.detach().clone())
        value = video.mean((1, 2, 3, 4)) * self.weight
        return value[:, None, None].expand(-1, 8 * 24 * 24, 1024)


def online_system(**config):
    """A JointSystem whose encoders are stubs but whose wiring is the real one."""
    system = JointSystem.__new__(JointSystem)
    nn.Module.__init__(system)
    from stage2.model.joint_system import OnlineDinoLocal, OnlineJointGlobal

    dino, vjepa = DummyDino(), DummyVjepa()
    local = OnlineDinoLocal.__new__(OnlineDinoLocal)
    nn.Module.__init__(local)
    local.encoder, local.targets, local.batch_size = (
        dino,
        [],
        int(config.get("dino_batch_size", 4)),
    )
    glob = OnlineJointGlobal.__new__(OnlineJointGlobal)
    nn.Module.__init__(glob)
    glob.encoder, glob.targets = vjepa, []
    glob.clip_batch_size = int(config.get("vjepa_clip_batch_size", 1))
    system.local_visual, system.global_visual = local, glob
    system.head = JointStage2Model()
    return system


def _sample_item(length, boxes=None, valid=None):
    if valid is None:
        valid = torch.zeros(length, 12, dtype=torch.bool)
        valid[:, [3, 9]] = True
    if boxes is None:
        boxes = torch.zeros(length, 12, 4)
        boxes[:, 3] = torch.tensor([1.0, 1.0, 4.0, 4.0])
        boxes[:, 9] = torch.tensor([18.0, 18.0, 23.0, 23.0])
    return {
        "rgb": torch.rand(length, 3, 384, 384),
        "roi_boxes": boxes,
        "geometry": torch.randn(length, 12, GEOMETRY_DIM),
        "object_valid": valid,
        "time_valid": torch.ones(length, dtype=torch.bool),
        "local_time": torch.linspace(0, 1, length),
        "frame_seconds": torch.arange(length) / 15,
        "entry_index": min(2, length - 1),
        "collision_index": max(0, length - 2),
        "entry_side": 1,
        "evasion": 0.0,
        "frame_ids": torch.arange(length),
        "track_ids": torch.arange(12),
        "sample_id": f"sample-{length}",
        "source_id": f"source-{length}",
        "flipped": False,
    }


def test_online_dino_roi_scatter_preserves_holes_and_flows_gradient():
    system = online_system()
    batch = joint_collate([_sample_item(2)])
    scene, roi = system.local_visual(
        batch["rgb"], batch["roi_boxes"], batch["object_valid"]
    )
    assert scene.shape == (1, 2, 17, 768)
    assert roi.shape == (1, 2, 12, 768)
    valid = batch["object_valid"]
    assert not roi[~valid].count_nonzero()
    assert roi[0, 0, 9].mean() > roi[0, 0, 3].mean()
    roi.sum().backward()
    assert system.local_visual.encoder.scale.grad is not None


@pytest.fixture
def video(tmp_path):
    """Real JPEG frames plus a schema-2 detector cache, as training consumes them."""
    from PIL import Image

    frames = tmp_path / "frames"
    frames.mkdir()
    geometry = tmp_path / "geometry"
    geometry.mkdir()
    torch.save({"schema": 2}, geometry / "metadata.pt")
    ids = [2 * i + 10 for i in range(24)]
    width, height = 64, 48
    for step, i in enumerate(ids):
        Image.new("RGB", (width, height), (10 + step, 90, 140)).save(
            frames / f"{i}.jpg"
        )
        # One vehicle drifting right, one static; both well inside the frame.
        torch.save(
            {
                "boxes": torch.tensor(
                    [[8.0 + step, 20.0, 20.0 + step, 34.0], [40.0, 18.0, 52.0, 30.0]]
                ),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": ["car", "truck"],
                "size": [width, height],
                "frame_id": i,
                "source_sha256": "x",
            },
            geometry / f"{i}.pt",
        )
    row = dict(
        sample_id="clip",
        source_id="AIHUB:clip",
        frames_dir=str(frames),
        geometry_dir=str(geometry),
        native_fps=15,
        entry_frame=ids[6],
        collision_frame=ids[16],
        entry_side="LEFT",
        evasion_space=1,
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    return row, manifest, ids, width


def _dataset(manifest, **config):
    config.setdefault("tracking", {})
    return JointFeatureDataset(manifest, training=True, seed=7, config=config)


def test_full_video_samples_keep_original_labels(video):
    row, manifest, ids, _ = video
    dataset = _dataset(
        manifest,
        temporal_augmentation={
            "full_video": 1.0,
            "ordinary_crop": 0.0,
            "pre_video_entry": 0.0,
        },
        augmentation={"horizontal_flip": {"probability": 0.0}},
    )
    for epoch in range(5):
        dataset.set_epoch(epoch)
        item = dataset[0]
        assert len(item["time_valid"]) == len(ids)
        assert int(item["frame_ids"][item["entry_index"]]) == row["entry_frame"]
        assert int(item["frame_ids"][item["collision_index"]]) == row["collision_frame"]
        assert item["entry_side"] == 0  # LEFT
        assert item["evasion"] == 1.0


def test_ordinary_crops_contain_both_events_at_varying_offsets(video):
    row, manifest, ids, _ = video
    dataset = _dataset(
        manifest,
        temporal_augmentation={
            "full_video": 0.0,
            "ordinary_crop": 1.0,
            "pre_video_entry": 0.0,
        },
        augmentation={"horizontal_flip": {"probability": 0.0}},
    )
    offsets, spans = set(), set()
    for epoch in range(40):
        dataset.set_epoch(epoch)
        item = dataset[0]
        entry, collision = item["entry_index"], item["collision_index"]
        assert 0 <= entry <= collision < len(item["time_valid"])
        # Crop-relative indices must still point at the real annotated frames.
        assert int(item["frame_ids"][entry]) == row["entry_frame"]
        assert int(item["frame_ids"][collision]) == row["collision_frame"]
        offsets.add(entry)
        spans.add(len(item["time_valid"]))
    assert len(offsets) > 1 and len(spans) > 1


def test_pre_video_entry_relabels_first_frame_and_keeps_side(video):
    row, manifest, ids, _ = video
    dataset = _dataset(
        manifest,
        temporal_augmentation={
            "full_video": 0.0,
            "ordinary_crop": 0.0,
            "pre_video_entry": 1.0,
        },
        augmentation={"horizontal_flip": {"probability": 0.0}},
    )
    seen = 0
    for epoch in range(40):
        dataset.set_epoch(epoch)
        item = dataset[0]
        if int(item["frame_ids"][0]) == ids[0]:
            continue  # fell back to the full video
        seen += 1
        assert item["entry_index"] == 0
        # The crop starts strictly after the real ENTRY, within 0.3 s of it.
        assert int(item["frame_ids"][0]) > row["entry_frame"]
        delta = ids.index(int(item["frame_ids"][0])) - ids.index(row["entry_frame"])
        assert 0 < delta <= math.floor(0.3 * row["native_fps"])
        # COLLISION keeps its real identity at the crop-relative position.
        assert int(item["frame_ids"][item["collision_index"]]) == row["collision_frame"]
        assert item["entry_side"] == 0 and item["evasion"] == 1.0
    assert seen, "pre-video ENTRY crops were never produced"


def test_horizontal_flip_swaps_side_once_and_mirrors_boxes(video):
    row, manifest, ids, width = video
    always = {"horizontal_flip": {"probability": 1.0}}
    never = {"horizontal_flip": {"probability": 0.0}}
    full = {"full_video": 1.0, "ordinary_crop": 0.0, "pre_video_entry": 0.0}
    photometric = {
        name: {"probability": 0.0} for name in augmentation_config(None)["photometric"]
    }
    plain = _dataset(
        manifest,
        temporal_augmentation=full,
        augmentation={**never, "photometric": photometric},
    )[0]
    flipped = _dataset(
        manifest,
        temporal_augmentation=full,
        augmentation={**always, "photometric": photometric},
    )[0]

    assert plain["entry_side"] == 0 and flipped["entry_side"] == 1  # LEFT -> RIGHT
    assert flip_side(flip_side("LEFT")) == "LEFT"  # exactly once, not twice
    # Events and evasion are unaffected by mirroring.
    assert flipped["entry_index"] == plain["entry_index"]
    assert flipped["collision_index"] == plain["collision_index"]
    assert flipped["evasion"] == plain["evasion"]
    # center_x mirrors and dx negates, as a consequence of mirroring the boxes.
    channel = {name: i for i, name in enumerate(GEOMETRY_CHANNELS)}
    both = plain["object_valid"] & flipped["object_valid"]
    cx, fx = (
        plain["geometry"][..., channel["center_x"]],
        flipped["geometry"][..., channel["center_x"]],
    )
    dx, fdx = (
        plain["geometry"][..., channel["dx_per_frame"]],
        flipped["geometry"][..., channel["dx_per_frame"]],
    )
    assert torch.allclose(fx[both], 1 - cx[both], atol=1e-5)
    assert torch.allclose(fdx[both], -dx[both], atol=1e-5)
    # Track identities survive mirroring.
    assert plain["track_ids"].tolist() == flipped["track_ids"].tolist()
    # The pixels really are mirrored.
    assert torch.allclose(
        flipped["rgb"], torch.flip(plain["rgb"], dims=[-1]), atol=1e-5
    )


def test_flip_boxes_uses_the_exact_mirror_formula():
    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    assert flip_boxes(boxes, 100).tolist() == [[70.0, 20.0, 90.0, 40.0]]
    assert torch.allclose(flip_boxes(flip_boxes(boxes, 100), 100), boxes)


def test_crop_local_motion_ignores_observations_before_the_crop(video):
    row, manifest, ids, _ = video
    records = load_detections(row["geometry_dir"], ids)
    channel = {name: i for i, name in enumerate(GEOMETRY_CHANNELS)}
    _, whole, valid_all, _, _ = crop_objects(records, False, {})
    _, cropped, valid_crop, _, _ = crop_objects(records[10:], False, {})
    # Inside the full video frame 10 has motion history; as a crop start it cannot.
    assert valid_all[10].any() and valid_crop[0].any()
    assert whole[10][valid_all[10]][:, channel["dx_per_frame"]].abs().sum() > 0
    assert cropped[0][valid_crop[0]][:, channel["dx_per_frame"]].abs().sum() == 0
    assert (
        cropped[0][valid_crop[0]][:, channel["log_area_growth_per_frame"]].abs().sum()
        == 0
    )
    # Continuity restarts at the crop boundary rather than inheriting a long track.
    assert cropped[0][valid_crop[0]][:, channel["track_continuity"]].max() < (
        whole[10][valid_all[10]][:, channel["track_continuity"]].max()
    )


def test_one_photometric_configuration_is_shared_by_every_frame(video, monkeypatch):
    """Exactly one configuration is drawn per sample and used for every frame."""
    import stage2.data.joint as joint_module

    row, manifest, ids, _ = video
    always = {
        name: dict(section, probability=1.0)
        for name, section in augmentation_config(None)["photometric"].items()
    }
    drawn, applied = [], []
    original = joint_module.sample_photometric

    def spy(config, rng):
        clip = original(config, rng)
        drawn.append(clip)
        wrapped = clip.__class__.__call__

        def record(self, image, generator=None):
            applied.append(self)
            return wrapped(self, image, generator=generator)

        monkeypatch.setattr(clip.__class__, "__call__", record, raising=False)
        return clip

    monkeypatch.setattr(joint_module, "sample_photometric", spy)
    dataset = _dataset(
        manifest,
        temporal_augmentation={
            "full_video": 1.0,
            "ordinary_crop": 0.0,
            "pre_video_entry": 0.0,
        },
        augmentation={"horizontal_flip": {"probability": 0.0}, "photometric": always},
    )
    item = dataset[0]

    assert len(drawn) == 1, "photometric parameters must be sampled once per clip"
    assert len(applied) == len(ids), "every frame must be augmented"
    # Identity, not just equality: one frozen configuration object per clip.
    assert all(clip is drawn[0] for clip in applied)
    assert item["rgb"].shape == (len(ids), 3, 384, 384)
    assert torch.isfinite(item["rgb"]).all()


def test_photometric_parameters_are_sampled_once_per_clip():
    config = augmentation_config(
        {
            "photometric": {
                name: dict(section, probability=1.0)
                for name, section in augmentation_config(None)["photometric"].items()
            }
        }
    )
    rng = np.random.default_rng(3)
    clip = sample_photometric(config, rng)
    assert set(clip.active) >= {"brightness", "contrast", "gamma", "saturation", "jpeg"}
    frames = [torch.rand(3, 24, 24) for _ in range(4)]
    # Disabling noise makes the transform exactly deterministic per pixel value.
    fixed = ClipPhotometric(
        order=clip.order,
        brightness=clip.brightness,
        contrast=clip.contrast,
        gamma=clip.gamma,
        saturation=clip.saturation,
    )
    constant = torch.full((3, 24, 24), 0.5)
    outputs = [fixed(constant) for _ in range(4)]
    for other in outputs[1:]:
        torch.testing.assert_close(outputs[0], other)
    for frame in frames:
        out = fixed(frame)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_noise_strength_is_fixed_per_clip_even_though_pixels_vary():
    clip = ClipPhotometric(order=(), noise_std=0.02)
    base = torch.full((3, 64, 64), 0.5)
    generator = torch.Generator().manual_seed(0)
    a, b = clip(base, generator), clip(base, generator)
    assert not torch.equal(a, b)  # the pixel pattern may differ
    assert abs(float((a - base).std()) - float((b - base).std())) < 0.005
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0


def test_dino_and_vjepa_receive_identical_augmented_frames(video):
    row, manifest, ids, _ = video
    dataset = _dataset(
        manifest,
        temporal_augmentation={
            "full_video": 1.0,
            "ordinary_crop": 0.0,
            "pre_video_entry": 0.0,
        },
    )
    batch = joint_collate([dataset[0]])
    system = online_system()
    system(batch)
    # There is only one image path in the batch: no branch can re-decode its own.
    assert "rgb" in batch and "frame_paths" not in batch

    dino_frames = torch.cat(system.local_visual.encoder.seen)
    assert torch.allclose(dino_frames, batch["rgb"][0], atol=0)
    clips = system.global_visual.encoder.seen
    assert all(tuple(clip.shape[1:]) == (3, 16, 384, 384) for clip in clips)
    # Every V-JEPA frame must be pixel-identical to the DINO frame at that index.
    length = int(batch["time_valid"][0].sum())
    for clip, positions in zip(clips, clip_positions(length)):
        for slot, index in enumerate(positions):
            torch.testing.assert_close(
                clip[0, :, slot], dino_frames[int(index)], atol=0, rtol=0
            )


def test_validation_and_inference_get_no_random_augmentation(video):
    row, manifest, ids, _ = video
    always = {
        "horizontal_flip": {"probability": 1.0},
        "photometric": {
            name: dict(section, probability=1.0)
            for name, section in augmentation_config(None)["photometric"].items()
        },
    }
    evaluation = JointFeatureDataset(
        manifest,
        training=False,
        seed=7,
        config={
            "tracking": {},
            "augmentation": always,
            "temporal_augmentation": {
                "full_video": 0.0,
                "ordinary_crop": 0.0,
                "pre_video_entry": 1.0,
            },
        },
    )
    item = evaluation[0]
    assert item["flipped"] is False
    assert item["entry_side"] == 0  # LEFT survived: no flip was applied
    assert len(item["time_valid"]) == len(ids)  # complete video, never cropped
    assert int(item["frame_ids"][item["entry_index"]]) == row["entry_frame"]


def test_inference_item_needs_no_fps_or_labels(video):
    row, manifest, ids, _ = video
    from stage2.data.cache_geometry import frame_paths

    unlabelled = dict(row)
    for key in (
        "native_fps",
        "entry_frame",
        "collision_frame",
        "entry_side",
        "evasion_space",
    ):
        unlabelled.pop(key)
    paths, frame_ids = frame_paths(unlabelled["frames_dir"])
    records = load_detections(unlabelled["geometry_dir"], frame_ids)
    item = joint_item(unlabelled, paths, frame_ids, records, 0, len(paths))
    assert "frame_seconds" not in item and "entry_index" not in item
    assert "frame_seconds" not in joint_collate([item])


def test_stale_detector_cache_is_rejected(video, tmp_path):
    row, manifest, ids, _ = video
    torch.save({"schema": 1}, Path(row["geometry_dir"]) / "metadata.pt")
    with pytest.raises(ValueError, match="schema 1"):
        load_detections(row["geometry_dir"], ids)


def test_lora_adapts_only_attention_of_the_last_blocks_and_freezes_bases():
    from transformers import Dinov2Config, Dinov2Model

    model = Dinov2Model(
        Dinov2Config(
            hidden_size=24,
            num_attention_heads=3,
            num_hidden_layers=12,
            intermediate_size=48,
        )
    )
    targets = add_lora_to_last_blocks(model, rank=8, alpha=16, dropout=0.05, blocks=4)
    assert len(targets) == 16  # q, k, v and output projection in each of four blocks
    assert all(
        any(f"layer.{i}.attention." in name for i in range(8, 12)) for name in targets
    )
    assert not isinstance(model.encoder.layer[7].attention.attention.query, LoRALinear)
    # Every pretrained weight stays frozen; only A/B adapters train.
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith((".A", ".B")) for n in trainable)
    assert not any(".base." in n for n in trainable)
    model(torch.randn(1, 3, 28, 28)).last_hidden_state.square().mean().backward()
    assert all(model.get_submodule(n).B.grad is not None for n in targets)


def test_frozen_backbones_have_no_gradient_and_head_does(video):
    row, manifest, ids, _ = video
    dataset = _dataset(
        manifest,
        temporal_augmentation={
            "full_video": 1.0,
            "ordinary_crop": 0.0,
            "pre_video_entry": 0.0,
        },
    )
    batch = joint_collate([dataset[0]])
    system = online_system()
    outputs = system(batch)
    joint_loss(outputs, batch)[0].backward()
    assert system.head.global_projection[0].weight.grad.abs().sum() > 0
    assert system.local_visual.encoder.scale.grad is not None
    assert system.global_visual.encoder.weight.grad is not None


def test_rfdetr_is_never_constructed_during_cached_training(video, monkeypatch):
    """Training must read the detector cache; it must not build RF-DETR."""
    import stage2.model.backbones as backbones
    import stage2.model.local_assets as local_assets

    def forbidden(*args, **kwargs):
        raise AssertionError("RF-DETR must not be constructed or run during training")

    monkeypatch.setattr(backbones, "FrozenAdapter", forbidden)
    monkeypatch.setattr(local_assets, "detector", forbidden)
    row, manifest, ids, _ = video
    item = _dataset(manifest)[0]
    assert item["geometry"].shape[-1] == GEOMETRY_DIM
    assert item["object_valid"].any()

    # The cached detections carry no parameters at all, so nothing can require grad.
    records = load_detections(row["geometry_dir"], ids)
    assert all(
        not value.requires_grad
        for record in records
        for value in record.values()
        if isinstance(value, torch.Tensor)
    )


def test_temporal_policy_is_seventy_fifteen_fifteen():
    probabilities = temporal_probabilities(None)
    assert probabilities == {
        "full_video": 0.70,
        "ordinary_crop": 0.15,
        "pre_video_entry": 0.15,
    }
    rng = np.random.default_rng(0)
    counts = {name: 0 for name in probabilities}
    for _ in range(6000):
        *_, mode = choose_crop(160, 60, 100, 15.0, probabilities, rng)
        counts[mode] += 1
    assert 0.66 < counts["full_video"] / 6000 < 0.74
    assert 0.12 < counts["ordinary_crop"] / 6000 < 0.18
    assert 0.12 < counts["pre_video_entry"] / 6000 < 0.18


def test_metrics_use_ordered_decode_seconds_and_interval_confusion():
    outputs = {
        "entry_logits": torch.tensor([[0.0, 2.0, 4.0], [0.0, 4.0, 1.0]]),
        "collision_logits": torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 4.0]]),
        "side_logits": torch.tensor([[3.0, 0.0], [3.0, 0.0]]),
        "evasion_logits": torch.tensor([-1.0, 1.0]),
    }
    batch = {
        "frame_seconds": torch.tensor([[0.0, 0.3, 0.6], [0.0, 1 / 30, 2 / 30]]),
        "entry_index": torch.tensor([1, 0]),
        "collision_index": torch.tensor([1, 2]),
        "entry_side": torch.tensor([0, 1]),
        "evasion": torch.tensor([0.0, 1.0]),
    }
    packet = joint_metric_packet(outputs, batch)
    metric = JointMetricAccumulator()
    metric.update({k: v[:1] for k, v in packet.items()})
    state = metric.state_dict()
    metric = JointMetricAccumulator()
    metric.load_state_dict(state, "cpu")
    metric.update({k: v[1:] for k, v in packet.items()})
    result = metric.compute()
    assert result["acc_entry_0.3s"] == 1
    assert result["acc_collision_0.3s"] == 1
    assert result["f1_entry_side_macro"] == pytest.approx(1 / 3)
    assert result["f1_evasion_space_macro"] == 1
    assert result["competition_score"] == pytest.approx(0.9)


def test_distinct_gaussian_widths_change_only_entry_loss():
    batch = head_batch([_item(19)])
    outputs = JointStage2Model()(batch)
    _, narrow = joint_loss(outputs, batch, entry_sigma_seconds=0.1)
    _, broad = joint_loss(outputs, batch, entry_sigma_seconds=0.2)
    assert not torch.isclose(narrow["loss_entry"], broad["loss_entry"])
    torch.testing.assert_close(narrow["loss_collision"], broad["loss_collision"])


class MetricTestAccelerator:
    num_processes = 1
    optimizer_step_was_skipped = False
    device = torch.device("cpu")

    def __init__(self):
        self.microbatch = 0
        self.logs = []

    @contextmanager
    def accumulate(self, model):
        self.microbatch += 1
        self.sync_gradients = self.microbatch % 2 == 0
        yield

    @contextmanager
    def autocast(self):
        yield

    def backward(self, loss):
        (loss / 2).backward()

    def reduce(self, value, reduction):
        return value

    def gather_for_metrics(self, values):
        return values

    def clip_grad_norm_(self, parameters, maximum):
        torch.nn.utils.clip_grad_norm_(parameters, maximum)

    def log(self, values, step):
        self.logs.append((step, values))


def tiny_joint_trainer(epochs, val_every, checkpoint_metric="loss"):
    from stage2.trainer.trainer import Trainer

    class TinyTrainer(Trainer):
        def _forward(self, batch, reduction="none"):
            losses = self.model(batch).flatten().square()
            return losses, {
                **{
                    k: torch.ones_like(losses)
                    for k in (
                        "loss_entry",
                        "loss_collision",
                        "loss_entry_side",
                        "loss_evasion_space",
                        "loss_invalid_order",
                    )
                },
                "_acc_entry_0.3s": torch.ones_like(losses),
                "_acc_collision_0.3s": torch.ones_like(losses),
                "_confusion_entry_side": torch.tensor([[1, 0, 0, 0]]).expand(
                    len(batch), -1
                ),
                "_confusion_evasion_space": torch.tensor([[1, 0, 0, 0]]).expand(
                    len(batch), -1
                ),
            }

        def save(self, epoch, filename):
            self.saved.append((epoch, filename))

    config = {
        "stage": "joint",
        "optimization": {
            "epochs": epochs,
            "accumulation_steps": 2,
            "grad_clip_norm": 1,
        },
        "logging": {
            "log_every": 10,
            "val_every": val_every,
            "checkpoint_metric": checkpoint_metric,
        },
    }
    trainer = TinyTrainer(MetricTestAccelerator(), config)
    trainer.model = nn.Linear(1, 1)
    # Three groups: DINO LoRA, V-JEPA LoRA, joint head - as the trainer logs them.
    trainer.optimizer = torch.optim.SGD(
        [
            {"params": [next(trainer.model.parameters())], "lr": 2e-5},
            {"params": [trainer.model.bias], "lr": 1e-5},
            {"params": [], "lr": 2e-4},
        ]
    )
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda _: 1
    )
    trainer.train_dataset = []
    trainer.train_loader = [torch.ones(1, 1)] * 12
    trainer.validation_loader = [torch.ones(1, 1)]
    trainer.saved = []
    return trainer


def test_joint_trainer_logging_intervals():
    trainer = tiny_joint_trainer(epochs=2, val_every=2)
    trainer.train_loop()
    losses = [
        (step, values)
        for step, values in trainer.accelerator.logs
        if "train/loss" in values
    ]
    assert [step for step, _ in losses] == [10]
    assert set(losses[0][1]) == {
        "train/loss",
        "train/loss_entry",
        "train/loss_collision",
        "train/loss_entry_side",
        "train/loss_evasion_space",
        "train/loss_invalid_order",
    }
    summaries = [
        (step, values)
        for step, values in trainer.accelerator.logs
        if "train/competition_score" in values
    ]
    assert len(summaries) == 1 and summaries[0][0] == 12
    assert summaries[0][1]["train/competition_score"] == pytest.approx(0.85)
    assert summaries[0][1]["train_config/epoch"] == 2
    assert trainer.joint_train_metrics.count == 0
    assert all(
        step == 12 for step, values in trainer.accelerator.logs if "val/loss" in values
    )


def test_checkpoint_selection_can_maximize_competition_score():
    trainer = tiny_joint_trainer(
        epochs=3, val_every=1, checkpoint_metric="competition_score"
    )
    values = iter([(3, 0.2), (1, 0.1), (2, 0.5)])

    def validate():
        loss, score = next(values)
        trainer.last_validation_metrics = {"competition_score": score}
        return loss

    trainer.validate = validate
    trainer.train_loop()
    assert [(e, name) for e, name in trainer.saved if name == "best.pt"] == [
        (0, "best.pt"),
        (2, "best.pt"),
    ]
    assert trainer.best_competition_score == 0.5
    assert trainer.best_validation_loss == 1


def test_skipped_optimizer_update_does_not_advance_log_cadence():
    trainer = tiny_joint_trainer(epochs=2, val_every=2)
    accelerator = trainer.accelerator
    original_accumulate = accelerator.accumulate
    original_step = trainer.optimizer.step

    @contextmanager
    def accumulate(model):
        with original_accumulate(model):
            accelerator.optimizer_step_was_skipped = accelerator.microbatch == 6
            yield

    def optimizer_step(*args, **kwargs):
        if not accelerator.optimizer_step_was_skipped:
            return original_step(*args, **kwargs)

    accelerator.accumulate = accumulate
    trainer.optimizer.step = optimizer_step
    trainer.train_loop()
    assert trainer.step == 11
    assert [step for step, values in accelerator.logs if "train/loss" in values] == [10]
