"""Joint architecture, data, gradient-flow and metric regression tests."""

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from stage2.data.cache_joint_features import extract_local
from stage2.data.joint import (
    JointFeatureDataset,
    joint_collate,
    joint_item,
    load_joint_cache,
)
from stage2.data.joint_sampling import clip_positions
from stage2.model.joint import JointStage2Model, HybridTemporalBlock
from stage2.model.joint_system import JointSystem
from stage2.model.joint_tracking import object_tensors, ego_lane_score, percentile_rank
from stage2.model.tracking import Detection, Track
from stage2.utils.joint_losses import constrained_decode, joint_loss
from stage2.utils.joint_metrics import joint_metric_packet, JointMetricAccumulator


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
        "geometry": torch.randn(length, 12, 13),
        "object_valid": torch.rand(length, 12) > 0.3,
        "time_valid": torch.ones(length, dtype=torch.bool),
        "local_time": torch.linspace(0, 1, length),
        "frame_seconds": torch.arange(length) / 15,
        "entry_index": min(2, length - 1),
        "collision_index": max(0, length - 2),
        "entry_side": 1,
        "evasion": 0.0,
        "frame_ids": torch.arange(length),
        "frame_paths": [str(i) for i in range(length)],
        "sample_id": f"sample-{length}",
        "source_id": f"source-{length}",
    }


def head_batch(items):
    batch = joint_collate(items)
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


def test_hybrid_block_locality_and_empty_rows():
    block = HybridTemporalBlock(radius=16).eval()
    x = torch.randn(2, 65, 384)
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
    return Detection(np.array([x, 40, x + 20, 80], dtype=np.float32), score, "car", 2.0)


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
    def forward_features(self, images):
        batch = len(images)
        dense = (
            torch.arange(576, dtype=torch.float32)
            .reshape(1, 576, 1)
            .expand(batch, -1, 768)
        )
        return {"x_norm_clstoken": torch.zeros(batch, 768), "x_norm_patchtokens": dense}


def test_roi_scatter_preserves_holes(monkeypatch):
    monkeypatch.setattr(
        "stage2.data.cache_joint_features.read_image",
        lambda *a, **k: torch.zeros(3, 100, 100, dtype=torch.uint8),
    )
    boxes = torch.zeros(1, 12, 4)
    boxes[0, 3] = torch.tensor([10, 10, 30, 30])
    boxes[0, 9] = torch.tensor([60, 60, 80, 80])
    valid = torch.zeros(1, 12, dtype=torch.bool)
    valid[0, [3, 9]] = True
    scene, roi = extract_local(DummyDino(), ["frame"], boxes, valid, "cpu", 1)
    assert scene.shape == (1, 17, 768)
    assert not roi[~valid].count_nonzero()
    assert roi[0, 3].mean() > 0
    assert roi[0, 9].mean() > roi[0, 3].mean()


@pytest.fixture
def cached_video(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    ids = [2 * i + 10 for i in range(20)]
    for i in ids:
        (frames / f"{i}.jpg").touch()
    item = _item(20)
    cache = {
        k: item[k]
        for k in ("scene_features", "roi_features", "geometry", "object_valid")
    }
    cache.update(
        schema=2,
        sample_id="clip",
        frame_ids=torch.tensor(ids),
        track_ids=torch.arange(12),
    )
    torch.save(cache, tmp_path / "clip.pt")
    row = dict(
        sample_id="clip",
        source_id="source",
        frames_dir=str(frames),
        native_fps=15,
        entry_frame=ids[6],
        collision_frame=ids[12],
        entry_side="LEFT",
        evasion_space=1,
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    return row, cache, manifest, tmp_path


def test_random_crops_keep_both_events_and_align_rgb(cached_video):
    row, cache, manifest, directory = cached_video
    dataset = JointFeatureDataset(manifest, directory, training=True, seed=123)
    spans = set()
    for epoch in range(20):
        dataset.set_epoch(epoch)
        item = dataset[0]
        assert item["frame_ids"][item["entry_index"]] == row["entry_frame"]
        assert item["frame_ids"][item["collision_index"]] == row["collision_frame"]
        assert [int(Path(path).stem) for path in item["frame_paths"]] == item[
            "frame_ids"
        ].tolist()
        spans.add((int(item["frame_ids"][0]), int(item["frame_ids"][-1])))
        torch.testing.assert_close(item["scene_features"], dataset[0]["scene_features"])
    assert len(spans) > 1
    assert len(JointFeatureDataset(manifest, directory)[0]["time_valid"]) == 20


def test_inference_needs_no_fps_or_labels_and_rejects_old_cache(cached_video):
    row, cache, _, directory = cached_video
    for key in (
        "native_fps",
        "entry_frame",
        "collision_frame",
        "entry_side",
        "evasion_space",
    ):
        row.pop(key)
    loaded, paths = load_joint_cache(row, directory)
    item = joint_item(row, loaded, paths)
    assert "frame_seconds" not in item
    assert "frame_seconds" not in joint_collate([item])
    cache["schema"] = 1
    torch.save(cache, directory / "clip.pt")
    with pytest.raises(ValueError, match="schema-2"):
        load_joint_cache(row, directory)


class DummyVjepa(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, video):
        assert not self.training and not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        self.calls.append(tuple(video.shape))
        value = video.mean((1, 2, 3, 4)) * self.weight
        return value[:, None, None].expand(-1, 8 * 24 * 24, 1024)


def test_online_vjepa_freeze_sampling_and_projection_backward(monkeypatch):
    seen = []

    def read(path, **kwargs):
        seen.append(int(path))
        return torch.full((3, 8, 8), int(path), dtype=torch.uint8)

    monkeypatch.setattr("stage2.model.joint_system.read_image", read)
    encoder = DummyVjepa()
    model = JointSystem({}, encoder=encoder).train()
    batch = joint_collate([_item(65), _item(9)])
    outputs = model(batch)
    joint_loss(outputs, batch)[0].backward()
    assert encoder.weight.grad is None and not encoder.weight.requires_grad
    assert not encoder.training
    assert all(shape[1:] == (3, 16, 384, 384) for shape in encoder.calls)
    assert model.head.global_projection[0].weight.grad.abs().sum() > 0
    assert set(seen).issubset(set(range(65)))
    assert len(encoder.calls) == 3


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
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
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
