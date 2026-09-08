"""CPU regression tests for real failures found in the architecture review.

Run with Python's built-in runner: python -m unittest stage2.tests.test_review -v
No pretrained weights, GPU, network, pytest or Accelerate installation is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torchvision.io import write_png

from stage2.data.cache_geometry import compact_observations
from stage2.data.dataset import NativeDataset, collate
from stage2.data.geometry_stats import fit_statistics
from stage2.data.preparation import build_window_geometry
from stage2.data.sampling import build_coarse_bins, event_bin, sample_fine_window
from stage2.data.transforms import box_to_grid, letterbox, ClipPhotometric
from stage2.model.backbones import load_local, FrozenAdapter
from stage2.model.geometry import build_geometry
from stage2.model.inference import Stage2Pipeline
from stage2.model.lora import LoRALinear, add_lora_to_last_blocks, merge_lora
from stage2.model.model import FineModel
from stage2.model.modules import Transformer, sinusoidal
from stage2.model.pipeline import FineSystem
from stage2.model.tracking import Detection, Track, HungarianTracker
from stage2.utils.losses import fine_loss
from stage2.utils.utils import atomic_save
from stage2.trainer.trainer import Trainer

TRACKING = {
    "max_gap": 2,
    "max_center_distance": 0.2,
    "max_match_cost": 0.65,
    "detection_threshold": 0.2,
    "loom_clip": float(np.log(2)),
}


class CountingDINO(nn.Module):
    """Tiny differentiable adapter to measure frames actually sent to DINO."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.seen = 0

    def forward(
        self,
        images,
    ):
        self.seen += len(images)
        signal = images.mean(dim=(1, 2, 3)) * self.weight
        global_features = signal[:, None].expand(
            -1,
            384,
        )
        dense = signal[:, None, None, None].expand(
            -1,
            24,
            24,
            384,
        )
        return global_features, dense


class FakeCoarse(nn.Module):
    def forward(
        self,
        batch,
    ):
        logits = torch.full(
            (1, 32),
            -100.0,
        )
        logits[:, 16] = 1.0
        return {
            "entry_logits": logits,
            "collision_logits": logits,
            "direction_logits": torch.tensor([[1.0, 0.0]]),
            "evasion_logits": torch.tensor([[1.0]]),
        }


class FakeFineHead(nn.Module):
    def forward(
        self,
        global_tokens,
        dense,
        boxes,
        geometry,
        object_valid,
        time_valid,
    ):
        scores = torch.arange(
            time_valid.shape[1],
            dtype=torch.float32,
        )[None]
        scores = scores.masked_fill(
            ~time_valid,
            -torch.inf,
        )
        return {"entry_logits": scores, "collision_logits": scores}


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_short_bins_and_fine_sampling_preserve_original_events(self):
        for length in (1, 2, 7, 31, 32, 33, 700):
            ids = [5 + 7 * index for index in range(length)]
            bins = build_coarse_bins(
                list(range(length)),
                ids,
                training=False,
            )
            self.assertEqual(
                int(bins.valid.sum()),
                min(
                    length,
                    32,
                ),
            )
            for event in (0, length - 1):
                self.assertTrue(
                    bins.valid[
                        event_bin(
                            event,
                            bins,
                        )
                    ]
                )
                for seed in range(5):
                    window = sample_fine_window(
                        ids,
                        event,
                        np.random.default_rng(seed),
                    )
                    self.assertIn(
                        event,
                        window,
                    )
                    self.assertTrue(np.all(np.diff(window) == 1))

    def test_masked_loss_has_finite_gradients(self):
        logits = torch.randn(
            2,
            64,
            requires_grad=True,
        )
        states = torch.randn(
            2,
            64,
            requires_grad=True,
        )
        valid = torch.arange(64)[None] < torch.tensor([[1], [7]])
        outputs = {
            "entry_logits": logits.masked_fill(
                ~valid,
                -torch.inf,
            ),
            "collision_logits": logits.masked_fill(
                ~valid,
                -torch.inf,
            ),
            "entry_state_logits": states,
            "collision_state_logits": states,
        }
        loss, _ = fine_loss(
            outputs,
            {
                "event_type": torch.tensor([0, 1]),
                "time_valid": valid,
                "event_local_index": torch.tensor([0, 6]),
            },
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(
            float(logits.grad[~valid].abs().sum()),
            0.0,
        )

    def test_all_masked_transformer_rows_are_zero(self):
        model = Transformer(2).eval()
        states = torch.randn(
            2,
            5,
            384,
            requires_grad=True,
        )
        valid = torch.tensor([[True, True, False, False, False], [False] * 5])
        contaminated = states.masked_fill(
            ~valid[..., None],
            float("nan"),
        )
        output = model(
            contaminated,
            valid,
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(
            float(output[~valid].detach().abs().sum()),
            0.0,
        )
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(states.grad).all())

    def test_fine_padding_does_not_change_real_frame_predictions(self):
        model = FineModel().eval()
        global_tokens = torch.randn(
            1,
            3,
            384,
        )
        dense = torch.randn(
            1,
            3,
            24,
            24,
            384,
        )
        geometry = torch.randn(
            1,
            3,
            12,
            9,
        )
        boxes = torch.zeros(
            1,
            3,
            12,
            4,
        )
        objects = torch.zeros(
            1,
            3,
            12,
            dtype=torch.bool,
        )
        valid = torch.ones(
            1,
            3,
            dtype=torch.bool,
        )
        with torch.no_grad():
            short = model(
                global_tokens,
                dense,
                boxes,
                geometry,
                objects,
                valid,
            )

            def padded(tensor):
                extra = tensor.new_full(
                    (1, 61, *tensor.shape[2:]),
                    float("nan"),
                )
                return torch.cat(
                    (tensor, extra),
                    dim=1,
                )

            time_mask = torch.arange(64)[None] < 3
            full = model(
                padded(global_tokens),
                padded(dense),
                padded(boxes),
                padded(geometry),
                torch.zeros(
                    1,
                    64,
                    12,
                    dtype=torch.bool,
                ),
                time_mask,
            )
        torch.testing.assert_close(
            short["entry_logits"],
            full["entry_logits"][:, :3],
            atol=2e-6,
            rtol=2e-5,
        )

    def test_coarse_positions_always_use_fixed_denominator(self):
        valid = torch.zeros(
            1,
            16,
            dtype=torch.bool,
        )
        valid[:, -1] = True
        position = sinusoidal(
            valid,
            coarse=True,
        )
        self.assertAlmostEqual(
            float(position[0, -1, 0]),
            float(np.sin(1)),
            places=6,
        )

    def test_roi_align_under_cpu_bfloat16_autocast(self):
        from stage2.model.model import roi_appearance

        dense = torch.randn(
            1,
            1,
            24,
            24,
            384,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        boxes = torch.tensor([[[[1.0, 1.0, 10.0, 10.0]]]])
        valid = torch.ones(
            1,
            1,
            1,
            dtype=torch.bool,
        )
        with torch.autocast(
            device_type="cpu",
            dtype=torch.bfloat16,
        ):
            features = roi_appearance(
                dense,
                boxes,
                valid,
            )
            loss = features.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(dense.grad).all())

    def test_coarse_union_expands_total_width_by_fifteen_percent(self):
        records = []
        for _ in range(32):
            records.append(
                {
                    "size": [100, 100],
                    "boxes": torch.tensor([[20.0, 20.0, 40.0, 40.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": ["car"],
                    "proximity": torch.tensor([0.0]),
                }
            )
        item = build_window_geometry(
            records,
            np.ones(
                32,
                dtype=bool,
            ),
            TRACKING,
            coarse=True,
        )
        expected = torch.tensor([18.5, 18.5, 41.5, 41.5]) * (384 / 100 / 16)
        torch.testing.assert_close(
            item["boxes_grid"][0, 0],
            expected,
        )

    def test_tracker_rejects_nonvehicles_and_class_changes(self):
        frame = [
            Detection(
                np.array([2.0, 2.0, 8.0, 8.0]),
                0.9,
                "person",
            ),
            Detection(
                np.array([2.0, 2.0, 8.0, 8.0]),
                0.1,
                "car",
            ),
        ]
        self.assertEqual(
            HungarianTracker().track(
                [frame],
                [(10, 10)],
            ),
            [],
        )
        frames = [
            [
                Detection(
                    np.array([2.0, 2.0, 8.0, 8.0]),
                    0.9,
                    label,
                )
            ]
            for label in ("car", "truck")
        ]
        self.assertEqual(
            len(
                HungarianTracker().track(
                    frames,
                    [(10, 10)] * 2,
                )
            ),
            2,
        )

    def test_accumulation_matches_actual_sample_mean_including_final_group(self):
        class CPUAccelerator:
            """Model just the documented accumulation contract, not Accelerate internals."""

            num_processes = 1
            optimizer_step_was_skipped = False
            sync_gradients = False

            def __init__(self):
                self.microbatch = 0

            @contextmanager
            def accumulate(
                self,
                model,
            ):
                self.microbatch += 1
                self.sync_gradients = self.microbatch in (2, 3)
                yield

            @contextmanager
            def autocast(self):
                yield

            def backward(
                self,
                loss,
            ):
                (loss / 2).backward()

            def reduce(
                self,
                value,
                reduction,
            ):
                return value

            def clip_grad_norm_(
                self,
                parameters,
                maximum,
            ):
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    maximum,
                )

            def end_training(self):
                pass

        class TinyTrainer(Trainer):
            def _forward(
                self,
                batch,
                reduction="mean",
            ):
                loss = (self.model(batch) - 1).square().flatten()
                return loss, {}

            def save(self, *args):
                pass

        config = {
            "stage": "coarse",
            "optimization": {
                "epochs": 1,
                "accumulation_steps": 2,
                "grad_clip_norm": 1000,
            },
            "logging": {"log_every": 100, "val_every": 100},
        }
        trainer = TinyTrainer(
            CPUAccelerator(),
            config,
        )
        trainer.model = nn.Linear(
            1,
            1,
            bias=False,
        )
        trainer.model.weight.data.fill_(0.3)
        trainer.train_dataset = []
        trainer.train_loader = [
            torch.tensor([[1.0], [2.0]]),
            torch.tensor([[3.0]]),
            torch.tensor([[4.0], [5.0], [6.0]]),
        ]
        trainer.optimizer = torch.optim.SGD(
            trainer.model.parameters(),
            lr=0.01,
        )
        trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer,
            lambda _: 1.0,
        )
        reference = nn.Linear(
            1,
            1,
            bias=False,
        )
        reference.weight.data.fill_(0.3)
        optimizer = torch.optim.SGD(
            reference.parameters(),
            lr=0.01,
        )
        for group in (
            torch.tensor([[1.0], [2.0], [3.0]]),
            torch.tensor([[4.0], [5.0], [6.0]]),
        ):
            (reference(group) - 1).square().mean().backward()
            optimizer.step()
            optimizer.zero_grad()
        trainer.train_loop()
        torch.testing.assert_close(
            trainer.model.weight,
            reference.weight,
        )
        self.assertEqual(
            trainer.step,
            2,
        )

    def test_lora_only_adapts_attention_and_merges_equivalently(self):
        encoder = nn.Module()
        encoder.blocks = nn.ModuleList()
        for _ in range(12):
            block = nn.Module()
            block.attn = nn.ModuleDict(
                {
                    "qkv": nn.Linear(
                        4,
                        12,
                    ),
                    "proj": nn.Linear(
                        4,
                        4,
                    ),
                }
            )
            block.mlp = nn.ModuleDict(
                {
                    "proj": nn.Linear(
                        4,
                        4,
                    )
                }
            )
            encoder.blocks.append(block)
        names = add_lora_to_last_blocks(encoder)
        self.assertEqual(
            len(names),
            8,
        )
        self.assertFalse(any("mlp" in name for name in names))
        layer = encoder.blocks[8].attn["proj"]
        layer.B.data.normal_()
        encoder.eval()
        inputs = torch.randn(
            3,
            4,
        )
        expected = layer(inputs)
        merge_lora(encoder)
        torch.testing.assert_close(
            expected,
            encoder.blocks[8].attn["proj"](inputs),
        )
        self.assertFalse(
            any(
                isinstance(
                    module,
                    LoRALinear,
                )
                for module in encoder.modules()
            )
        )

    def test_checkpoint_missing_base_weights_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.pt"
            torch.save(
                {"bias": torch.zeros(4)},
                path,
            )
            with patch(
                "stage2.model.backbones.resolve_factory",
                return_value=lambda: nn.Linear(
                    4,
                    4,
                ),
            ):
                with self.assertRaises(RuntimeError):
                    load_local(
                        "local:model",
                        str(path),
                    )

    def test_live_fine_skips_padded_frames_and_keeps_gradients(self):
        with patch(
            "stage2.model.pipeline.DINOAdapter",
            side_effect=lambda *args: CountingDINO(),
        ):
            system = FineSystem(
                {
                    "dino_factory": "mock",
                    "dino_checkpoint": "mock",
                    "lora_rank": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "frame_batch_size": 2,
                }
            )
        images = torch.randn(
            1,
            64,
            3,
            2,
            2,
        )
        valid = torch.arange(64)[None] < 3
        output = system(
            {
                "fine_rgb": images,
                "time_valid": valid,
                "boxes_grid": torch.zeros(
                    1,
                    64,
                    12,
                    4,
                ),
                "geometry": torch.zeros(
                    1,
                    64,
                    12,
                    9,
                ),
                "object_valid": torch.zeros(
                    1,
                    64,
                    12,
                    dtype=torch.bool,
                ),
            }
        )
        output["entry_logits"][:, :3].sum().backward()
        self.assertEqual(
            system.visual.seen,
            3,
        )
        self.assertIsNotNone(system.visual.weight.grad)
        self.assertTrue(torch.isfinite(system.visual.weight.grad))

    def test_exact_resize_scales_and_integer_boxes(self):
        image = torch.full(
            (3, 101, 203),
            128,
            dtype=torch.uint8,
        )
        normalized, transform = letterbox(
            image,
            336,
        )
        boxes = torch.tensor([[0, 0, 203, 101]])
        converted = box_to_grid(
            boxes,
            transform,
            14,
        )
        self.assertEqual(
            converted.dtype,
            torch.float32,
        )
        self.assertAlmostEqual(
            float(converted[0, 2]),
            24.0,
        )
        self.assertAlmostEqual(
            float(converted[0, 3] * 14),
            round(101 * transform.scale) + transform.pad_y,
            places=4,
        )
        self.assertEqual(
            float(normalized[:, 0].abs().sum()),
            0.0,
        )

    def test_cached_depth_features_match_native_depth_path(self):
        boxes = torch.tensor([[2.0, 2.0, 10.0, 12.0], [12.0, 2.0, 20.0, 12.0]])
        depth = (
            torch.arange(24)
            .expand(
                16,
                24,
            )
            .float()
        )
        detected = {"boxes": boxes, "scores": [0.9, 0.8], "labels": ["car", "car"]}
        cache = compact_observations(
            detected,
            depth,
            24,
            16,
            0.2,
            True,
        )
        raw_tracks = [
            Track(
                index,
                "car",
                {
                    0: Detection(
                        box.numpy(),
                        0.9,
                        "car",
                    )
                },
                0,
            )
            for index, box in enumerate(boxes)
        ]
        cached_tracks = [
            Track(
                index,
                "car",
                {
                    0: Detection(
                        box.numpy(),
                        0.9,
                        "car",
                        float(cache["proximity"][index]),
                    )
                },
                0,
            )
            for index, box in enumerate(boxes)
        ]
        raw, _ = build_geometry(
            raw_tracks,
            [depth.numpy()],
            [(24, 16)],
            1,
        )
        cached, _ = build_geometry(
            cached_tracks,
            None,
            [(24, 16)],
            1,
        )
        np.testing.assert_allclose(
            raw,
            cached,
            atol=1e-6,
        )

    def test_native_dataset_and_inference_use_same_original_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb_dir = root / "rgb"
            cache_dir = root / "geometry"
            rgb_dir.mkdir()
            ids = [10 + 7 * index for index in range(40)]
            for original_id in ids:
                write_png(
                    torch.full(
                        (3, 16, 24),
                        128,
                        dtype=torch.uint8,
                    ),
                    str(rgb_dir / f"frame_{original_id:06}.png"),
                )
                atomic_save(
                    {
                        "frame_id": original_id,
                        "size": [24, 16],
                        "boxes": torch.tensor([[2.0, 2.0, 10.0, 12.0]]),
                        "scores": torch.tensor([0.9]),
                        "labels": ["car"],
                        "proximity": torch.tensor([0.5]),
                    },
                    cache_dir / f"{original_id}.pt",
                )
            row = {
                "sample_id": "sample",
                "source_id": "source",
                "frames_dir": str(rgb_dir),
                "geometry_dir": str(cache_dir),
                "entry_frame": ids[10],
                "collision_frame": ids[20],
                "entry_side": "LEFT",
                "evasion_space": 1,
            }
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            coarse = NativeDataset(
                str(manifest),
                "coarse",
                TRACKING,
                training=False,
            )
            item = coarse[0]
            self.assertEqual(
                item["coarse_rgb"].shape,
                (3, 32, 384, 384),
            )
            self.assertEqual(
                item["geometry"].shape,
                (16, 12, 9),
            )
            self.assertEqual(
                collate([item])["entry_bin"].dtype,
                torch.int64,
            )
            fine = NativeDataset(
                str(manifest),
                "fine",
                TRACKING,
                training=True,
            )
            self.assertEqual(
                len(fine),
                2,
            )
            event = fine[1]
            self.assertEqual(
                int(event["representative_frame_id"][event["event_local_index"]]),
                row["collision_frame"],
            )
            statistics = fit_statistics(
                [item],
                "coarse",
            )
            self.assertEqual(
                statistics["split"],
                "train",
            )

            fine_system = nn.Module()
            fine_system.visual = CountingDINO()
            fine_system.head = FakeFineHead()
            fine_system.frame_batch_size = 2
            pipeline = Stage2Pipeline(
                FakeCoarse(),
                fine_system,
                {"tracking": TRACKING},
            )
            predicted = pipeline.predict(row)
            self.assertIn(
                predicted["entry_frame"],
                ids,
            )
            self.assertIn(
                predicted["collision_frame"],
                ids,
            )
            self.assertEqual(
                predicted["entry_side"],
                "LEFT",
            )
            self.assertEqual(
                predicted["evasion_space"],
                1,
            )
            # Both events use the same region. Each native frame is encoded once.
            bins = build_coarse_bins(
                list(range(len(ids))),
                ids,
                False,
            )
            from stage2.data.sampling import recover_region

            self.assertEqual(
                fine_system.visual.seen,
                len(
                    recover_region(
                        bins,
                        16,
                    )
                ),
            )


def load_tests(
    loader,
    tests,
    pattern,
):
    # Also execute the original function-style tests without requiring pytest.
    from stage2.tests import test_contracts

    for name in dir(test_contracts):
        if name.startswith("test_"):
            tests.addTest(
                unittest.FunctionTestCase(
                    getattr(
                        test_contracts,
                        name,
                    )
                )
            )
    return tests


if __name__ == "__main__":
    unittest.main()
