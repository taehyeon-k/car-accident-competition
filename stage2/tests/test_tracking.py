"""Tracking configuration and aggregation tests: no W&B account or network."""

import tempfile
import unittest
from unittest.mock import Mock

import torch

from stage2.utils.tracking import (
    TrainingMetrics,
    initialize_tracking,
    tracker_backend,
)


class TrackingTests(unittest.TestCase):
    def test_disabled_does_not_initialize(self):
        accelerator = Mock()
        self.assertIsNone(tracker_backend({}))
        initialize_tracking(
            accelerator,
            {},
        )
        accelerator.init_trackers.assert_not_called()

    def test_offline_initialization_excludes_local_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "stage": "coarse",
                "seed": 42,
                "output_dir": directory,
                "optimization": {"epochs": 2},
                "tracking": {},
                "model": {
                    "vjepa_checkpoint": "/private/weights.pt",
                    "vjepa_factory": "private.module:factory",
                    "lora_rank": 8,
                },
                "data": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "manifest": "/private/data.jsonl",
                },
                "logging": {
                    "wandb": {
                        "enabled": True,
                        "project": "test-project",
                        "mode": "offline",
                    }
                },
            }
            accelerator = Mock()
            initialize_tracking(
                accelerator,
                config,
            )
            args, kwargs = accelerator.init_trackers.call_args
            self.assertEqual(
                args,
                ("test-project",),
            )
            self.assertEqual(
                kwargs["init_kwargs"]["wandb"]["mode"],
                "offline",
            )
            self.assertNotIn(
                "resume",
                kwargs["init_kwargs"]["wandb"],
            )
            self.assertNotIn(
                "/private",
                str(kwargs["config"]),
            )

            settings = config["logging"]["wandb"]
            settings.update(
                mode="online",
                run_id="abc123",
                resume="must",
            )
            initialize_tracking(
                accelerator,
                config,
            )
            options = accelerator.init_trackers.call_args.kwargs["init_kwargs"]["wandb"]
            self.assertEqual(
                options["id"],
                "abc123",
            )
            self.assertEqual(
                options["resume"],
                "must",
            )

    def test_resume_requires_online_and_explicit_id(self):
        settings = {
            "enabled": True,
            "project": "test",
            "mode": "online",
            "resume": "must",
        }
        config = {"logging": {"wandb": settings}}
        with self.assertRaises(ValueError):
            tracker_backend(config)
        settings.update(
            run_id="abc",
            mode="offline",
        )
        with self.assertRaises(ValueError):
            tracker_backend(config)

    def test_metrics_are_sample_weighted_and_detached(self):
        metrics = TrainingMetrics()
        metrics.update(
            torch.tensor(
                [1.0, 3.0],
                requires_grad=True,
            ),
            {},
        )
        metrics.update(
            torch.tensor(
                [8.0],
                requires_grad=True,
            ),
            {},
        )
        self.assertFalse(metrics.sums["loss"].requires_grad)
        accelerator = Mock()
        accelerator.reduce.side_effect = lambda values, reduction: values * 2
        self.assertEqual(
            metrics.flush(accelerator),
            {"train/loss": 4.0},
        )
        self.assertEqual(
            metrics.flush(accelerator),
            {},
        )


if __name__ == "__main__":
    unittest.main()
