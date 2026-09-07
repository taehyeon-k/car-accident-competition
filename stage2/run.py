"""Training entry point. Run from the ``stage2`` directory."""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from trainer.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Stage 2 localization model")
    parser.add_argument("--config", required=True, help="Path to coarse.yaml or fine.yaml")
    parser.add_argument("--resume", help="Optional full-training checkpoint to resume")
    arguments = parser.parse_args()

    with open(arguments.config, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    # Keep Python, NumPy, and PyTorch sampling reproducible for a given run seed.
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    optimization = config["optimization"]
    accelerator = Accelerator(
        gradient_accumulation_steps=optimization["accumulation_steps"],
        mixed_precision=optimization["mixed_precision"],
        project_config=ProjectConfiguration(project_dir=config["output_dir"]),
    )

    trainer = Trainer(accelerator, config)
    trainer.build()
    trainer.train_loop(arguments.resume or config.get("resume"))


if __name__ == "__main__":
    main()
