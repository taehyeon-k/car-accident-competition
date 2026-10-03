"""Train the Stage 3 acceleration-centric motion TCN."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
    from stage3.trainer.trainer import Trainer
    from stage3.utils.config import load_config, seed_everything
    from stage3.utils.tracking import initialize_tracking, tracker_backend

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    opt = cfg["optimization"]
    accelerator = Accelerator(
        gradient_accumulation_steps=opt["accumulation_steps"], mixed_precision=opt["mixed_precision"],
        log_with=tracker_backend(cfg), project_config=ProjectConfiguration(project_dir=cfg["output_dir"]),
    )
    trainer = Trainer(accelerator, cfg)
    trainer.build()
    try:
        initialize_tracking(accelerator, cfg)
        trainer.train_loop(args.resume or cfg.get("resume"))
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()
