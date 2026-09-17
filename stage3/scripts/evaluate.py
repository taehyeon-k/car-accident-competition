from __future__ import annotations

import argparse

from accelerate import Accelerator

from stage3.trainer.trainer import Trainer
from stage3.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Stage 3 checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    trainer = Trainer(Accelerator(mixed_precision=cfg["optimization"]["mixed_precision"]), cfg)
    trainer.build()
    trainer.resume(args.checkpoint)
    print(trainer.validate())


if __name__ == "__main__":
    main()
