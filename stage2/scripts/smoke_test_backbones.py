"""Optional local pretrained-backbone checks; not required for CPU logic tests."""

import argparse

import torch

from stage2.model.backbones import VJEPAAdapter, DINOAdapter
from stage2.utils.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    values = config["model"]
    if config["stage"] == "coarse":
        model = VJEPAAdapter(
            values["vjepa_factory"],
            values["vjepa_checkpoint"],
            checkpoint_key=values.get(
                "vjepa_checkpoint_key",
                "ema_encoder",
            ),
        )
        image = torch.zeros(
            1,
            3,
            32,
            384,
            384,
            device=arguments.device,
        )
    else:
        model = DINOAdapter(
            values["dino_factory"],
            values["dino_checkpoint"],
        )
        image = torch.zeros(
            1,
            3,
            336,
            336,
            device=arguments.device,
        )
    model.to(arguments.device).eval()
    with torch.no_grad():
        model(image)
    print("Local visual backbone output contract passed")


if __name__ == "__main__":
    main()
