"""Export a trusted training checkpoint with merged visual LoRA weights."""

import argparse

import torch

from stage2.model.lora import merge_lora
from stage2.model.pipeline import CoarseSystem, FineSystem
from stage2.utils.utils import atomic_save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    arguments = parser.parse_args()
    checkpoint = torch.load(
        arguments.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    if (
        config["model"].get(
            "training_mode",
            "lora",
        )
        != "lora"
    ):
        raise ValueError(
            "A head-only ablation cannot be exported as the fixed architecture"
        )
    model = (
        CoarseSystem(config["model"])
        if config["stage"] == "coarse"
        else FineSystem(config["model"])
    )
    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )
    model.eval()
    merge_lora(model.visual)
    # Auxiliary state heads are training-only. Keep their removal explicit.
    weights = {
        name: value.cpu()
        for name, value in model.state_dict().items()
        if "state_head" not in name
    }
    atomic_save(
        {"format_version": 2, "merged_lora": True, "config": config, "model": weights},
        arguments.output,
    )


if __name__ == "__main__":
    main()
