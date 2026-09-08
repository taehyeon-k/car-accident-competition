"""Epoch-boundary checkpoints including per-rank RNG and precision state."""

import random

import numpy as np
import torch
import torch.distributed as distributed

from stage2.utils.utils import atomic_save


def capture_rng() -> dict:
    """Collect RNG states without changing any generator."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def rank_rng_states() -> list[dict]:
    """All ranks call this before the main rank writes a checkpoint."""
    state = capture_rng()
    if not distributed.is_initialized():
        return [state]
    states = [None] * distributed.get_world_size()
    distributed.all_gather_object(
        states,
        state,
    )
    return states


def save_checkpoint(
    value: dict,
    path,
) -> None:
    atomic_save(
        {"format_version": 2, **value},
        path,
    )


def load_inference_system(
    path,
    expected_stage: str,
):
    """Load either a training checkpoint or the merged inference export.

    Factories and original local weights are still needed to construct encoders.
    Only explicitly removed auxiliary heads may be absent in an exported state.
    """
    from stage2.model.pipeline import CoarseSystem, FineSystem
    from stage2.model.lora import merge_lora

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    if checkpoint.get("format_version") != 2 or config["stage"] != expected_stage:
        raise ValueError("Incompatible checkpoint format or stage")
    if (
        config["model"].get(
            "training_mode",
            "lora",
        )
        != "lora"
    ):
        raise ValueError("Submission requires trained LoRA backbones")
    constructor = CoarseSystem if expected_stage == "coarse" else FineSystem
    system = constructor(config["model"]).eval()
    if checkpoint.get(
        "merged_lora",
        False,
    ):
        merge_lora(system.visual)
        missing, unexpected = system.load_state_dict(
            checkpoint["model"],
            strict=False,
        )
        permitted_missing = (
            {
                "head.entry_state_head.weight",
                "head.entry_state_head.bias",
                "head.collision_state_head.weight",
                "head.collision_state_head.bias",
            }
            if expected_stage == "fine"
            else set()
        )
        if set(missing) != permitted_missing or unexpected:
            raise ValueError(
                f"Unexpected export keys: missing={missing}, unexpected={unexpected}"
            )
    else:
        system.load_state_dict(
            checkpoint["model"],
            strict=True,
        )
        merge_lora(system.visual)
    return system, config
