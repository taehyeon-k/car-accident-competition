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
