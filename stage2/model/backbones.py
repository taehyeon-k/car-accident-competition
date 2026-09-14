"""Strict local-only adapters. Factories are project-owned Python callables, never Hub downloads."""

from __future__ import annotations
import argparse
import importlib
from pathlib import Path
import torch
import torch.nn as nn


def resolve_factory(path: str):
    if not path or ":" not in path:
        raise ValueError("Factory must be 'package.module:callable'")
    module, name = path.split(
        ":",
        1,
    )
    return getattr(
        importlib.import_module(module),
        name,
    )


def load_local(
    factory_path: str,
    checkpoint: str,
    *,
    checkpoint_key: str | None = None,
    **kwargs,
) -> nn.Module:
    """Load every base tensor; partial initialization is not pretrained loading."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Required local checkpoint is missing: {path}")
    factory = resolve_factory(factory_path)
    model = factory(**kwargs)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        with torch.serialization.safe_globals([argparse.Namespace]):
            state = torch.load(path, map_location="cpu", weights_only=True)
    if factory_path == "stage2.model.local_assets:detector":
        state = state["model"]
    if checkpoint_key is not None:
        state = state[checkpoint_key]
    elif (
        isinstance(
            state,
            dict,
        )
        and "state_dict" in state
    ):
        state = state["state_dict"]
    expected = model.state_dict()
    state = {
        (
            name
            if name in expected or factory_path == "stage2.model.local_assets:detector"
            else name.removeprefix("module.").removeprefix("backbone.")
        ): value
        for name, value in state.items()
    }
    model.load_state_dict(
        state,
        strict=True,
    )
    model.requires_grad_(False)
    return model


class FrozenAdapter(nn.Module):
    """Keep frozen detector inference frozen even when a parent calls train()."""

    def __init__(
        self,
        factory,
        checkpoint,
    ):
        super().__init__()
        self.model = load_local(
            factory,
            checkpoint,
        )
        self.model.eval()

    def train(
        self,
        mode: bool = True,
    ):
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, *args, **kwargs):
        return self.model(
            *args,
            **kwargs,
        )
