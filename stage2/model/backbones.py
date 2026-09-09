"""Strict local-only adapters. Factories are project-owned Python callables, never Hub downloads."""

from __future__ import annotations
import argparse
import importlib
from pathlib import Path
import torch
import torch.nn as nn
from .lora import add_lora_to_last_blocks


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


class VJEPAAdapter(nn.Module):
    """Adapt a local V-JEPA encoder while retaining dense tubelet tokens."""

    def __init__(
        self,
        factory,
        checkpoint,
        rank=8,
        alpha=16,
        dropout=0.05,
        checkpoint_key="ema_encoder",
    ):
        super().__init__()
        self.encoder = load_local(
            factory,
            checkpoint,
            checkpoint_key=checkpoint_key,
        )
        self.targets = add_lora_to_last_blocks(
            self.encoder,
            rank,
            alpha,
            dropout,
        )

    def forward(
        self,
        x,
    ):
        out = self.encoder(x)
        out = (
            out["dense"]
            if isinstance(
                out,
                dict,
            )
            else out
        )
        if out.ndim == 3 and tuple(out.shape[1:]) == (16 * 24 * 24, 768):
            out = out.reshape(
                out.shape[0],
                16,
                24,
                24,
                768,
            )
        if out.ndim == 5 and out.shape[1] == 768:
            out = out.permute(
                0,
                2,
                3,
                4,
                1,
            )
        if tuple(out.shape[1:]) != (16, 24, 24, 768):
            raise RuntimeError(
                f"V-JEPA returned {tuple(out.shape)}; expected [B,16,24,24,768]"
            )
        return out


class DINOAdapter(nn.Module):
    """Use the official forward_features interface when available."""

    def __init__(
        self,
        factory,
        checkpoint,
        rank=8,
        alpha=16,
        dropout=0.05,
    ):
        super().__init__()
        self.encoder = load_local(
            factory,
            checkpoint,
        )
        self.targets = add_lora_to_last_blocks(
            self.encoder,
            rank,
            alpha,
            dropout,
        )

    def forward(
        self,
        x,
    ):
        out = (
            self.encoder.forward_features(x)
            if hasattr(
                self.encoder,
                "forward_features",
            )
            else self.encoder(x)
        )
        if (
            isinstance(
                out,
                dict,
            )
            and "x_norm_clstoken" in out
        ):
            out = {
                "global": out["x_norm_clstoken"],
                "dense": out["x_norm_patchtokens"].reshape(
                    x.shape[0],
                    24,
                    24,
                    384,
                ),
            }
        global_token, dense = (
            (out["global"], out["dense"])
            if isinstance(
                out,
                dict,
            )
            else out
        )
        if tuple(global_token.shape[1:]) != (384,) or tuple(dense.shape[1:]) != (
            24,
            24,
            384,
        ):
            raise RuntimeError(
                "DINO factory must return global [B,384] and dense [B,24,24,384]"
            )
        return global_token, dense


class FrozenAdapter(nn.Module):
    """Keep detector/depth inference frozen even when a parent calls train()."""

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
