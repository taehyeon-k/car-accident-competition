"""Regression coverage for strict local checkpoint loading."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file

from stage2.model.backbones import load_local


def test_safetensors_preserves_real_backbone_prefix():
    model = nn.Module()
    model.backbone = nn.Linear(3, 2)
    state = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "model.safetensors"
        save_file(state, str(checkpoint))
        with patch(
            "stage2.model.backbones.resolve_factory", return_value=lambda: model
        ):
            loaded = load_local("test:factory", str(checkpoint))
    assert all(torch.equal(v, state[k]) for k, v in loaded.state_dict().items())
    assert not any(p.requires_grad for p in loaded.parameters())
