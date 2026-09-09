"""Regression coverage for strict loading and Hugging Face attention naming."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file

from stage2.model.backbones import load_local
from stage2.model.lora import add_lora_to_last_blocks, LoRALinear


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


def test_hf_lora_only_last_four_complete_attention_sets():
    from transformers import Dinov2Config, Dinov2Model

    model = Dinov2Model(
        Dinov2Config(
            hidden_size=24,
            num_attention_heads=3,
            num_hidden_layers=12,
            intermediate_size=48,
        )
    )
    targets = add_lora_to_last_blocks(model)
    assert len(targets) == 16
    assert all(
        any(f"layer.{i}.attention." in name for i in range(8, 12)) for name in targets
    )
    assert not isinstance(model.encoder.layer[7].attention.attention.query, LoRALinear)
    assert isinstance(model.encoder.layer[8].attention.output.dense, LoRALinear)
    output = model(torch.randn(1, 3, 28, 28)).last_hidden_state
    output.square().mean().backward()
    assert all(model.get_submodule(n).B.grad is not None for n in targets)
