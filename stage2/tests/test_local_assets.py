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


def test_deeper_backbone_unfreezes_only_final_four_blocks():
    from transformers import Dinov2Config, Dinov2Model
    from stage2.model.lora import unfreeze_last_blocks

    model = Dinov2Model(
        Dinov2Config(
            hidden_size=24,
            num_attention_heads=3,
            num_hidden_layers=24,
            intermediate_size=48,
        )
    )
    targets = add_lora_to_last_blocks(model, rank=16)
    unfreeze_last_blocks(model, 4)
    assert all(any(f"layer.{i}." in n for i in range(20, 24)) for n in targets)
    assert not any(
        p.requires_grad
        for block in model.encoder.layer[:20]
        for p in block.parameters()
    )
    assert all(
        p.requires_grad
        for block in model.encoder.layer[20:]
        for p in block.parameters()
    )
    model(torch.randn(1, 3, 28, 28)).last_hidden_state.square().mean().backward()
    assert (
        model.encoder.layer[20].attention.attention.query.base.weight.grad is not None
    )
    assert model.encoder.layer[20].mlp.fc1.weight.grad is not None
    assert model.encoder.layer[19].mlp.fc1.weight.grad is None


def test_larger_feature_heads_forward_and_backward():
    from stage2.model.model import CoarseModel, FineModel

    coarse = CoarseModel(feature_dim=1024, num_frames=64)
    dense = torch.randn(1, 32, 24, 24, 1024, requires_grad=True)
    coarse(
        dense,
        torch.zeros(1, 32, 12, 4),
        torch.zeros(1, 32, 12, 9),
        torch.zeros(1, 32, 12, dtype=torch.bool),
        torch.ones(1, 64, dtype=torch.bool),
    )["entry_logits"].sum().backward()
    assert dense.grad is not None
    fine = FineModel(feature_dim=768)
    global_tokens = torch.randn(1, 2, 768, requires_grad=True)
    fine(
        global_tokens,
        torch.zeros(1, 2, 24, 24, 768),
        torch.zeros(1, 2, 12, 4),
        torch.zeros(1, 2, 12, 9),
        torch.zeros(1, 2, 12, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
    )["entry_logits"].sum().backward()
    assert global_tokens.grad is not None
    assert fine.global_projection.weight.grad is not None


def test_64_bin_sampling_geometry_and_region_recovery():
    import numpy as np
    from stage2.data.sampling import build_coarse_bins, recover_region, event_bin
    from stage2.model.geometry import tubelet_geometry

    bins = build_coarse_bins(list(range(100)), list(range(100)), False, num_bins=64)
    assert len(bins.valid) == 64
    assert event_bin(99, bins) == 63
    assert recover_region(bins, 63)[-1] == 99
    short = build_coarse_bins(list(range(5)), list(range(5)), False, num_bins=64)
    assert short.valid.sum() == 5
    frame = torch.randn(1, 64, 12, 9)
    geometry, valid = tubelet_geometry(frame, torch.ones(1, 64, 12, dtype=torch.bool))
    assert geometry.shape == (1, 32, 12, 9)
    assert valid.shape == (1, 32, 12)
    np.testing.assert_allclose(geometry[0, 0, :, :7], frame[0, :2, :, :7].mean(0))
