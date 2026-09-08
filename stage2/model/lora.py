"""Attention-only low-rank adapters and deployment weight merging."""

from __future__ import annotations
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen linear base plus alpha/r * B(A(dropout(x)))."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout)
        self.A = nn.Parameter(
            base.weight.new_empty(
                rank,
                base.in_features,
            )
        )
        self.B = nn.Parameter(
            base.weight.new_zeros(
                base.out_features,
                rank,
            )
        )
        nn.init.kaiming_uniform_(
            self.A,
            a=math.sqrt(5),
        )
        base.requires_grad_(False)

    def forward(
        self,
        x,
    ):
        return F.linear(
            x,
            self.base.weight,
            self.base.bias,
        ) + self.scale * F.linear(
            F.linear(
                self.drop(x),
                self.A,
            ),
            self.B,
        )

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Materialize an ordinary linear map for dropout-disabled inference."""
        linear = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
        ).to(
            self.base.weight.device,
            self.base.weight.dtype,
        )
        linear.weight.copy_(self.base.weight + self.scale * (self.B @ self.A))
        if self.base.bias is not None:
            linear.bias.copy_(self.base.bias)
        return linear


def _set(
    root: nn.Module,
    path: str,
    value: nn.Module,
):
    parent, name = (
        path.rsplit(
            ".",
            1,
        )
        if "." in path
        else ("", path)
    )
    obj = root.get_submodule(parent) if parent else root
    setattr(
        obj,
        name,
        value,
    )


def add_lora_to_last_blocks(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    blocks=range(
        8,
        12,
    ),
) -> list[str]:
    """Validate complete attention projection sets before adding any adapters.

    A substring such as 'proj' can accidentally adapt MLPs. Restrict matches to
    attention submodules and require QKV plus an output projection in each block.
    """
    targets = []
    required_blocks = set(blocks)
    pattern = re.compile(
        r"(?:^|\.)(?:blocks|layer|layers)\.(\d+)\."
        r"(?:attn|attention)\.(qkv|proj|q_proj|k_proj|v_proj|out_proj)$"
    )
    for name, module in model.named_modules():
        match = pattern.search(name)
        if (
            isinstance(
                module,
                nn.Linear,
            )
            and match
            and int(match[1]) in required_blocks
        ):
            targets.append((name, module, int(match[1]), match[2]))

    for block in required_blocks:
        projections = {leaf for _, _, index, leaf in targets if index == block}
        has_qkv = "qkv" in projections or {"q_proj", "k_proj", "v_proj"} <= projections
        has_output = bool({"proj", "out_proj"} & projections)
        if not (has_qkv and has_output):
            raise ValueError(
                f"Block {block} must expose attention QKV and output projections"
            )

    model.requires_grad_(False)
    attached = []
    for name, module, _, _ in targets:
        _set(
            model,
            name,
            LoRALinear(
                module,
                rank,
                alpha,
                dropout,
            ),
        )
        attached.append(name)
    return attached


def merge_lora(model: nn.Module):
    for name, module in list(model.named_modules()):
        if isinstance(
            module,
            LoRALinear,
        ):
            _set(
                model,
                name,
                module.merge(),
            )
