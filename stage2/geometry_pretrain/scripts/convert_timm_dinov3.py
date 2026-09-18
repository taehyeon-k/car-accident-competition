"""Convert timm's public DINOv3 ViT-S/16 LVD-1689M weights to the official layout.

Meta's signed download URL for ``dinov3_vits16_pretrain_lvd1689m-08c60483.pth``
expired, and the ``facebook/`` Hugging Face repository is gated. timm re-hosts
the identical LVD-1689M tensors. The official checkpoints store all-zero qkv
biases (verified on the local ViT-B/16), so timm's bias-free variant is exact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

RENAMES = {"reg_token": "storage_tokens", ".gamma_1": ".ls1.gamma", ".gamma_2": ".ls2.gamma"}


def convert(timm_state: dict, source: Path) -> dict:
    sys.path.insert(0, str(source))
    from dinov3.hub.backbones import dinov3_vits16

    model = dinov3_vits16(pretrained=False)
    model.init_weights()
    expected = model.state_dict()
    out = {}
    for name, value in timm_state.items():
        for old, new in RENAMES.items():
            name = name.replace(old, new)
        out[name] = value
    for name, value in expected.items():
        if name in out:
            continue
        if name.endswith("attn.qkv.bias"):
            out[name] = torch.zeros_like(value)
        elif name.endswith("bias_mask") or name == "rope_embed.periods":
            out[name] = value.clone()
        elif name == "mask_token":
            out[name] = torch.zeros_like(value)  # used only for iBOT masking in pretraining
        else:
            raise KeyError(f"Unmapped official tensor {name}")
    model.load_state_dict(out, strict=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timm", required=True)
    parser.add_argument("--dinov3-source", default="/workspace/pretrained/dinov3-source")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    state = convert(load_file(args.timm), Path(args.dinov3_source))
    torch.save(state, args.output)
    print(f"saved {len(state)} tensors to {args.output}")


if __name__ == "__main__":
    main()
