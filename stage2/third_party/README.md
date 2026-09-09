# Required local model integrations

Workspace integrations are available in `model/local_assets.py`; see
[WORKSPACE.md](../WORKSPACE.md) for the pinned official implementation and GPU
smoke workflow. No pretrained implementation is vendored here. Factories configured as
`package.module:callable` must construct a `torch.nn.Module` without downloading
weights. Parameter names must match the local checkpoint exactly: loading is
strict, and wrappers changing prefixes need explicit checkpoint conversion.

- V-JEPA 2.1 ViT-B/16: input `[B,3,32,384,384]`, output dense
  `[B,16,24,24,768]`. Flattened tokens or channel-first dense are also supported.
  Configure the encoder checkpoint key (default `ema_encoder`).
- DINOv2 ViT-S/14: input `[B,3,336,336]`; official `forward_features` with
  `x_norm_clstoken` and `x_norm_patchtokens` is supported, as is a dictionary with
  `global [B,384]` and `dense [B,24,24,384]`.
- Visual blocks 8–11 must expose `attn`/`attention` projections named `qkv` or
  `q_proj/k_proj/v_proj`, plus `proj`/`out_proj`. Unsupported names fail explicitly.
- RF-DETR Small: accept a list of native RGB CHW uint8 tensors; return one
  dictionary per frame with native xyxy `boxes`, `scores`, and canonical string
  `labels`. The wrapper owns preprocessing, class mapping, and postprocessing.
  Retained classes are car, bus, truck, and motorcycle.
- Depth Anything V2 Small: same original-image input; return one finite native
  `[H,W]` map per frame. The wrapper owns preprocessing and output resizing.
  Check orientation before setting `depth_closer_is_larger`.

Resolve RF-DETR/Transformers compatibility in the final local environment.
Verify all four models with actual checkpoints and original-resolution images
offline before claiming the full training/submission integration is complete.
