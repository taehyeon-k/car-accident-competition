"""DINOv3 backbone with lightweight geometry heads.

The backbone is the asset; heads are deliberately small. Dense heads read the
final-norm patch tokens (the tensor Stage 2 consumes as ``x_norm_patchtokens``)
or, optionally, a 1x1 fusion of several blocks.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from stage2.geometry_pretrain.common import LABEL_STRIDE, OBJECT_CLASSES

PATCH = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_dinov3(arch: str, checkpoint_path: str | None, source: str = "/workspace/pretrained/dinov3-source"):
    if source not in sys.path:
        sys.path.insert(0, source)
    from dinov3.hub import backbones

    model = getattr(backbones, f"dinov3_{arch}")(pretrained=False)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "backbone" in state and isinstance(state["backbone"], dict):
            state = state["backbone"]
        model.load_state_dict(state, strict=True)
    return model


class DinoBackbone(nn.Module):
    """Official DINOv3 ViT with only the last ``trainable_blocks`` blocks trainable.

    Frozen leading blocks run without autograd. RoPE coordinate augmentation is
    disabled permanently so train/eval features are identical in geometry.
    """

    def __init__(
        self,
        arch: str = "vits16",
        checkpoint_path: str | None = None,
        source: str = "/workspace/pretrained/dinov3-source",
        trainable_blocks: int = 0,
        out_blocks: tuple[int, ...] | None = None,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.model = build_dinov3(arch, checkpoint_path, source)
        self.depth = len(self.model.blocks)
        self.embed_dim = self.model.embed_dim
        self.n_prefix = 1 + self.model.n_storage_tokens
        self.out_blocks = tuple(sorted({(b % self.depth) for b in (out_blocks or (-1,))}))
        self.grad_checkpointing = grad_checkpointing
        self.set_trainable_blocks(trainable_blocks)

    def set_trainable_blocks(self, n: int):
        self.trainable_blocks = int(n)
        self.model.requires_grad_(False)
        if n > 0:
            for blk in self.model.blocks[self.depth - n :]:
                blk.requires_grad_(True)
            self.model.norm.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.model.rope_embed is not None:
            self.model.rope_embed.eval()
        return self

    def forward(self, x: torch.Tensor) -> dict:
        B, _, H, W = x.shape
        h, w = H // PATCH, W // PATCH
        first_trainable = self.depth - self.trainable_blocks
        grad = torch.is_grad_enabled()
        outs = {}
        with torch.set_grad_enabled(grad and first_trainable == 0):
            tokens, (hh, ww) = self.model.prepare_tokens_with_masks(x)
            rope = self.model.rope_embed(H=hh, W=ww) if self.model.rope_embed is not None else None
        trunk = tokens if first_trainable == 0 else None
        for i, blk in enumerate(self.model.blocks):
            if i == first_trainable and i > 0:
                trunk = tokens
            if i < first_trainable:
                with torch.no_grad():
                    tokens = blk(tokens, rope)
            else:
                if self.grad_checkpointing and grad and self.training:
                    tokens = checkpoint(blk, tokens, rope, use_reentrant=False)
                else:
                    tokens = blk(tokens, rope)
            if i in self.out_blocks:
                outs[i] = tokens
        feats = []
        for i in self.out_blocks:
            normed = self.model.norm(outs[i])
            feats.append(normed[:, self.n_prefix :].transpose(1, 2).reshape(B, -1, h, w))
        final = self.model.norm(tokens)
        return {
            "patch": final[:, self.n_prefix :].transpose(1, 2).reshape(B, -1, h, w),
            "cls": final[:, 0],
            "layers": feats,
            "trunk": None if trunk is None else trunk.detach(),
            "rope": rope,
            "hw": (h, w),
        }


# ----------------------------------------------------------------------------
# Heads
# ----------------------------------------------------------------------------


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.GELU())


class DenseHead(nn.Module):
    """1/16 patch map -> 1/4 map: 1x1 reduce, then two (conv, x2 upsample) stages."""

    def __init__(self, cin: int, cout: int, mid: int = 128):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(cin, mid, 1), nn.GroupNorm(8, mid), nn.GELU())
        self.s16 = conv_block(mid, mid)
        self.s8 = conv_block(mid, mid)
        self.s4 = conv_block(mid, mid // 2)
        self.out = nn.Conv2d(mid // 2, cout, 1)

    def forward(self, f):
        x = self.s16(self.reduce(f))
        x = self.s8(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.s4(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.out(x)


class CameraHead(nn.Module):
    """Attention-pooled vanishing-point regressor (normalized x, y)."""

    def __init__(self, cin: int, mid: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(cin, mid, 1)
        self.attn = nn.Conv2d(mid, 1, 1)
        self.mlp = nn.Sequential(nn.Linear(mid + 2, mid), nn.GELU(), nn.Linear(mid, 2))

    def forward(self, f):
        B, _, h, w = f.shape
        x = F.gelu(self.proj(f))
        a = self.attn(x).flatten(2).softmax(-1)  # B,1,hw
        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, h, device=f.device), torch.linspace(0, 1, w, device=f.device), indexing="ij"
        )
        coords = torch.stack([xs, ys], 0).flatten(1).to(a.dtype)  # 2,hw
        pooled = torch.einsum("bcn,bn->bc", x.flatten(2), a[:, 0])
        where = torch.einsum("cn,bn->bc", coords, a[:, 0])  # attention centroid
        return where + self.mlp(torch.cat([pooled, where], 1))


class FlowHead(nn.Module):
    """Global soft-argmax correspondence on DINO features + a small refiner.

    Matching on (projected) backbone features forces the patch tokens themselves
    to become correspondence-aware; the refiner only adds sub-patch detail.
    Output flow is in label-resolution (stride-4) pixels.
    """

    def __init__(self, cin: int, dim: int = 128, mid: int = 96):
        super().__init__()
        self.proj = nn.Conv2d(cin, dim, 1)
        self.log_temp = nn.Parameter(torch.tensor(math.log(0.05)))
        self.refine = nn.Sequential(conv_block(dim * 2 + 4, mid), conv_block(mid, mid))
        self.up = nn.Sequential(conv_block(mid, mid // 2), nn.Conv2d(mid // 2, 2, 3, padding=1))
        nn.init.zeros_(self.up[-1].weight)
        nn.init.zeros_(self.up[-1].bias)

    def forward(self, f1, f2):
        B, _, h, w = f1.shape
        a = F.normalize(self.proj(f1).float(), dim=1)
        b = F.normalize(self.proj(f2).float(), dim=1)
        corr = torch.einsum("bcn,bcm->bnm", a.flatten(2), b.flatten(2)) / self.log_temp.exp().clamp_min(1e-3)
        prob = corr.softmax(-1)  # B, hw(src), hw(tgt)
        ys, xs = torch.meshgrid(
            torch.arange(h, device=f1.device, dtype=torch.float32),
            torch.arange(w, device=f1.device, dtype=torch.float32),
            indexing="ij",
        )
        grid = torch.stack([xs, ys], -1).reshape(-1, 2)  # hw,2 (patch units)
        target = prob @ grid  # B,hw,2
        coarse = (target - grid[None]).transpose(1, 2).reshape(B, 2, h, w)  # patch units
        conf = prob.max(-1).values.reshape(B, 1, h, w)
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(-1).reshape(B, 1, h, w) / math.log(h * w)
        # Warp projected target features to the source grid using the soft match.
        warped = torch.einsum("bnm,bcm->bcn", prob, b.flatten(2)).reshape(B, -1, h, w)
        x = self.refine(torch.cat([a, warped, coarse, conf, entropy], 1).to(f1.dtype))
        scale = PATCH // LABEL_STRIDE
        x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
        coarse_up = F.interpolate(coarse, scale_factor=scale, mode="bilinear", align_corners=False) * scale
        return {"flow": coarse_up + self.up(x).float(), "coarse": coarse_up}


TASK_CHANNELS = {"road": 5, "depth": 1, "objects": len(OBJECT_CLASSES), "contact": 1}


class OriginalTail(nn.Module):
    """Frozen copy of the original last blocks + norm, fed with the shared frozen trunk.

    Because blocks before the trainable tail are frozen and identical in the
    adapted model, the original DINOv3 output equals OriginalTail(trunk).
    """

    def __init__(self, backbone: DinoBackbone):
        super().__init__()
        import copy

        n = backbone.trainable_blocks
        self.blocks = nn.ModuleList([copy.deepcopy(b) for b in backbone.model.blocks[backbone.depth - n :]])
        self.norm = copy.deepcopy(backbone.model.norm)
        self.n_prefix = backbone.n_prefix
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, out: dict) -> torch.Tensor:
        t = out["trunk"]
        for blk in self.blocks:
            t = blk(t, out["rope"])
        t = self.norm(t)
        B = t.shape[0]
        h, w = out["hw"]
        return t[:, self.n_prefix :].transpose(1, 2).reshape(B, -1, h, w)


class GeometryDINO(nn.Module):
    def __init__(self, backbone: DinoBackbone, tasks: list[str], head_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        C = backbone.embed_dim
        n_layers = len(backbone.out_blocks)
        self.fuse = nn.Conv2d(C * n_layers, C, 1) if n_layers > 1 else None
        self.tasks = list(tasks)
        self.heads = nn.ModuleDict()
        for t in self.tasks:
            if t in TASK_CHANNELS:
                self.heads[t] = DenseHead(C, TASK_CHANNELS[t], head_dim)
            elif t == "camera":
                self.heads[t] = CameraHead(C, head_dim)
            elif t == "flow":
                self.heads[t] = FlowHead(C, head_dim)
            else:
                raise ValueError(f"Unknown task {t}")

    def features(self, images):
        out = self.backbone(images)
        f = out["patch"] if self.fuse is None else self.fuse(torch.cat(out["layers"], 1))
        return f, out

    def forward_static(self, images, tasks=None):
        f, out = self.features(images)
        preds = {"_patch": out["patch"], "_backbone_out": out}
        for t in tasks or self.tasks:
            if t in ("flow",) or t not in self.heads:
                continue
            preds[t] = self.heads[t](f)
        return preds

    def forward_pair(self, img_t, img_tp):
        f, _ = self.features(torch.cat([img_t, img_tp], 0))
        f1, f2 = f.chunk(2, 0)
        return self.heads["flow"](f1, f2)

    def head_parameters(self):
        params = list(self.heads.parameters())
        if self.fuse is not None:
            params += list(self.fuse.parameters())
        return params

    def backbone_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.backbone.model.state_dict().items()}


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
