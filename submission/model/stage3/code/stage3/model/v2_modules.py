"""Optional Stage 3 v2 components. Every module is selected by config; the v1
defaults in ``Stage3MotionModel`` never instantiate anything from this file."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .motion_cnn import ResidualBlock
from .tcn import TCNBlock


# ---------------------------------------------------------------- spatial ---

class MultiScaleMotionEncoder(nn.Module):
    """Hierarchical CNN that keeps spatial motion tokens instead of one vector.

    Stage outputs (48x84, 24x42, 12x21, 6x11 for a 96x168 input) are projected
    to ``token_dim``, pooled onto a shared ``grid`` and summed, so each token
    mixes fine, medium and coarse motion at its location. A global token from
    the coarsest stage carries ego motion.
    """

    def __init__(self, input_channels: int = 10, widths=(32, 64, 96, 128), blocks_per_stage: int = 1,
                 token_dim: int = 128, grid=(4, 7)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(input_channels, widths[0], 3, stride=2, padding=1, bias=False),
                                  nn.GroupNorm(8, widths[0]), nn.SiLU())
        self.stages = nn.ModuleList()
        for stage, width in enumerate(widths):
            layers: list[nn.Module] = []
            if stage:
                layers += [nn.Conv2d(widths[stage - 1], width, 3, stride=2, padding=1, bias=False),
                           nn.GroupNorm(8, width), nn.SiLU()]
            layers += [ResidualBlock(width) for _ in range(blocks_per_stage)]
            self.stages.append(nn.Sequential(*layers))
        self.lateral = nn.ModuleList(nn.Conv2d(w, token_dim, 1) for w in widths)
        self.grid = tuple(grid)
        self.position = nn.Parameter(torch.zeros(1, self.grid[0] * self.grid[1], token_dim))
        self.global_proj = nn.Linear(widths[-1], token_dim)
        self.norm = nn.LayerNorm(token_dim)
        self.token_dim = token_dim
        self.num_tokens = self.grid[0] * self.grid[1] + 1

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = motion.shape
        value = self.stem(motion.reshape(batch * time, channels, height, width))
        tokens = 0
        for stage, lateral in zip(self.stages, self.lateral):
            value = stage(value)
            tokens = tokens + F.adaptive_avg_pool2d(lateral(value), self.grid)
        tokens = tokens.flatten(2).transpose(1, 2) + self.position
        global_token = self.global_proj(value.mean((-2, -1)))[:, None]
        tokens = self.norm(torch.cat((global_token, tokens), 1))
        return tokens.reshape(batch, time, self.num_tokens, self.token_dim)


# ----------------------------------------------------------------- fusion ---

class ConcatFusion(nn.Module):
    """Mean-pool tokens, concatenate the physics embedding, project."""

    def __init__(self, token_dim: int, physics_dim: int, output_dim: int, **_):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(token_dim + physics_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, tokens: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat((tokens.mean(-2), physics), -1))


class GatedFusion(nn.Module):
    """Physics FiLM-modulates every token, then physics-dependent attention pooling."""

    def __init__(self, token_dim: int, physics_dim: int, output_dim: int, **_):
        super().__init__()
        self.film = nn.Linear(physics_dim, 2 * token_dim)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.score = nn.Linear(token_dim, 1)
        self.proj = nn.Sequential(nn.Linear(token_dim + physics_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, tokens: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(physics).unsqueeze(-2).chunk(2, -1)
        modulated = tokens * (1 + scale) + shift
        weights = torch.softmax(self.score(modulated).squeeze(-1), -1)
        pooled = torch.einsum("btn,btnd->btd", weights, modulated)
        return self.proj(torch.cat((pooled, physics), -1))


class CrossAttentionFusion(nn.Module):
    """Physics token queries the spatial motion (and optional visual) tokens."""

    def __init__(self, token_dim: int, physics_dim: int, output_dim: int, heads: int = 4, **_):
        super().__init__()
        self.query = nn.Linear(physics_dim, token_dim)
        self.attention = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(nn.Linear(token_dim, 2 * token_dim), nn.SiLU(), nn.Linear(2 * token_dim, token_dim))
        self.proj = nn.Sequential(nn.Linear(token_dim + physics_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, tokens: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        batch, time, count, dim = tokens.shape
        keys = tokens.reshape(batch * time, count, dim)
        query = self.query(physics).reshape(batch * time, 1, dim)
        attended, _ = self.attention(query, keys, keys, need_weights=False)
        value = self.norm(query + attended + tokens.mean(-2).reshape(batch * time, 1, dim))
        value = (value + self.ffn(value)).reshape(batch, time, dim)
        return self.proj(torch.cat((value, physics), -1))


FUSIONS = {"concat": ConcatFusion, "gated": GatedFusion, "cross_attention": CrossAttentionFusion}


# --------------------------------------------------------------- temporal ---

class _Masked(nn.Module):
    """Replicate each sample's last valid frame before every block (v1 contract)."""

    @staticmethod
    def index(value: torch.Tensor, lengths: torch.Tensor | None):
        if lengths is None:
            return None
        index = torch.arange(value.shape[1], device=value.device)[None, :]
        index = torch.minimum(index, lengths.to(value.device, torch.long)[:, None] - 1)
        return index[..., None].expand_as(value)


class DualDilatedBlock(nn.Module):
    """Two dilation streams (d, 2d) merged by a learned gate, residual."""

    def __init__(self, dim: int, dilations: tuple[int, int], dropout: float, kernel_size: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.convs = nn.ModuleList(nn.Conv1d(dim, dim, kernel_size, dilation=d) for d in dilations)
        self.dilations, self.kernel_size = dilations, kernel_size
        self.gate = nn.Conv1d(2 * dim, dim, 1)
        self.pointwise = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x = self.norm(value).transpose(1, 2)
        streams = []
        for conv, d in zip(self.convs, self.dilations):
            span = (self.kernel_size - 1) * d
            streams.append(F.silu(conv(F.pad(x, (span // 2, span - span // 2), mode="replicate"))))
        g = torch.sigmoid(self.gate(torch.cat(streams, 1)))
        merged = g * streams[0] + (1 - g) * streams[1]
        return value + self.dropout(self.pointwise(self.dropout(merged))).transpose(1, 2)


class DualDilatedTCN(_Masked):
    def __init__(self, dim: int, pairs=((1, 2), (2, 4), (4, 8), (8, 16), (16, 32)), dropout: float = 0.1, kernel_size: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList(DualDilatedBlock(dim, tuple(p), dropout, kernel_size) for p in pairs)
        self.receptive_field = 1 + (kernel_size - 1) * sum(max(p) for p in pairs)

    def forward(self, value, lengths=None):
        index = self.index(value, lengths)
        for block in self.blocks:
            value = block(value if index is None else value.gather(1, index))
        return value if index is None else value.gather(1, index)


class S4DKernel(nn.Module):
    """Diagonal state-space convolution kernel (S4D-Lin), computed in closed form."""

    def __init__(self, dim: int, state: int = 32, dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        log_dt = torch.rand(dim) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.log_a_real = nn.Parameter(torch.log(0.5 * torch.ones(dim, state // 2)))
        self.a_imag = nn.Parameter(math.pi * torch.arange(state // 2).float().repeat(dim, 1))
        self.c = nn.Parameter(torch.randn(dim, state // 2, dtype=torch.cfloat) * 0.5 ** 0.5)

    def forward(self, length: int) -> torch.Tensor:
        dt = self.log_dt.float().exp()[:, None]
        a = -self.log_a_real.float().exp() + 1j * self.a_imag.float()
        dta = a * dt
        c = self.c * (torch.exp(dta) - 1.0) / a
        steps = torch.arange(length, device=a.device)
        kernel = torch.einsum("dn,dnl->dl", c, torch.exp(dta[..., None] * steps)).real
        return 2 * kernel


class BiSSMBlock(nn.Module):
    """Pre-norm residual block: forward and reversed S4D convolutions + GLU mixing."""

    def __init__(self, dim: int, state: int = 32, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.forward_kernel = S4DKernel(dim, state)
        self.backward_kernel = S4DKernel(dim, state)
        self.skip = nn.Parameter(torch.ones(dim))
        self.out = nn.Linear(2 * dim, 2 * dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _conv(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        n = 2 * length
        return torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(kernel, n), n)[..., :length]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x = self.norm(value).transpose(1, 2).float()
        length = x.shape[-1]
        forward = self._conv(x, self.forward_kernel(length))
        backward = self._conv(x.flip(-1), self.backward_kernel(length)).flip(-1)
        mixed = torch.cat((forward, backward), 1).transpose(1, 2) + torch.cat((x, x), 1).transpose(1, 2) * self.skip.repeat(2)
        mixed = F.glu(self.out(self.dropout(F.gelu(mixed.to(value.dtype)))), -1)
        return value + self.dropout(mixed)


class BiSSM(_Masked):
    def __init__(self, dim: int, layers: int = 4, state: int = 32, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(BiSSMBlock(dim, state, dropout) for _ in range(layers))

    def forward(self, value, lengths=None):
        index = self.index(value, lengths)
        for block in self.blocks:
            value = block(value if index is None else value.gather(1, index))
        return value if index is None else value.gather(1, index)


class GatedTemporalHybrid(nn.Module):
    """g * local + (1 - g) * global, gate from both streams."""

    def __init__(self, dim: int, local: nn.Module, global_: nn.Module):
        super().__init__()
        self.local, self.global_ = local, global_
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, value, lengths=None):
        local, global_ = self.local(value, lengths), self.global_(value, lengths)
        g = torch.sigmoid(self.gate(torch.cat((local, global_), -1)))
        return g * local + (1 - g) * global_


# ------------------------------------------------------------- refinement ---

class RefinementStage(nn.Module):
    """Temporal features + stage-1 predictions -> residual temporal blocks."""

    PRED_DIM = 5  # accel s1, accel s2, speed, stop probability, steering angle

    def __init__(self, dim: int, blocks: int = 4, dropout: float = 0.1, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim + self.PRED_DIM, dim), nn.LayerNorm(dim))
        self.blocks = nn.ModuleList(TCNBlock(dim, dilations[i % len(dilations)], dropout) for i in range(blocks))

    @staticmethod
    def prediction_features(output: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat((output["acceleration"], output["speed"][..., None] / 10.0,
                          torch.sigmoid(output["stop_logit"])[..., None],
                          output["steering_angle"][..., None] / 30.0), -1)

    def forward(self, features: torch.Tensor, output: dict[str, torch.Tensor], lengths=None) -> torch.Tensor:
        value = self.proj(torch.cat((features, self.prediction_features(output).to(features.dtype)), -1))
        index = _Masked.index(value, lengths)
        for block in self.blocks:
            value = block(value if index is None else value.gather(1, index))
        return value if index is None else value.gather(1, index)
