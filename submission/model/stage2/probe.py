import torch
from torch import nn
import torch.nn.functional as F
GRID = (7, 10)

class TemporalProbe(nn.Module):
    """Deliberately small: token projection, flatten, 2 dilated temporal convs."""

    def __init__(self, n_tokens=GRID[0] * GRID[1], dim=384, tok=32, hidden=192, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.tok = nn.Linear(dim, tok)
        self.frame = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_tokens * tok, hidden), nn.GELU())
        self.t1 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.t2 = nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2)
        self.drop = nn.Dropout(dropout)
        self.event = nn.Linear(hidden, 2)
        self.attn = nn.Linear(hidden, 1)
        self.side = nn.Linear(hidden, 2)
        self.evasion = nn.Linear(hidden, 1)

    def forward(self, x, valid):
        B, T, N, C = x.shape
        h = self.tok(self.norm(x.float())).reshape(B, T, -1)
        h = self.frame(h).transpose(1, 2)  # B,H,T
        h = h + F.gelu(self.t1(h))
        h = h + F.gelu(self.t2(self.drop(h)))
        h = h.transpose(1, 2)  # B,T,H
        ev = self.event(self.drop(h))
        neg = torch.finfo(ev.dtype).min / 4
        entry = ev[..., 0].masked_fill(~valid, neg)
        collision = ev[..., 1].masked_fill(~valid, neg)
        a = self.attn(h)[..., 0].masked_fill(~valid, neg).softmax(-1)
        pooled = torch.einsum("bt,bth->bh", a, h)
        return {
            "entry_logits": entry,
            "collision_logits": collision,
            "side_logits": self.side(self.drop(pooled)),
            "evasion_logits": self.evasion(self.drop(pooled))[:, 0],
        }

