from torch import nn


class PhysicsMLP(nn.Sequential):
    def __init__(self, input_dim: int = 20, output_dim: int = 32, dropout: float = 0.05):
        super().__init__(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(64, output_dim),
        )
