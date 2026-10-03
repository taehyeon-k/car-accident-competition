from __future__ import annotations

import copy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).requires_grad_(False).eval()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            value.copy_(value * self.decay + source[name].detach() * (1 - self.decay))

    def state_dict(self):
        return self.model.state_dict()
