"""Train only the new actor-scene ENTRY branch on a fixed E4_sa trunk."""
from __future__ import annotations

import os

import torch

from stage2.robust_goal_20260927.actor_interaction import ActorInteractionPyramid


class FrozenActorInteraction(ActorInteractionPyramid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        source = os.environ["ACTOR_BASELINE_CKPT"]
        state = torch.load(source, map_location="cpu", weights_only=False)["model"]
        own = self.state_dict()
        incompatible = [(key, tuple(value.shape), tuple(own[key].shape)) for key, value in state.items()
                        if key not in own or value.shape != own[key].shape]
        if incompatible:
            raise RuntimeError(f"Baseline checkpoint incompatible: {incompatible[:5]}")
        self.load_state_dict(state, strict=False)
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith("actor_"))

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for name, module in self.named_children():
                if not name.startswith("actor_"):
                    module.eval()
        return self


def main():
    import stage2.aux_signal_experiments.train as trainer
    trainer.AuxPyramid = FrozenActorInteraction
    trainer.main()


if __name__ == "__main__":
    main()
