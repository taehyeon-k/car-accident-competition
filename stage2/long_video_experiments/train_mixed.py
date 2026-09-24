"""Isolated ENTRY-exact / COLLISION-soft ablation for the 349-video head."""
from __future__ import annotations

import torch

from stage2.spotting_experiments import train as trainer
from stage2.spotting_experiments.objective import distribution_loss, spotting_loss as original_loss


def mixed_loss(outputs, batch, cfg):
    # The baseline trainer still owns data loading, optimizer, checkpoint selection,
    # metrics, and output paths. Only the COLLISION term changes in this process.
    if cfg["event_target"] != "exact" or cfg["collision_target"] != "soft_normalized":
        raise ValueError("Mixed run requires exact ENTRY and soft-normalized COLLISION")
    exact_cfg = dict(cfg, event_target="exact")
    total, parts = original_loss(outputs, batch, exact_cfg)
    collision_soft = distribution_loss(
        outputs["collision_logits"], batch["collision_index"],
        batch["time_valid"], batch["normalized_positions"].float(),
        "soft_normalized", cfg["target_width"])
    collision_soft = collision_soft.mean()
    total = total + 0.35 * (collision_soft - parts["loss_collision"])
    parts["loss_collision"] = collision_soft
    return total, parts


def main():
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free < 2 * 1024 ** 3:
            raise RuntimeError(f"Only {free / 1024**3:.2f} GiB GPU free")
        torch.cuda.set_per_process_memory_fraction(0.04)
    trainer.spotting_loss = mixed_loss
    trainer.main()


if __name__ == "__main__":
    main()
