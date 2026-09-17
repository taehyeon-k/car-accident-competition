from __future__ import annotations

import argparse

import torch

from stage3.data.dataset import CachedMotionDataset, motion_collate
from stage3.model import Stage3MotionModel
from stage3.trainer.losses import stage3_loss
from stage3.utils.config import load_config, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Intentionally overfit one cached BATON clip")
    parser.add_argument("--config", default="stage3/configs/smoke.workspace.yaml")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = torch.device(args.device)
    dataset = CachedMotionDataset(
        cfg["data"]["manifest"], cfg["data"]["crop_frames"], False,
        cfg["seed"], 0, cfg["targets"],
    )
    batch = motion_collate([dataset[0]])
    stats = torch.load(cfg["data"]["statistics"], map_location="cpu", weights_only=True)
    batch["physics"] = (batch["physics"] - stats["center"]) / stats["scale"].clamp_min(1e-6)
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    model = Stage3MotionModel(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0)

    def evaluate() -> float:
        model.eval()
        torch.manual_seed(0)
        with torch.no_grad():
            return float(stage3_loss(model(batch["motion"], batch["physics"]), batch, cfg["loss"])[0])

    initial = evaluate()
    model.train()
    for step in range(args.steps):
        torch.manual_seed(step + 1)
        loss, _ = stage3_loss(model(batch["motion"], batch["physics"]), batch, cfg["loss"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = evaluate()
    print({"steps": args.steps, "initial_loss": initial, "final_loss": final, "ratio": final / initial})
    if not final < initial:
        raise RuntimeError("Tiny overfit did not reduce the fixed-seed objective")


if __name__ == "__main__":
    main()
