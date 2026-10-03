from __future__ import annotations

from pathlib import Path


def tracker_backend(config: dict) -> str | None:
    settings = config.get("logging", {}).get("wandb", {})
    if not settings.get("enabled", False):
        return None
    if settings.get("mode", "offline") not in {"online", "offline"}:
        raise ValueError("W&B mode must be online or offline")
    return "wandb"


def initialize_tracking(accelerator, config: dict) -> None:
    if tracker_backend(config) is None:
        return
    settings = config["logging"]["wandb"]
    options = {
        "mode": settings.get("mode", "offline"), "dir": config["output_dir"],
        "name": settings.get("name"), "group": settings.get("group"),
        "save_code": False, "job_type": "stage3",
    }
    options = {key: value for key, value in options.items() if value is not None}
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    accelerator.init_trackers(settings["project"], config={
        "seed": config["seed"], "model": config["model"],
        "optimization": config["optimization"], "loss": config["loss"],
    }, init_kwargs={"wandb": options})
