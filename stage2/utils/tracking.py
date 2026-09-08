"""Optional experiment tracking through Accelerate's W&B integration."""

from pathlib import Path

import torch


def tracker_backend(config: dict) -> str | None:
    """Keep older configurations and disabled logging independent of W&B."""
    settings = config.get(
        "logging",
        {},
    ).get(
        "wandb",
        {},
    )
    if not settings.get(
        "enabled",
        False,
    ):
        return None

    if settings.get(
        "mode",
        "offline",
    ) not in {"online", "offline"}:
        raise ValueError("W&B mode must be online or offline")
    if not settings.get("project"):
        raise ValueError("Enabled W&B logging requires a project")
    if settings.get(
        "resume",
        "never",
    ) not in {"never", "allow", "must"}:
        raise ValueError("W&B resume must be never, allow, or must")
    if (
        settings.get(
            "resume",
            "never",
        )
        != "never"
    ):
        if not settings.get("run_id"):
            raise ValueError("Resuming W&B requires an explicit run_id")
        if (
            settings.get(
                "mode",
                "offline",
            )
            != "online"
        ):
            raise ValueError("W&B run resumption requires online mode")
    return "wandb"


def initialize_tracking(
    accelerator,
    config: dict,
) -> None:
    """Accelerate creates the tracker only on its designated main process.

    Log selected hyperparameters, not manifests, credentials, images, or weights.
    W&B run resumption is explicit and independent of model checkpoint resumption.
    """
    if tracker_backend(config) is None:
        return

    settings = config["logging"]["wandb"]
    directory = Path(config["output_dir"])
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    options = {
        "mode": settings.get(
            "mode",
            "offline",
        ),
        "dir": str(directory),
        "job_type": config["stage"],
        "save_code": False,
    }
    for source, destination in {
        "entity": "entity",
        "name": "name",
        "group": "group",
        "tags": "tags",
        "run_id": "id",
    }.items():
        if settings.get(source) is not None:
            options[destination] = settings[source]
    if options["mode"] == "online":
        options["resume"] = settings.get(
            "resume",
            "never",
        )

    hyperparameters = {
        "stage": config["stage"],
        "seed": config["seed"],
        "optimization": config["optimization"],
        "tracking": config["tracking"],
        "model": {
            key: value
            for key, value in config["model"].items()
            if not key.endswith(("_checkpoint", "_factory"))
        },
        "data": {key: config["data"][key] for key in ("batch_size", "num_workers")},
    }
    accelerator.init_trackers(
        settings["project"],
        config=hyperparameters,
        init_kwargs={"wandb": options},
    )


class TrainingMetrics:
    """Accumulate detached sample sums; communicate only at logging boundaries."""

    def __init__(self):
        self.sums = {}
        self.count = 0

    def update(
        self,
        losses,
        components,
    ):
        for name, values in {"loss": losses, **components}.items():
            self.sums[name] = (
                self.sums.get(
                    name,
                    0,
                )
                + values.detach().float().sum()
            )
        self.count += losses.numel()

    def flush(
        self,
        accelerator,
    ) -> dict:
        if not self.count:
            return {}
        names = sorted(self.sums)
        values = torch.stack([self.sums[name] for name in names])
        counts = values.new_tensor([self.count])
        totals = (
            accelerator.reduce(
                torch.cat((values, counts)),
                reduction="sum",
            )
            .cpu()
            .tolist()
        )
        result = {
            f"train/{name}": total / totals[-1]
            for name, total in zip(
                names,
                totals[:-1],
            )
        }
        self.sums.clear()
        self.count = 0
        return result
