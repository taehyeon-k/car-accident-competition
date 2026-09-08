"""Per-sample metrics for aggregation without weighting short batches equally."""

import torch


def batch_metrics(
    outputs: dict,
    batch: dict,
    stage: str,
) -> dict:
    if stage == "coarse":
        values = {}
        for event in ("entry", "collision"):
            predicted = outputs[f"{event}_logits"].argmax(dim=-1)
            error = (predicted - batch[f"{event}_bin"]).abs().float()
            values[f"{event}_bin_mae"] = error
            values[f"{event}_region_recall"] = (error <= 2).float()
        values["direction_accuracy"] = (
            outputs["direction_logits"].argmax(-1) == batch["entry_side"]
        ).float()
        values["evasion_accuracy"] = (
            (outputs["evasion_logits"].squeeze(-1) >= 0) == batch["evasion"].bool()
        ).float()
        return values
    logits = torch.where(
        batch["event_type"][:, None].bool(),
        outputs["collision_logits"],
        outputs["entry_logits"],
    )
    error = (logits.argmax(-1) - batch["event_local_index"]).abs().float()
    return {"frame_mae": error, "exact_frame_accuracy": (error == 0).float()}
