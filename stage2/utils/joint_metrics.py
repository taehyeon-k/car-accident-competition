"""Interval-wide joint metrics from per-sample counts, never batch F1 averages."""

import torch
import torch.nn.functional as F

from stage2.utils.joint_losses import constrained_decode


@torch.no_grad()
def joint_metric_packet(outputs, batch):
    entry, collision = constrained_decode(
        outputs["entry_logits"], outputs["collision_logits"]
    )
    seconds = batch["frame_seconds"].float()
    result = {}
    for event, prediction in (("entry", entry), ("collision", collision)):
        predicted = seconds.gather(1, prediction[:, None]).squeeze(1)
        target = seconds.gather(1, batch[f"{event}_index"][:, None]).squeeze(1)
        result[f"_acc_{event}_0.3s"] = (
            (predicted - target).abs() <= 0.3 + 1e-6
        ).float()
    for name, prediction, target in (
        ("entry_side", outputs["side_logits"].argmax(-1), batch["entry_side"]),
        (
            "evasion_space",
            (outputs["evasion_logits"] >= 0).long(),
            batch["evasion"].long(),
        ),
    ):
        result[f"_confusion_{name}"] = F.one_hot(target * 2 + prediction, 4).long()
    return result


class JointMetricAccumulator:
    """Accept already gathered, duplicate-trimmed sample packets on every rank."""

    def __init__(self):
        self.sums = {}
        self.count = 0

    def update(self, packet):
        if not packet:
            return
        self.count += next(iter(packet.values())).shape[0]
        for name, values in packet.items():
            value = values.detach().double().sum(0)
            self.sums[name] = self.sums.get(name, torch.zeros_like(value)) + value

    def compute(self):
        if not self.count:
            return {}
        values = {
            name[1:]: float(self.sums[name] / self.count)
            for name in ("_acc_entry_0.3s", "_acc_collision_0.3s")
        }
        for attribute in ("entry_side", "evasion_space"):
            matrix = self.sums[f"_confusion_{attribute}"].reshape(2, 2)
            denominator = matrix.sum(0) + matrix.sum(1)
            f1 = 2 * matrix.diag() / denominator.clamp_min(1)
            values[f"f1_{attribute}_macro"] = float(f1.mean())
        values["competition_score"] = (
            0.35 * values["acc_entry_0.3s"]
            + 0.35 * values["acc_collision_0.3s"]
            + 0.15 * values["f1_entry_side_macro"]
            + 0.15 * values["f1_evasion_space_macro"]
        )
        return values

    def state_dict(self):
        return {"count": self.count, "sums": {k: v.cpu() for k, v in self.sums.items()}}

    def load_state_dict(self, state, device):
        self.count = state["count"]
        self.sums = {k: v.to(device) for k, v in state["sums"].items()}
