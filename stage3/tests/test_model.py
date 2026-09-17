import torch

from stage3.model import Stage3MotionModel
from stage3.trainer.losses import stage3_loss


MODEL = {
    "motion_cnn": {"widths": [32, 64, 96, 128], "blocks_per_stage": 1, "pooling": "attention"},
    "physics_dropout": 0.05, "head_dropout": 0.05, "yaw_aux": True,
    "tcn": {"dim": 128, "dilations": [1, 2, 4, 8, 16], "dropout": 0.1, "causal": False},
}


def test_forward_backward_shapes():
    model = Stage3MotionModel(MODEL)
    output = model(torch.randn(2, 8, 10, 96, 168), torch.randn(2, 8, 20))
    assert output["acceleration"].shape == (2, 8, 2)
    assert output["steering_angle"].shape == (2, 8)
    sum(value.mean() for value in output.values()).backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
