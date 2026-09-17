import torch

from stage3.data.augment import horizontal_flip


def test_double_flip_identity_and_lateral_signs():
    motion = torch.randn(5, 10, 8, 12)
    physics = torch.randn(5, 20)
    targets = {"a_long_s1": torch.randn(5), "steering_angle": torch.randn(5), "yaw_rate_aux": torch.randn(5)}
    once = horizontal_flip(motion, physics, targets)
    twice = horizontal_flip(*once)
    assert torch.equal(twice[0], motion)
    assert torch.equal(twice[1], physics)
    assert torch.equal(twice[2]["a_long_s1"], targets["a_long_s1"])
    assert torch.equal(once[2]["steering_angle"], -targets["steering_angle"])
