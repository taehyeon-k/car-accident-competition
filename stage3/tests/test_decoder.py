import numpy as np

from stage3.trainer.decoder import decode_predictions, potts_viterbi


CFG = {
    "acceleration": {"decelerating_below": -0.25, "accelerating_above": 0.25, "emission_scale": 0.5, "potts_penalty": 0},
    "steering": {"threshold_deg": 5, "emission_scale_deg": 5, "potts_penalty": 0},
}


def test_zero_potts_is_framewise():
    scores = np.asarray([[0, 2, 1], [4, 2, 3]])
    assert np.array_equal(potts_viterbi(scores, 0), scores.argmax(1))


def test_steering_uses_angle_not_yaw():
    outputs = {
        "acceleration": np.zeros((3, 2)), "stop_logit": np.full(3, -10),
        "steering_angle": np.asarray([12.0, 0.0, -12.0]),
        "yaw_rate_aux": np.asarray([-100.0, 100.0, 100.0]),
    }
    _, steering = decode_predictions(outputs, CFG)
    assert steering.tolist() == ["LEFT", "STRAIGHT", "RIGHT"]
