import numpy as np

from stage3.geometry.calibration import estimate_focal, focal_from_hfov
from stage3.geometry.foe import estimate_foe
from stage3.geometry.rho import rho_from_expansion
from stage3.geometry.rotation import estimate_rotation, rotational_design


def test_prior_focal_has_no_geocalib_dependency():
    assert np.isclose(estimate_focal(640, 360, {"focal_mode": "prior", "hfov_prior_deg": 90}), 320)
    assert np.isclose(focal_from_hfov(1280, 90) / focal_from_hfov(640, 90), 2)


def test_foe_recovery_and_sign():
    h, w = 48, 80
    expected = np.asarray([31.5, 19.0])
    y, x = np.mgrid[:h, :w]
    flow = np.stack(((x - expected[0]) * 0.03, (y - expected[1]) * 0.03)).astype(np.float32)
    foe, inlier, _ = estimate_foe(flow, np.ones((h, w), np.float32))
    assert np.allclose(foe, expected, atol=0.1)
    assert inlier > 0.95


def test_rotation_recovery():
    h, w, focal = 40, 64, 50.0
    expected = np.asarray([0.003, -0.004, 0.006])
    flow = np.einsum("hwkc,c->hwk", rotational_design(h, w, focal), expected).transpose(2, 0, 1)
    recovered, predicted, inlier, residual = estimate_rotation(flow, np.ones((h, w)), focal)
    assert np.allclose(recovered, expected, atol=1e-5)
    assert np.allclose(predicted, flow, atol=1e-5)
    assert inlier > 0.99 and residual < 1e-5


def test_rotation_is_invariant_to_joint_focal_and_image_scaling():
    expected = np.asarray([0.002, -0.003, 0.004])
    recovered = []
    for h, w, focal in ((40, 64, 50.0), (80, 128, 100.0)):
        flow = np.einsum("hwkc,c->hwk", rotational_design(h, w, focal), expected).transpose(2, 0, 1)
        recovered.append(estimate_rotation(flow, np.ones((h, w)), focal)[0])
    assert np.allclose(recovered[0], recovered[1], atol=1e-5)


def test_rho_recovers_acceleration_over_speed():
    dt, v0, acceleration, z0 = 0.4, 10.0, 2.0, 30.0
    v1 = v0 + acceleration * dt
    z1 = z0 - 0.5 * (v0 + v1) * dt
    previous = np.full((8, 8), v0 / z0)
    current = np.full((8, 8), v1 / z1)
    estimate, valid = rho_from_expansion(previous, current, dt)
    assert valid.all()
    assert np.allclose(estimate, np.log(v1 / v0) / dt, atol=1e-6)
