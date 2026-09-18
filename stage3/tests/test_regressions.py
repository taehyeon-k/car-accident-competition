from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from stage3.data.dataset import motion_collate
from stage3.geometry.rho import rho_from_expansion
from stage3.model.tcn import TemporalConvNet
from stage3.trainer.decoder import decode_predictions
from stage3.trainer.losses import stage3_loss
from stage3.trainer.metrics import classification_metrics
from stage3.trainer.trainer import Trainer
from stage3.tests.test_decoder import CFG


def test_decoder_thresholds_stop_gate_and_all_frame_labels():
    cfg = deepcopy(CFG)
    cfg['acceleration']['stopped_probability'] = .7
    cfg['steering']['emission_scale_deg'] = 2
    outputs = {'acceleration': np.repeat(np.array([.3, 3., -3., 0., 0.])[:, None], 2, axis=1),
               'stop_logit': np.log(np.array([.01, .01, .01, .6, .8]) / np.array([.99, .99, .99, .4, .2])),
               'steering_angle': np.array([5.1, -5.1, 0., 0., 12.])}
    accel, steer = decode_predictions(outputs, cfg)
    assert accel.tolist() == ['ACCELERATING', 'ACCELERATING', 'DECELERATING', 'CONSTANT', 'STOPPED']
    assert steer.tolist() == ['LEFT', 'RIGHT', 'STRAIGHT', 'STRAIGHT', 'LEFT']


def test_steering_metric_excludes_true_stopped_only():
    true_a = np.array(['CONSTANT', 'ACCELERATING', 'DECELERATING', 'STOPPED'])
    pred_a = np.array(['STOPPED', 'ACCELERATING', 'DECELERATING', 'CONSTANT'])
    true_s = np.array(['LEFT', 'RIGHT', 'STRAIGHT', 'LEFT'])
    pred_s = np.array(['LEFT', 'RIGHT', 'STRAIGHT', 'RIGHT'])
    assert classification_metrics(pred_a, true_a, pred_s, true_s)['steering_macro_f1'] == 1
    pred_s[0] = 'RIGHT'
    assert classification_metrics(pred_a, true_a, pred_s, true_s)['steering_macro_f1'] < 1
    assert classification_metrics(pred_a[-1:], true_a[-1:], pred_s[-1:], true_s[-1:])['steering_macro_f1'] == 0
    with pytest.raises(ValueError, match='same shape'):
        classification_metrics(pred_a, true_a, pred_s[:-1], true_s)


@pytest.mark.parametrize('acceleration', [0., 2., -2.])
def test_rho_accounts_for_depth_change(acceleration):
    dt, v0, z0 = .4, 10., 30.
    v1 = v0 + acceleration * dt
    z1 = z0 - .5 * (v0 + v1) * dt
    estimate, valid = rho_from_expansion(np.full((8, 8), v0/z0), np.full((8, 8), v1/z1), dt)
    assert valid.all()
    assert np.allclose(estimate, np.log(v1/v0)/dt, atol=1e-6)


def test_tcn_variable_lengths_match_whole_sequences_including_endpoints():
    torch.manual_seed(4)
    model = TemporalConvNet(dim=16, dilations=(1, 2, 4), dropout=0).eval()
    data = torch.randn(2, 30, 16, requires_grad=True)
    actual = model(data, torch.tensor([21, 30]))
    for i, length in enumerate([21, 30]):
        assert torch.allclose(actual[i, :length], model(data[i:i+1, :length])[0], atol=1e-6)
    actual.sum().backward()
    assert torch.isfinite(data.grad).all()
    assert not data.grad[0, 21:].any()


def test_stopped_loss_ignores_missing_speed():
    shape = (1, 2)
    batch = {name: torch.zeros(shape) for name in ['a_long_s1', 'a_long_s2', 'a_dvdt_s1', 'a_dvdt_s2', 'speed', 'steering_angle', 'stopped']}
    batch.update({name: torch.ones(shape, dtype=torch.bool) for name in ['time_valid', 'valid_accel', 'valid_accel_speed', 'valid_speed', 'valid_steer', 'valid_yaw']})
    batch['valid_speed'][0, 1] = False
    batch['speed'][0, 1] = float('nan')
    stop = torch.tensor([[0., 10.]], requires_grad=True)
    outputs = {'acceleration': torch.zeros(1, 2, 2, requires_grad=True), 'speed': torch.zeros(shape), 'steering_angle': torch.zeros(shape), 'stop_logit': stop}
    loss, parts = stage3_loss(outputs, batch, {'ordinal': {}, 'weights': {'stopped': 1}})
    assert parts['stopped'].item() == pytest.approx(np.log(2))
    loss.backward()
    assert stop.grad[0, 1] == 0
    assert stop.grad[0, 0] != 0


def test_validation_respects_holes_target_masks_and_sequence_lengths():
    class Model(torch.nn.Module):
        def forward(self, motion, physics, lengths):
            assert lengths.tolist() == [5]
            # Errors occur exclusively on the invalid time or invalid target.
            return {'acceleration': torch.tensor([[[1., 1.], [-2., -2.], [-1., -1.], [2., 2.], [0., 0.]]]),
                    'speed': torch.ones(1, 5), 'stop_logit': torch.full((1, 5), -10.),
                    'steering_angle': torch.tensor([[10., -10., -10., 10., 0.]])}
    item = {'clip_id': 'x', 'motion': torch.zeros(5, 10, 2, 2), 'physics': torch.zeros(5, 20),
            'time_valid': torch.tensor([True, False, True, True, True]),
            'a_long_s1': torch.tensor([1., 1., -1., float('nan'), 0.]),
            'a_long_s2': torch.tensor([1., 1., -1., float('nan'), 0.]),
            'speed': torch.ones(5), 'stopped': torch.zeros(5),
            'steering_angle': torch.tensor([10., 10., -10., float('nan'), 0.]),
            'valid_accel': torch.tensor([True, True, True, False, True]),
            'valid_steer': torch.tensor([True, True, True, False, True]),
            'valid_speed': torch.ones(5, dtype=torch.bool)}
    trainer = Trainer(SimpleNamespace(device='cpu'), {'decoder': CFG})
    trainer.ema = SimpleNamespace(model=Model())
    trainer.physics_center, trainer.physics_scale = torch.zeros(20), torch.ones(20)
    trainer.val_loader = [motion_collate([item])]
    metrics = trainer.validate()
    assert metrics['steering_macro_f1'] == 1
    assert metrics['acceleration_macro_f1'] == .75
    assert metrics['direct_acceleration_mae'] == 0


def test_baseline_config_is_valid_yaml():
    from pathlib import Path
    path = Path(__file__).parents[1] / 'configs/baseline_v1_2.yaml'
    assert yaml.safe_load(path.read_text())['stage'] == 'stage3'


def test_projected_constant_speed_has_zero_rho(monkeypatch):
    import stage3.geometry.pipeline as pipeline
    frames, h, w, dt, speed, initial_depth = 8, 48, 80, .1, 10., 30.
    foe = np.array([(w - 1)/2, (h - 1)/2])
    y, x = np.mgrid[:h, :w]
    radius = np.stack((x - foe[0], y - foe[1]))
    flows = np.zeros((frames, 2, h, w), np.float32)
    for i in range(1, frames):
        flows[i] = radius * (speed * dt / (initial_depth - speed * i * dt))
    monkeypatch.setattr(pipeline, 'estimate_rotation', lambda flow, confidence, focal: (np.zeros(3), np.zeros_like(flow), 1., 0.))
    monkeypatch.setattr(pipeline, 'estimate_foe', lambda flow, confidence: (foe, 1., 0.))
    motion, physics, _ = pipeline.build_motion_features(flows, np.ones((frames, h, w)), np.arange(frames)*dt,
                                                     {'focal_mode': 'prior', 'hfov_prior_deg': 90}, (h, w))
    assert np.allclose(physics[5:, 5:7], 0, atol=1e-5)
    assert np.isfinite(motion).all()


def test_empty_metric_masks_are_well_defined():
    labels = np.array(['CONSTANT'])
    steering = np.array(['STRAIGHT'])
    result = classification_metrics(labels, labels, steering, steering, np.array([False]), np.array([False]))
    assert result['acceleration_macro_f1'] == result['steering_macro_f1'] == 0


def test_decoder_exact_boundaries():
    outputs = {'acceleration': np.array([[.25, .25], [-.25, -.25], [0., 0.]]),
               'stop_logit': np.array([-10., -10., 0.]), 'steering_angle': np.array([5., -5., 0.])}
    accel, steer = decode_predictions(outputs, CFG)
    assert accel.tolist() == ['CONSTANT', 'CONSTANT', 'STOPPED']
    assert steer.tolist() == ['STRAIGHT'] * 3


def test_stopped_frames_still_require_steering_prediction():
    with pytest.raises(ValueError, match="every frame"):
        classification_metrics(np.array(['STOPPED']), np.array(['STOPPED']),
                               np.array([None]), np.array(['LEFT']))


def test_full_model_padding_preserves_all_heads():
    from stage3.model import Stage3MotionModel
    from stage3.tests.test_model import MODEL
    torch.manual_seed(3)
    model = Stage3MotionModel(MODEL).eval()
    motion, physics = torch.randn(1, 5, 10, 32, 32), torch.randn(1, 5, 20)
    with torch.no_grad():
        expected = model(motion, physics)
        actual = model(torch.cat((motion, torch.randn(1, 3, 10, 32, 32)), 1),
                       torch.cat((physics, torch.randn(1, 3, 20)), 1), torch.tensor([5]))
    for name in expected:
        assert torch.allclose(actual[name][:, :5], expected[name], atol=1e-5), name
