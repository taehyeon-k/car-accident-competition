from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stage3.data.dataset import motion_collate
from stage3.tests.distributed_validation_worker import Samples, Predictions
from stage3.tests.test_decoder import CFG
from stage3.trainer.metrics import TrainingCompetitionMetrics, classification_metrics, competition_score
from stage3.trainer.trainer import Trainer


def test_official_weights_and_stopped_steering_exclusion():
    accel = np.array(['ACCELERATING', 'DECELERATING', 'CONSTANT', 'STOPPED'])
    steer = np.array(['LEFT', 'RIGHT', 'STRAIGHT', 'LEFT'])
    # The stopped frame's wrong steering prediction must not affect the score.
    prediction = np.array(['LEFT', 'RIGHT', 'STRAIGHT', 'RIGHT'])
    metrics = classification_metrics(accel, accel, prediction, steer)
    assert metrics['competition_score'] == pytest.approx(1.)
    wrong_steer = np.array(['RIGHT', 'STRAIGHT', 'LEFT', 'RIGHT'])
    assert classification_metrics(accel, accel, wrong_steer, steer)['competition_score'] == pytest.approx(.7)
    wrong_accel = np.array(['DECELERATING', 'CONSTANT', 'STOPPED', 'ACCELERATING'])
    assert classification_metrics(wrong_accel, accel, prediction, steer)['competition_score'] == pytest.approx(.3)
    assert competition_score(.8, .4) == pytest.approx(.68)


def test_training_score_aggregates_counts_not_batch_macro_f1():
    samples, model = Samples(), Predictions()
    whole = motion_collate([samples[i] for i in range(len(samples))])
    expected = TrainingCompetitionMetrics()
    expected.update(model(whole['motion'], whole['physics'], whole['lengths']), whole, CFG)
    actual = TrainingCompetitionMetrics()
    individual_scores = []
    for i in range(len(samples)):
        batch = motion_collate([samples[i]])
        outputs = model(batch['motion'], batch['physics'], batch['lengths'])
        actual.update(outputs, batch, CFG)
        individual = TrainingCompetitionMetrics()
        individual.update(outputs, batch, CFG)
        individual_scores.append(individual.compute()['competition_score'])
    assert actual.compute() == expected.compute()
    assert actual.compute()['competition_score'] != pytest.approx(np.mean(individual_scores))
    # Four moving clips: 5 + 6 + 7 + 9 frames; stopped clip is excluded.
    assert actual.steering.sum() == 27
    assert actual.acceleration.sum() == 35


def test_no_valid_frames_scores_zero():
    accumulator = TrainingCompetitionMetrics()
    batch = motion_collate([Samples()[0]])
    batch['time_valid'].fill_(False)
    accumulator.update(Predictions()(batch['motion'], batch['physics'], batch['lengths']), batch, CFG)
    assert accumulator.compute()['competition_score'] == 0


def make_trainer(monkeypatch, epochs=3, val_every=1):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))
        def forward(self, motion, physics, lengths):
            b, t = motion.shape[:2]
            value = self.weight.expand(b, t)
            return {'acceleration': value[..., None].expand(-1, -1, 2),
                    'steering_angle': value, 'stop_logit': value - 10., 'speed': value + 1.}
    logs = []
    accelerator = SimpleNamespace(
        device='cpu', num_processes=1, sync_gradients=True,
        accumulate=lambda model: nullcontext(), backward=lambda loss: loss.backward(),
        clip_grad_norm_=torch.nn.utils.clip_grad_norm_, unwrap_model=lambda model: model,
        log=lambda values, **kwargs: logs.append((dict(values), kwargs)), print=lambda *args: None,
    )
    cfg = {'optimization': {'epochs': epochs, 'grad_clip_norm': 1.},
           'logging': {'val_every': val_every}, 'decoder': CFG, 'loss': {}}
    trainer = Trainer(accelerator, cfg)
    trainer.model = Model()
    trainer.ema = SimpleNamespace(update=lambda model: None, model=Model())
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.01)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda step: 1.)
    trainer.train_set = SimpleNamespace(set_epoch=lambda epoch: None)
    trainer.train_loader = [motion_collate([Samples()[2]])]
    trainer.physics_center, trainer.physics_scale = torch.zeros(20), torch.ones(20)
    trainer.start_epoch, trainer.global_step, trainer.best, trainer.bad_validations = 0, 0, -float('inf'), 0
    monkeypatch.setattr('stage3.trainer.trainer.stage3_loss', lambda outputs, batch, cfg: (outputs['acceleration'].mean(), {}))
    saved = []
    trainer._save = lambda name, epoch, metrics: saved.append((name, epoch, dict(metrics), trainer.best))
    return trainer, logs, saved


def test_logging_checkpoint_and_early_stop_use_competition_score(monkeypatch):
    trainer, logs, saved = make_trainer(monkeypatch, epochs=5)
    trainer.cfg['early_stopping'] = {'enabled': True, 'patience': 1}
    # Competition score improves while the old selection score gets worse.
    results = iter([(.4, .9), (.7, .1), (.6, .99)])
    def validate():
        score, old_score = next(results)
        return {'competition_score': score, 'acceleration_macro_f1_threshold_grid_mean': old_score}
    trainer.validate = validate
    trainer.train_loop()
    assert [(name, epoch) for name, epoch, _, _ in saved if name == 'best.pt'] == [('best.pt', 0), ('best.pt', 1)]
    assert trainer.best == .7 and trainer.bad_validations == 1
    assert len(logs) == 3  # stops despite improvement in the old score
    for values, kwargs in logs:
        assert 'train/competition_score' in values and 'val/competition_score' in values
        assert values['train/competition_score'] == pytest.approx(.7 * values['train/acceleration_macro_f1'] + .3 * values['train/steering_macro_f1'])
        assert kwargs['log_kwargs']['wandb']['commit'] is True


def test_train_and_validation_score_intervals_match(monkeypatch):
    trainer, logs, _ = make_trainer(monkeypatch, epochs=3, val_every=2)
    trainer.validate = lambda: {'competition_score': .5}
    trainer.train_loop()
    assert [('train/competition_score' in values, 'val/competition_score' in values) for values, _ in logs] == [(False, False), (True, True), (True, True)]


@pytest.mark.parametrize('metric_name, expected_best, expected_bad', [(None, -float('inf'), 0), ('competition_score', .65, 2)])
def test_resume_resets_old_selection_history_only(monkeypatch, metric_name, expected_best, expected_bad):
    trainer, _, _ = make_trainer(monkeypatch)
    state = {'feature_cache_key': 'key', 'physics_center': torch.zeros(20), 'physics_scale': torch.ones(20),
             'model': trainer.model.state_dict(), 'ema_model': trainer.ema.model.state_dict(),
             'optimizer': trainer.optimizer.state_dict(), 'scheduler': trainer.scheduler.state_dict(),
             'epoch': 4, 'global_step': 100, 'best_metric': .65, 'bad_validations': 2,
             'best_metric_name': metric_name}
    monkeypatch.setattr('stage3.trainer.trainer.load_checkpoint', lambda path, device: state)
    monkeypatch.setattr('stage3.trainer.trainer.cache_key', lambda cfg: 'key')
    trainer.resume('unused.pt')
    assert trainer.best == expected_best and trainer.bad_validations == expected_bad
    assert trainer.start_epoch == 5 and trainer.global_step == 100
