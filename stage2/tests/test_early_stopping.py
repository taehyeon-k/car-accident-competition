import pytest
from stage2.utils.early_stopping import EarlyStopping
from stage2.tests.test_joint import tiny_joint_trainer


def test_patience_delta_and_resume():
    settings = dict(
        enabled=True, monitor="competition_score", patience=3, min_delta=0.002
    )
    stopper = EarlyStopping(settings)
    assert not stopper.update({"competition_score": 0.6})
    assert not stopper.update({"competition_score": 0.601})
    resumed = EarlyStopping(settings)
    resumed.load_state_dict(stopper.state_dict())
    assert resumed.bad_validations == 1
    assert not resumed.update({"competition_score": 0.60})
    assert resumed.update({"competition_score": 0.59})
    assert not resumed.update({"competition_score": 0.61})
    assert resumed.bad_validations == 0


def test_loss_mode_and_nonfinite_values():
    stopper = EarlyStopping(
        dict(enabled=True, monitor="loss", patience=2, min_delta=0.01)
    )
    assert not stopper.update({"loss": 3.0})
    assert not stopper.update({"loss": 2.9})
    assert not stopper.update({"loss": float("nan")})
    assert stopper.update({"loss": float("inf")})
    assert not EarlyStopping().update({})
    with pytest.raises(ValueError):
        EarlyStopping({"patience": 0})


def test_changed_policy_resets_counter():
    original = EarlyStopping(dict(enabled=True, patience=2))
    original.update({"loss": 1})
    changed = EarlyStopping(dict(enabled=True, patience=4))
    changed.load_state_dict(original.state_dict())
    assert changed.best is None


def test_trainer_saves_last_before_stopping():
    trainer = tiny_joint_trainer(
        epochs=8, val_every=2, checkpoint_metric="competition_score"
    )
    trainer.early_stopping = EarlyStopping(
        dict(enabled=True, monitor="competition_score", patience=2)
    )
    trainer.accelerator.is_main_process = True
    scores = iter([0.7, 0.6, 0.65, 0.8])

    def validate():
        trainer.last_validation_metrics = {"competition_score": next(scores)}
        return 3.0

    trainer.validate = validate
    trainer.train_loop()
    assert trainer.saved[-1] == (5, "last.pt")
    assert (1, "best.pt") in trainer.saved
    assert not any(epoch > 5 for epoch, _ in trainer.saved)
