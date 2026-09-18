from pathlib import Path

import numpy as np
import torch

from stage3.inference.dacon import OUTPUT_COLUMNS, predict_stage3
from stage3.model import Stage3MotionModel
from stage3.tests.test_timing import make_video
from stage3.utils.checkpoint import save_checkpoint
from stage3.utils.config import load_config


def test_dacon_output_has_one_row_per_decoded_frame(tmp_path):
    data = tmp_path / "stage3" / "videos"
    data.mkdir(parents=True)
    make_video(data / "sample.mp4", count=7, fps=20)
    cfg = load_config("stage3/configs/smoke.workspace.yaml")
    model = Stage3MotionModel(cfg["model"])
    model_dir = tmp_path / "model" / "stage3"
    model_dir.mkdir(parents=True)
    save_checkpoint(
        model_dir / "best.pt", config=cfg, model=model.state_dict(),
        ema_model=model.state_dict(), physics_center=torch.zeros(20), physics_scale=torch.ones(20),
    )
    output = predict_stage3(tmp_path, tmp_path / "model")
    assert list(output.columns) == OUTPUT_COLUMNS
    assert len(output) == 7
    assert output["sample_index"].tolist() == list(range(7))
    assert output["steer_label"].isin(["LEFT", "STRAIGHT", "RIGHT"]).all()
