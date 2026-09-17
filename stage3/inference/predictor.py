from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from stage3.data.timing import decode_dacon_stage3_video
from stage3.flow import build_flow_estimator
from stage3.flow.sea_raft import resize_frames
from stage3.geometry import build_motion_features
from stage3.model import Stage3MotionModel
from stage3.trainer.decoder import decode_predictions
from stage3.utils.checkpoint import load_checkpoint


class Stage3Predictor:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = load_checkpoint(checkpoint, self.device)
        self.cfg = state["config"]
        packaged = Path(checkpoint).resolve().parent / "pretrained" / "sea_raft"
        if packaged.is_dir():
            self.cfg["flow"] = dict(self.cfg["flow"])
            self.cfg["flow"]["source_path"] = str(packaged / "source")
            self.cfg["flow"]["checkpoint"] = str(packaged / "model.safetensors")
        self.model = Stage3MotionModel(self.cfg["model"]).to(self.device)
        self.model.load_state_dict(state.get("ema_model", state["model"]))
        self.model.eval()
        self.physics_center = torch.as_tensor(state.get("physics_center", np.zeros(20)), device=self.device).float()
        self.physics_scale = torch.as_tensor(state.get("physics_scale", np.ones(20)), device=self.device).float().clamp_min(1e-6)
        self.flow = build_flow_estimator(self.cfg["flow"], str(self.device))

    @torch.inference_mode()
    def predict_video(self, path: str | Path, max_frames: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        timing: dict[str, float] = {}
        started = perf_counter()
        decoded = decode_dacon_stage3_video(path, max_frames=max_frames)
        timing["decode"] = perf_counter() - started
        height, width = self.cfg["flow"]["working_size"]
        frames = resize_frames(decoded.frames, (height, width))
        flow_start = perf_counter()
        flows, confidence = self.flow.estimate_sequence(frames)
        timing["sea_raft"] = perf_counter() - flow_start
        geometry_start = perf_counter()
        # Competition contract is fixed 0.1 s and does not use PTS for dt.
        fixed_times = np.arange(len(frames), dtype=np.float64) * 0.1
        motion, physics, geometry_metadata = build_motion_features(
            flows, confidence, fixed_times, self.cfg["calibration"], tuple(self.cfg["geometry"]["canonical_size"]), frames[0]
        )
        timing["geometry"] = perf_counter() - geometry_start
        for name in ("calibration", "rotation", "foe", "tracks_rho", "feature_construction"):
            timing[name] = float(geometry_metadata[f"timing_{name}"][0])
        model_start = perf_counter()
        motion_tensor = torch.from_numpy(motion)[None].to(self.device)
        physics_tensor = (torch.from_numpy(physics).to(self.device) - self.physics_center) / self.physics_scale
        spatial = self.model.motion_cnn(motion_tensor)
        timing["cnn"] = perf_counter() - model_start
        tcn_start = perf_counter()
        physical = self.model.physics_mlp(physics_tensor[None])
        encoded = self.model.temporal(self.model.fusion(torch.cat((spatial, physical), dim=-1)))
        outputs = self.model.heads(encoded)
        timing["tcn"] = perf_counter() - tcn_start
        decode_start = perf_counter()
        accel, steer = decode_predictions(outputs, self.cfg["decoder"])
        timing["decoder"] = perf_counter() - decode_start
        timing["total"] = perf_counter() - started
        if len(accel) != len(decoded.frames):
            raise RuntimeError("Stage 3 inference violated one-row-per-decoded-frame contract")
        return accel, steer, timing
