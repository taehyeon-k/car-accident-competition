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
        state = load_checkpoint(checkpoint, "cpu")
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
        def timestamp():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            return perf_counter()

        timing: dict[str, float] = {}
        started = timestamp()
        height, width = self.cfg["flow"]["working_size"]
        decoded = decode_dacon_stage3_video(path, max_frames=max_frames, resize_hw=(height, width))
        timing["decode"] = timestamp() - started
        height, width = self.cfg["flow"]["working_size"]
        frames = decoded.frames
        flow_start = timestamp()
        flows, confidence = self.flow.estimate_sequence(frames)
        timing["sea_raft"] = timestamp() - flow_start
        geometry_start = timestamp()
        # Competition contract is fixed 0.1 s and does not use PTS for dt.
        fixed_times = np.arange(len(frames), dtype=np.float64) * 0.1
        motion, physics, geometry_metadata = build_motion_features(
            flows, confidence, fixed_times, self.cfg["calibration"], tuple(self.cfg["geometry"]["canonical_size"]), frames[0],
            tracking_device=str(self.device),
            tracking_batch_size=int(self.cfg["geometry"].get("tracking_batch_size", 32)),
            geometry_backend=self.cfg["geometry"].get("backend", "auto"),
        )
        timing["geometry"] = timestamp() - geometry_start
        for name in ("calibration", "rotation", "foe", "tracks_rho", "feature_construction"):
            timing[name] = float(geometry_metadata[f"timing_{name}"][0])
        if "timing_transfers" in geometry_metadata:
            timing["geometry_transfers"] = float(geometry_metadata["timing_transfers"][0])
        model_start = timestamp()
        motion_tensor = torch.from_numpy(motion)[None]
        physics_tensor = (torch.from_numpy(physics).to(self.device) - self.physics_center) / self.physics_scale
        spatial = self.model.encode_motion(motion_tensor, int(self.cfg.get("inference", {}).get("cnn_chunk_frames", 32)))
        timing["cnn"] = timestamp() - model_start
        tcn_start = timestamp()
        physical = self.model.physics_mlp(physics_tensor[None])
        encoded = self.model.temporal(self.model.fusion(torch.cat((spatial, physical), dim=-1)))
        outputs = self.model.heads(encoded)
        timing["tcn"] = timestamp() - tcn_start
        decode_start = timestamp()
        accel, steer = decode_predictions(outputs, self.cfg["decoder"])
        timing["decoder"] = timestamp() - decode_start
        timing["total"] = timestamp() - started
        if len(accel) != len(decoded.frames):
            raise RuntimeError("Stage 3 inference violated one-row-per-decoded-frame contract")
        return accel, steer, timing
