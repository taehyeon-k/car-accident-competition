from __future__ import annotations

import math

import numpy as np


_GEOCALIB_MODELS: dict[tuple[str, str], object] = {}


def focal_from_hfov(width: int, hfov_deg: float) -> float:
    if width < 1 or not 1.0 < hfov_deg < 179.0:
        raise ValueError("Need positive width and horizontal FOV in (1, 179) degrees")
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def estimate_focal(width: int, height: int, cfg: dict, image: np.ndarray | None = None) -> float:
    mode = cfg.get("focal_mode", "prior")
    if mode == "prior":
        return focal_from_hfov(width, float(cfg["hfov_prior_deg"]))
    if mode == "known":
        focal = float(cfg["known_focal_px"])
        if focal <= 0:
            raise ValueError("known_focal_px must be positive")
        return focal
    if mode == "geocalib":
        try:
            from geocalib import GeoCalib
        except ImportError as error:
            raise RuntimeError("focal_mode=geocalib requires the optional geocalib package") from error
        if image is None:
            raise ValueError("focal_mode=geocalib requires a representative RGB frame")
        import torch

        device = str(cfg.get("geocalib_device", "cuda" if torch.cuda.is_available() else "cpu"))
        weights = str(cfg.get("geocalib_weights", "pinhole"))
        key = (device, weights)
        if key not in _GEOCALIB_MODELS:
            _GEOCALIB_MODELS[key] = GeoCalib(weights=weights).to(device).eval()
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255).to(device)
        with torch.inference_mode():
            result = _GEOCALIB_MODELS[key].calibrate(tensor)
        focal = float(result["camera"].f.reshape(-1).mean().item())
        if not math.isfinite(focal) or focal <= 0:
            raise ValueError(f"GeoCalib returned invalid focal length {focal}")
        return focal
    raise ValueError(f"Unknown focal mode {mode!r}")
