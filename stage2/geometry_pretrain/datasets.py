"""Training/evaluation datasets over the cached manifests and label cache.

Missing targets are expressed as IGNORE / zero weight so that every task is
simply masked for samples that do not carry it. Images are returned as uint8
(normalized on the GPU) to keep the 8-core host out of the critical path.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from stage2.geometry_pretrain.common import IGNORE, INPUT_HW, LABEL_HW, LABEL_STRIDE, read_jsonl
from stage2.geometry_pretrain.pseudo_labels.generate import flow_path, label_path

cv2.setNumThreads(1)  # dataloader workers already parallelize; avoid CPU oversubscription

SOURCE_IDS = {"bdd100k": 0, "tusimple": 1, "accident": 2, "baton": 3}


def load_manifest(manifest_dir, sources, split, pairs=False):
    rows = []
    for s in sources:
        p = Path(manifest_dir) / f"{s}_{split}{'_pairs' if pairs else ''}.jsonl"
        if p.is_file():
            rows += read_jsonl(p)
    return rows


# ----------------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------------


class Photometric:
    def __init__(self, cfg: dict | None):
        self.cfg = cfg or {}

    def params(self, rng: random.Random):
        c = self.cfg
        return {
            "brightness": rng.uniform(*c.get("brightness", (0.8, 1.2))) if rng.random() < c.get("p_color", 0.8) else 1.0,
            "contrast": rng.uniform(*c.get("contrast", (0.8, 1.2))) if rng.random() < c.get("p_color", 0.8) else 1.0,
            "saturation": rng.uniform(*c.get("saturation", (0.8, 1.2))) if rng.random() < c.get("p_color", 0.8) else 1.0,
            "gamma": rng.uniform(*c.get("gamma", (0.85, 1.15))) if rng.random() < c.get("p_gamma", 0.3) else 1.0,
            "blur": rng.uniform(0.3, 1.2) if rng.random() < c.get("p_blur", 0.15) else 0.0,
            "noise": rng.uniform(0.0, 6.0) if rng.random() < c.get("p_noise", 0.15) else 0.0,
            "jpeg": rng.randint(45, 90) if rng.random() < c.get("p_jpeg", 0.1) else 0,
        }

    @staticmethod
    def apply(img: np.ndarray, p: dict, rng: np.random.Generator) -> np.ndarray:
        x = img.astype(np.float32)
        if p["brightness"] != 1.0:
            x *= p["brightness"]
        if p["contrast"] != 1.0:
            m = x.mean()
            x = (x - m) * p["contrast"] + m
        if p["saturation"] != 1.0:
            g = x.mean(-1, keepdims=True)
            x = (x - g) * p["saturation"] + g
        x = np.clip(x, 0, 255)
        if p["gamma"] != 1.0:
            x = 255.0 * (x / 255.0) ** p["gamma"]
        if p["blur"] > 0:
            x = cv2.GaussianBlur(x, (0, 0), p["blur"])
        if p["noise"] > 0:
            x = x + rng.normal(0, p["noise"], x.shape).astype(np.float32)
        x = np.clip(x, 0, 255).astype(np.uint8)
        if p["jpeg"]:
            ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])
            x = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
        return x


def sample_geometry(rng: random.Random, cfg: dict | None, enabled: bool):
    """Flip + zoom-in crop aligned to the stride-4 label grid (same for a pair)."""
    cfg = cfg or {}
    if not enabled:
        return {"flip": False, "crop": None}
    flip = rng.random() < cfg.get("p_flip", 0.5)
    crop = None
    if rng.random() < cfg.get("p_crop", 0.5):
        s = rng.uniform(1.0, cfg.get("max_zoom", 1.25))
        hl, wl = int(round(LABEL_HW[0] / s)), int(round(LABEL_HW[1] / s))
        y0 = rng.randint(0, LABEL_HW[0] - hl)
        x0 = rng.randint(0, LABEL_HW[1] - wl)
        crop = (y0, x0, hl, wl)
    return {"flip": flip, "crop": crop}


def geo_image(img, g):
    if g["crop"] is not None:
        y0, x0, hl, wl = g["crop"]
        s = LABEL_STRIDE
        img = cv2.resize(img[y0 * s : (y0 + hl) * s, x0 * s : (x0 + wl) * s], (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_LINEAR)
    if g["flip"]:
        img = img[:, ::-1]
    return np.ascontiguousarray(img)


def geo_label(lab, g, interp=cv2.INTER_NEAREST):
    if g["crop"] is not None:
        y0, x0, hl, wl = g["crop"]
        lab = cv2.resize(lab[y0 : y0 + hl, x0 : x0 + wl], (LABEL_HW[1], LABEL_HW[0]), interpolation=interp)
    if g["flip"]:
        lab = lab[:, ::-1]
    return np.ascontiguousarray(lab)


def geo_flow(flow, g):
    """flow: 2,h,w in stride-4 px. Crop-zoom scales vectors; flip negates u."""
    u, v = flow[0], flow[1]
    if g["crop"] is not None:
        y0, x0, hl, wl = g["crop"]
        u = cv2.resize(u[y0 : y0 + hl, x0 : x0 + wl], (LABEL_HW[1], LABEL_HW[0]), interpolation=cv2.INTER_LINEAR) * (LABEL_HW[1] / wl)
        v = cv2.resize(v[y0 : y0 + hl, x0 : x0 + wl], (LABEL_HW[1], LABEL_HW[0]), interpolation=cv2.INTER_LINEAR) * (LABEL_HW[0] / hl)
    if g["flip"]:
        u, v = -u[:, ::-1], v[:, ::-1]
    return np.ascontiguousarray(np.stack([u, v]))


def geo_point(xy, g):
    """Normalized (x, y) through the same transform; None if it leaves range."""
    if xy is None:
        return None
    x, y = xy
    if g["crop"] is not None:
        y0, x0, hl, wl = g["crop"]
        x = (x * LABEL_HW[1] - x0) / wl
        y = (y * LABEL_HW[0] - y0) / hl
    if g["flip"]:
        x = 1.0 - x
    if not (-0.25 <= x <= 1.25 and 0.0 <= y <= 1.0):
        return None
    return x, y


def read_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr[:, :, ::-1]


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------


class StaticGeometryDataset(Dataset):
    def __init__(self, rows, augment: bool = False, aug_cfg: dict | None = None, label_cfg: dict | None = None, seed: int = 0):
        self.rows = rows
        self.augment = augment
        self.aug_cfg = aug_cfg or {}
        self.photo = Photometric(self.aug_cfg.get("photometric"))
        self.label_cfg = label_cfg or {}
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def targets(self, row):
        lc = self.label_cfg
        h, w = LABEL_HW
        pseudo_road = row.get("road_label", "pseudo") == "pseudo"
        t = {}
        rp = label_path(row, "road")
        if rp.is_file():
            z = np.load(rp)
            t["drivable"] = z["drivable"]
            t["lane"] = z["lane"]
            t["curb"] = z["curb"]
            base = lc.get("pseudo_road_weight", 0.7) if pseudo_road else 1.0
            if "drivable_conf" in z:
                t["drivable_w"] = z["drivable_conf"].astype(np.float32) / 255.0 * base
            else:
                t["drivable_w"] = np.full((h, w), base, np.float32)
            t["lanecurb_w"] = np.full((h, w), base, np.float32)
        else:
            for k in ("drivable", "lane", "curb"):
                t[k] = np.full((h, w), IGNORE, np.uint8)
            t["drivable_w"] = np.zeros((h, w), np.float32)
            t["lanecurb_w"] = np.zeros((h, w), np.float32)
        op = label_path(row, "objects")
        if op.is_file():
            z = np.load(op)
            lut = np.where(z["ok"] > 0, z["cls"], IGNORE).astype(np.uint8)
            lut[0] = 0
            t["objects"] = lut[z["inst"][1::2, 1::2]]  # 224x400 -> 112x200
            human = row["source"] == "bdd100k"
            t["objects_w"] = np.full((h, w), 1.0 if human else lc.get("pseudo_object_weight", 0.7), np.float32)
        else:
            t["objects"] = np.full((h, w), IGNORE, np.uint8)
            t["objects_w"] = np.zeros((h, w), np.float32)
        dp = label_path(row, "depth")
        if dp.is_file():
            z = np.load(dp)
            t["depth"] = z["disp"].astype(np.float32) / 65535.0
            conf = z["conf"].astype(np.float32) / 255.0
            t["depth_w"] = np.where(conf >= lc.get("depth_min_conf", 0.3), conf, 0.0).astype(np.float32)
        else:
            t["depth"] = np.zeros((h, w), np.float32)
            t["depth_w"] = np.zeros((h, w), np.float32)
        cp = label_path(row, "contact")
        if cp.is_file():
            t["contact"] = np.load(cp)["contact"]
            human = row["source"] == "bdd100k"
            t["contact_w"] = np.full((h, w), 1.0 if human else lc.get("pseudo_object_weight", 0.7), np.float32)
        else:
            t["contact"] = np.full((h, w), IGNORE, np.uint8)
            t["contact_w"] = np.zeros((h, w), np.float32)
        vp = None
        if row.get("meta") and row.get("has_vp"):
            m = json.loads(Path(row["meta"]).read_text())
            if m.get("vp"):
                vp = (m["vp"]["x"], m["vp"]["y"])
        return t, vp

    def __getitem__(self, i):
        row = self.rows[i]
        rng = random.Random((self.seed, i, torch.initial_seed()).__hash__()) if self.augment else random.Random(i)
        img = read_rgb(row["image"])
        t, vp = self.targets(row)
        g = sample_geometry(rng, self.aug_cfg.get("geometric"), self.augment)
        img = geo_image(img, g)
        if self.augment:
            img = self.photo.apply(img, self.photo.params(rng), np.random.default_rng(rng.randrange(1 << 30)))
        out = {"image": torch.from_numpy(img).permute(2, 0, 1)}
        for k, v in t.items():
            interp = cv2.INTER_LINEAR if k in ("depth",) else cv2.INTER_NEAREST
            arr = geo_label(v, g, interp)
            if arr.dtype == np.uint8:
                out[k] = torch.from_numpy(arr).long()
            else:
                out[k] = torch.from_numpy(arr.astype(np.float32))
        vp = geo_point(vp, g)
        out["vp"] = torch.tensor(vp if vp is not None else (0.0, 0.0), dtype=torch.float32)
        out["vp_valid"] = torch.tensor(vp is not None)
        out["source"] = torch.tensor(SOURCE_IDS[row["source"]])
        out["index"] = torch.tensor(i)
        return out


class PairFlowDataset(Dataset):
    def __init__(self, pairs, augment=False, aug_cfg=None, min_consistent: float = 0.3, min_conf: float = 0.2, seed: int = 0):
        self.pairs = pairs
        self.augment = augment
        self.aug_cfg = aug_cfg or {}
        self.photo = Photometric(self.aug_cfg.get("photometric"))
        self.min_consistent = min_consistent
        self.min_conf = min_conf
        self.seed = seed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        rng = random.Random((self.seed, i, torch.initial_seed()).__hash__()) if self.augment else random.Random(i)
        a, b = read_rgb(p["image_t"]), read_rgb(p["image_tplus"])
        fp = flow_path(p)
        if fp.is_file():
            z = np.load(fp)
            flow = z["flow"].astype(np.float32)
            w = z["conf"].astype(np.float32) / 255.0
            w = np.where(w >= self.min_conf, w, 0.0)
            if float(z["consistent_frac"]) < self.min_consistent:
                w[:] = 0.0  # whole pair rejected (blur, crash, occlusion)
        else:
            flow = np.zeros((2, *LABEL_HW), np.float32)
            w = np.zeros(LABEL_HW, np.float32)
        g = sample_geometry(rng, self.aug_cfg.get("geometric"), self.augment)
        a, b = geo_image(a, g), geo_image(b, g)
        flow = geo_flow(flow, g)
        w = geo_label(w.astype(np.float32), g)
        if self.augment:
            params = self.photo.params(rng)  # shared photometric params; noise differs per frame
            nrng = np.random.default_rng(rng.randrange(1 << 30))
            a, b = self.photo.apply(a, params, nrng), self.photo.apply(b, params, nrng)
        return {
            "image_t": torch.from_numpy(a).permute(2, 0, 1),
            "image_tplus": torch.from_numpy(b).permute(2, 0, 1),
            "flow": torch.from_numpy(flow),
            "flow_w": torch.from_numpy(w.astype(np.float32)),
            "dt": torch.tensor(float(p["dt"])),
            "source": torch.tensor(SOURCE_IDS[p["source"]]),
        }


class SourceWeightedSampler(Sampler):
    """Infinite-style sampler drawing sources by weight, then uniformly within a source.

    Absent sources are dropped and the remaining weights renormalized, so a
    very large source cannot overwhelm the others.
    """

    def __init__(self, rows, weights: dict, num_samples: int, seed: int = 0):
        self.by_source = {}
        for i, r in enumerate(rows):
            self.by_source.setdefault(r["source"], []).append(i)
        w = {s: float(weights.get(s, 0.0)) for s in self.by_source}
        w = {s: v for s, v in w.items() if v > 0}
        total = sum(w.values())
        if total <= 0:
            raise ValueError("No source has positive weight")
        self.weights = {s: v / total for s, v in w.items()}
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = np.random.default_rng((self.seed, self.epoch))
        sources = list(self.weights)
        probs = np.array([self.weights[s] for s in sources])
        picks = g.choice(len(sources), size=self.num_samples, p=probs)
        for k in picks:
            idx = self.by_source[sources[k]]
            yield idx[int(g.integers(len(idx)))]


def infinite(loader):
    epoch = 0
    while True:
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1
