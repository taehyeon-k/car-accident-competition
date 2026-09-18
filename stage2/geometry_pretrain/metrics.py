"""Held-out geometry metrics, accumulated per source over a validation loader."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from stage2.geometry_pretrain.common import IGNORE, INPUT_HW, LABEL_HW, OBJECT_CLASSES
from stage2.geometry_pretrain.datasets import SOURCE_IDS

ID_TO_SOURCE = {v: k for k, v in SOURCE_IDS.items()}


def _dilate(x, r):
    return F.max_pool2d(x[:, None].float(), 2 * r + 1, 1, r)[:, 0] > 0


class GeometryMeter:
    """Streaming counts; ``compute`` returns flat ``{source/metric: value}``."""

    def __init__(self, tol: int = 1):
        self.tol = tol
        self.c = defaultdict(float)

    def _add(self, src, name, value):
        self.c[f"{src}|{name}"] += float(value)

    # -- road ---------------------------------------------------------------
    def update_road(self, logits, batch):
        pred_drv = logits[:, :3].argmax(1)
        tgt = batch["drivable"]
        for b in range(len(tgt)):
            src = ID_TO_SOURCE[int(batch["source"][b])]
            valid = tgt[b] != IGNORE
            if valid.sum() == 0:
                continue
            for c in range(3):
                p, t = (pred_drv[b] == c) & valid, (tgt[b] == c) & valid
                self._add(src, f"drv_inter_{c}", (p & t).sum())
                self._add(src, f"drv_union_{c}", (p | t).sum())
                self._add(src, f"drv_tgt_{c}", t.sum())
            self._add(src, "drv_images", 1)
        for key, ch in (("lane", 3), ("curb", 4)):
            pred = logits[:, ch] > 0
            tgt = batch[key]
            for b in range(len(tgt)):
                src = ID_TO_SOURCE[int(batch["source"][b])]
                valid = tgt[b] != IGNORE
                if valid.sum() == 0 or (key == "curb" and src == "tusimple"):
                    continue
                t = (tgt[b] == 1) & valid
                p = pred[b] & valid
                self._add(src, f"{key}_inter", (p & t).sum())
                self._add(src, f"{key}_union", (p | t).sum())
                # Boundary F1 with a +-tol pixel tolerance at stride 4.
                self._add(src, f"{key}_tp_p", (p & _dilate(t[None], self.tol)[0]).sum())
                self._add(src, f"{key}_np", p.sum())
                self._add(src, f"{key}_tp_r", (t & _dilate(p[None], self.tol)[0]).sum())
                self._add(src, f"{key}_nt", t.sum())

    # -- objects ------------------------------------------------------------
    def update_objects(self, logits, batch):
        pred = logits.argmax(1)
        tgt = batch["objects"]
        for b in range(len(tgt)):
            src = ID_TO_SOURCE[int(batch["source"][b])]
            valid = tgt[b] != IGNORE
            if batch["objects_w"][b].max() == 0:
                continue
            for c in range(len(OBJECT_CLASSES)):
                p, t = (pred[b] == c) & valid, (tgt[b] == c) & valid
                self._add(src, f"obj_inter_{c}", (p & t).sum())
                self._add(src, f"obj_union_{c}", (p | t).sum())
                self._add(src, f"obj_tgt_{c}", t.sum())
            pv = (pred[b] >= 1) & (pred[b] <= 4) & valid
            tv = (tgt[b] >= 1) & (tgt[b] <= 4) & valid
            self._add(src, "veh_inter", (pv & tv).sum())
            self._add(src, "veh_union", (pv | tv).sum())

    # -- contact ------------------------------------------------------------
    def update_contact(self, logits, batch):
        pred = logits[:, 0] > 0
        tgt = batch["contact"]
        for b in range(len(tgt)):
            src = ID_TO_SOURCE[int(batch["source"][b])]
            if batch["contact_w"][b].max() == 0:
                continue
            valid = tgt[b] != IGNORE
            t = (tgt[b] == 1) & valid
            p = pred[b] & valid
            self._add(src, "contact_inter", (p & t).sum())
            self._add(src, "contact_union", (p | t).sum())
            self._add(src, "contact_tp_p", (p & _dilate(t[None], self.tol)[0]).sum())
            self._add(src, "contact_np", p.sum())
            self._add(src, "contact_tp_r", (t & _dilate(p[None], self.tol)[0]).sum())
            self._add(src, "contact_nt", t.sum())
            # Localization: mean distance of target contact rows to nearest predicted row per column.
            tc = t.any(0)
            if tc.any():
                cols = torch.nonzero(tc)[:, 0]
                ty = torch.stack([torch.nonzero(t[:, c])[:, 0].float().mean() for c in cols])
                pc = p[:, cols]
                has = pc.any(0)
                if has.any():
                    ar = torch.arange(p.shape[0], device=p.device).float()[:, None]
                    py = (pc.float() * ar).sum(0) / pc.float().sum(0).clamp_min(1)
                    self._add(src, "contact_loc_err_px", ((py - ty).abs()[has] * 4).sum())
                    self._add(src, "contact_loc_n", has.sum())
                self._add(src, "contact_loc_cols", len(cols))
                self._add(src, "contact_loc_found", has.sum())

    # -- depth --------------------------------------------------------------
    def update_depth(self, pred, batch, n_pairs: int = 2000):
        tgt, w = batch["depth"], batch["depth_w"]
        for b in range(len(tgt)):
            src = ID_TO_SOURCE[int(batch["source"][b])]
            # Far range / sky (normalized disparity < 0.05) is excluded, like a depth cap.
            m = (w[b] > 0.3) & (tgt[b] >= 0.05)
            if m.sum() < 100:
                continue
            p, t = pred[b][m].float(), tgt[b][m].float()
            # Least-squares scale/shift alignment in (relative inverse) depth space.
            A = torch.stack([p, torch.ones_like(p)], 1)
            sol = torch.linalg.lstsq(A, t[:, None]).solution[:, 0]
            pa = (A @ sol).clamp_min(1e-3)
            tt = t.clamp_min(1e-3)
            self._add(src, "depth_absrel", ((pa - tt).abs() / tt).mean())
            ratio = torch.maximum(pa / tt, tt / pa)
            self._add(src, "depth_delta1", (ratio < 1.25).float().mean())
            d = torch.log(pa) - torch.log(tt)
            self._add(src, "depth_silog", torch.sqrt((d**2).mean() - d.mean() ** 2).clamp_min(0) * 100)
            # Ordinal (rank) consistency on random point pairs with a clear gap.
            g = torch.Generator(device="cpu").manual_seed(b)
            i = torch.randint(len(t), (n_pairs,), generator=g).to(t.device)
            j = torch.randint(len(t), (n_pairs,), generator=g).to(t.device)
            gap = (t[i] - t[j]).abs() > 0.1 * (t[i] + t[j]) / 2
            if gap.sum() > 10:
                agree = torch.sign(p[i] - p[j]) == torch.sign(t[i] - t[j])
                self._add(src, "depth_ordinal", agree[gap].float().mean())
            self._add(src, "depth_images", 1)

    # -- camera -------------------------------------------------------------
    def update_camera(self, pred, batch):
        for b in range(len(pred)):
            if not bool(batch["vp_valid"][b]):
                continue
            src = ID_TO_SOURCE[int(batch["source"][b])]
            dx = (pred[b, 0] - batch["vp"][b, 0]).abs() * INPUT_HW[1]
            dy = (pred[b, 1] - batch["vp"][b, 1]).abs() * INPUT_HW[0]
            self._add(src, "vp_err_px", torch.sqrt(dx**2 + dy**2))
            self._add(src, "horizon_err_px", dy)
            self._add(src, "vp_n", 1)

    # -- flow ---------------------------------------------------------------
    def update_flow(self, pred, batch):
        tgt, w = batch["flow"], batch["flow_w"]
        for b in range(len(tgt)):
            src = ID_TO_SOURCE[int(batch["source"][b])]
            m = w[b] > 0.3
            if m.sum() < 50:
                continue
            epe = torch.linalg.norm(pred[b] - tgt[b], dim=0)[m] * 4  # input-resolution px
            mag = torch.linalg.norm(tgt[b], dim=0)[m] * 4
            dt = float(batch["dt"][b])
            bucket = "dt_s" if dt < 0.1 else ("dt_m" if dt < 0.3 else "dt_l")
            for tag in ("", f"_{bucket}"):
                self._add(src, f"flow_epe{tag}", epe.mean())
                self._add(src, f"flow_acc3{tag}", (epe < 3).float().mean())
                self._add(src, f"flow_n{tag}", 1)
            self._add(src, "flow_mag", mag.mean())

    # -- compute ------------------------------------------------------------
    def compute(self) -> dict:
        per = defaultdict(dict)
        for k, v in self.c.items():
            src, name = k.split("|")
            per[src][name] = v
        out = {}
        for src, c in per.items():
            g = lambda n: c.get(n, 0.0)
            if g("drv_images"):
                ious = [g(f"drv_inter_{k}") / g(f"drv_union_{k}") for k in range(3) if g(f"drv_tgt_{k}") > 0]
                if ious and src != "tusimple":
                    out[f"{src}/drivable_miou"] = float(np.mean(ious))
                if g("drv_union_1"):
                    out[f"{src}/ego_lane_iou"] = g("drv_inter_1") / g("drv_union_1")
                if g("drv_tgt_1"):
                    out[f"{src}/ego_lane_recall"] = g("drv_inter_1") / g("drv_tgt_1")
            for key in ("lane", "curb", "contact"):
                if g(f"{key}_nt") > 0 or g(f"{key}_np") > 0:
                    prec = g(f"{key}_tp_p") / max(g(f"{key}_np"), 1)
                    rec = g(f"{key}_tp_r") / max(g(f"{key}_nt"), 1)
                    out[f"{src}/{key}_iou"] = g(f"{key}_inter") / max(g(f"{key}_union"), 1)
                    out[f"{src}/{key}_bf1"] = 2 * prec * rec / max(prec + rec, 1e-9)
                    out[f"{src}/{key}_bprec"] = prec
                    out[f"{src}/{key}_brec"] = rec
            if g("contact_loc_n"):
                out[f"{src}/contact_loc_err_px"] = g("contact_loc_err_px") / g("contact_loc_n")
                out[f"{src}/contact_col_recall"] = g("contact_loc_found") / max(g("contact_loc_cols"), 1)
            ious = [g(f"obj_inter_{k}") / g(f"obj_union_{k}") for k in range(len(OBJECT_CLASSES)) if g(f"obj_tgt_{k}") > 0]
            if ious:
                out[f"{src}/object_miou"] = float(np.mean(ious))
                out[f"{src}/vehicle_iou"] = g("veh_inter") / max(g("veh_union"), 1)
                for k in range(1, len(OBJECT_CLASSES)):
                    if g(f"obj_tgt_{k}") > 0:
                        out[f"{src}/iou_{OBJECT_CLASSES[k]}"] = g(f"obj_inter_{k}") / g(f"obj_union_{k}")
            if g("depth_images"):
                n = g("depth_images")
                for m in ("absrel", "delta1", "silog", "ordinal"):
                    out[f"{src}/depth_{m}"] = g(f"depth_{m}") / n
            if g("vp_n"):
                out[f"{src}/vp_err_px"] = g("vp_err_px") / g("vp_n")
                out[f"{src}/horizon_err_px"] = g("horizon_err_px") / g("vp_n")
            for tag in ("", "_dt_s", "_dt_m", "_dt_l"):
                if g(f"flow_n{tag}"):
                    out[f"{src}/flow_epe{tag}"] = g(f"flow_epe{tag}") / g(f"flow_n{tag}")
                    out[f"{src}/flow_acc3{tag}"] = g(f"flow_acc3{tag}") / g(f"flow_n{tag}")
        return dict(sorted(out.items()))


def geometry_score(m: dict) -> float:
    """Model-selection score in [0,1]; human-labelled metrics dominate.

    Uses held-out human labels (BDD drivable/lane/curb, TuSimple lanes), SAM
    masks from human boxes (BDD objects/contact), and teacher agreement for
    depth/flow on all sources.
    """
    terms = []
    for k in ("bdd100k/drivable_miou", "bdd100k/lane_bf1", "bdd100k/curb_bf1", "tusimple/lane_bf1",
              "tusimple/ego_lane_recall", "bdd100k/object_miou", "bdd100k/contact_bf1"):
        if k in m:
            terms.append(m[k])
    for src in ("bdd100k", "accident", "tusimple", "baton"):
        if f"{src}/depth_delta1" in m:
            terms.append(m[f"{src}/depth_delta1"])
        if f"{src}/flow_acc3" in m:
            terms.append(m[f"{src}/flow_acc3"])
    return float(np.mean(terms)) if terms else 0.0
