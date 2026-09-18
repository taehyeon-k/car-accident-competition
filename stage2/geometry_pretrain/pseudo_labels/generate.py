"""Cached, resumable pseudo-label generation (separate from training).

Stages (``--stage``):
  static   detector boxes (non-BDD) -> SAM 2.1 masks -> instance map + acceptance,
           Depth Anything V2 disparity + flip-consistency confidence
  road     road-teacher (DINOv3 ViT-B/16 trained on BDD human labels) drivable /
           lane / curb pseudo labels for footage without human road labels
  flow     SEA-RAFT forward/backward flow for every *manifest* pair only,
           stored at stride 4 in float16 with a consistency/confidence byte
  contact  derived vehicle-road contact bands (CPU) from accepted vehicle masks
           plus road labels
  summary  cache manifest with sizes and acceptance statistics

Every file is written atomically and skipped if present, so any stage can be
interrupted and resumed. The disk guard aborts before the safety margin.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from stage2.geometry_pretrain.common import (
    IGNORE,
    INPUT_HW,
    LABEL_HW,
    VEHICLE_CLASSES,
    DiskGuard,
    atomic_json,
    atomic_savez,
    dir_size_gb,
    read_jsonl,
)

CACHE = Path("/workspace/cache/geometry_pretrain")
INST_HW = (INPUT_HW[0] // 2, INPUT_HW[1] // 2)  # 224 x 400 instance maps

SOURCES = ("bdd100k", "tusimple", "accident", "baton")


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------


def key_of(row_id: str) -> str:
    source, rest = row_id.split("/", 1)
    return rest.replace("/", "__")


def label_path(row: dict, kind: str, cache: Path = CACHE) -> Path:
    if kind == "road" and row.get("road"):
        return Path(row["road"])
    return cache / "labels" / row["source"] / f"{key_of(row['id'])}.{kind}.npz"


def flow_path(pair: dict, cache: Path = CACHE) -> Path:
    return cache / "labels" / pair["source"] / "pairs" / f"{key_of(pair['id'])}.flow.npz"


def load_frames(manifest_dir: Path, sources=SOURCES, splits=("train", "val")) -> list[dict]:
    rows = []
    for s in sources:
        for sp in splits:
            p = manifest_dir / f"{s}_{sp}.jsonl"
            if p.is_file():
                rows += read_jsonl(p)
    return rows


def load_pairs(manifest_dir: Path, sources=SOURCES, splits=("train", "val")) -> list[dict]:
    rows = []
    for s in sources:
        for sp in splits:
            p = manifest_dir / f"{s}_{sp}_pairs.jsonl"
            if p.is_file():
                rows += read_jsonl(p)
    return rows


class ImageRows(Dataset):
    def __init__(self, rows, keys=("image",)):
        self.rows, self.keys = rows, keys

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        out = []
        for k in self.keys:
            bgr = cv2.imread(self.rows[i][k])
            if bgr is None or bgr.shape[:2] != INPUT_HW:
                raise ValueError(f"bad image {self.rows[i][k]}")
            out.append(torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).permute(2, 0, 1))
        return i, out


def collate(batch):
    idx = [b[0] for b in batch]
    n = len(batch[0][1])
    return idx, [torch.stack([b[1][k] for b in batch]) for k in range(n)]


# ----------------------------------------------------------------------------
# Static pass: objects + depth
# ----------------------------------------------------------------------------


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def accept_mask(box, iou_score: float, tight, area: float, human: bool, cfg) -> tuple[bool, str]:
    x1, y1, x2, y2 = box
    if min(x2 - x1, y2 - y1) < cfg["min_box_side"]:
        return False, "tiny"
    if iou_score < (cfg["sam_iou_human"] if human else cfg["sam_iou_det"]):
        return False, "sam_score"
    if area <= 0:
        return False, "empty"
    if box_iou(tight, box) < cfg["box_consistency"]:
        return False, "box_inconsistent"
    if area / max((x2 - x1) * (y2 - y1), 1) < cfg["min_fill"]:
        return False, "low_fill"
    return True, "ok"


def build_instances(boxes, seg, cfg):
    """Paint far-to-near (by box bottom) into a 224x400 instance map."""
    masks, ious, tight, area = seg
    order = sorted(range(len(boxes)), key=lambda i: boxes[i]["box"][3])[:254]
    inst = np.zeros(INST_HW, np.uint8)
    cls, ok, info = [0], [0], []
    reasons = Counter()
    masks = masks.numpy()
    for new_id, i in enumerate(order, start=1):
        b = boxes[i]
        human = b.get("source") == "bdd_human"
        good, why = accept_mask(b["box"], float(ious[i]), tight[i].tolist(), float(area[i]), human, cfg)
        reasons[why] += 1
        if good:
            inst[masks[i]] = new_id
        else:
            x1, y1, x2, y2 = (np.asarray(b["box"]) / 2).round().astype(int)
            inst[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)] = new_id
        cls.append(b["cls"])
        ok.append(int(good))
        info.append([*(float(v) for v in b["box"]), float(b.get("score", 1.0)), float(ious[i]),
                     float(b.get("truncated", False)), float(human)])
    return inst, np.asarray(cls, np.uint8), np.asarray(ok, np.uint8), np.asarray(info, np.float32).reshape(-1, 8), reasons


def run_static(rows, args, cfg):
    from stage2.geometry_pretrain.pseudo_labels.teachers import BoxSegmenter, DepthTeacher, Detector

    todo = [
        r for r in rows
        if not (label_path(r, "objects").is_file() and label_path(r, "depth").is_file())
    ]
    print(f"static: {len(todo)}/{len(rows)} frames to process")
    if not todo:
        return
    guard = DiskGuard(CACHE, args.min_free_gb)
    detector = Detector(score_threshold=cfg["det_score"]) if any(r["source"] != "bdd100k" for r in todo) else None
    sam = BoxSegmenter()
    depth = DepthTeacher()
    loader = DataLoader(ImageRows(todo), batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)
    writer = ThreadPoolExecutor(max_workers=3)
    pending = []
    for step, (idx, (imgs,)) in enumerate(tqdm(loader, desc="static")):
        if step % 200 == 0:
            guard.check("during static pseudo-labeling")
        imgs = imgs.cuda(non_blocking=True)
        batch_rows = [todo[i] for i in idx]
        boxes = [None] * len(idx)
        det_ids = [k for k, r in enumerate(batch_rows) if r["source"] != "bdd100k"]
        if det_ids:
            dets = detector(imgs[det_ids])
            for k, d in zip(det_ids, dets):
                boxes[k] = d
        for k, r in enumerate(batch_rows):
            if boxes[k] is None:
                boxes[k] = json.loads(Path(r["meta"]).read_text())["boxes"]
        seg = sam(imgs, [[b["box"] for b in bx] for bx in boxes])
        disp, conf, rel = depth(imgs)
        q_all = []
        for k in range(len(batch_rows)):
            d = disp[k]
            lo, hi = torch.quantile(d.flatten(), 0.001), torch.quantile(d.flatten(), 0.999)
            q_all.append(((d.clamp(lo, hi) - lo) / (hi - lo).clamp_min(1e-6) * 65535).round().to(torch.int32).cpu().numpy().astype(np.uint16))
        conf_u8 = (conf * 255).round().byte().cpu().numpy()
        rel = rel.cpu().numpy()
        pending.append(writer.submit(_write_static, batch_rows, boxes, seg, q_all, conf_u8, rel, cfg))
        while len(pending) > 6:  # bound host memory; re-raise writer errors
            pending.pop(0).result()
    for f in pending:
        f.result()
    writer.shutdown(wait=True)


def _write_static(batch_rows, boxes, seg, q_all, conf_u8, rel, cfg):
    for k, r in enumerate(batch_rows):
        inst, cls, ok, info, _ = build_instances(boxes[k], seg[k], cfg)
        atomic_savez(label_path(r, "objects"), inst=inst, cls=cls, ok=ok, info=info,
                     teacher=np.asarray("sam2.1-hiera-small+" + ("bdd_human_boxes" if r["source"] == "bdd100k" else "rfdetr-small")))
        atomic_savez(label_path(r, "depth"), disp=q_all[k], conf=conf_u8[k],
                     flip_rel_err=np.float32(rel[k]), teacher=np.asarray("depth-anything-v2-small"))


# ----------------------------------------------------------------------------
# Road-teacher pass
# ----------------------------------------------------------------------------


def run_road(rows, args, cfg):
    from stage2.geometry_pretrain.models.geometry_dino import GeometryDINO, DinoBackbone
    from stage2.geometry_pretrain.pseudo_labels.teachers import normalize

    todo = [r for r in rows if not r.get("road") and not label_path(r, "road").is_file()]
    print(f"road: {len(todo)}/{len(rows)} frames to process")
    if not todo:
        return
    ckpt = torch.load(args.road_teacher, map_location="cpu", weights_only=False)
    tcfg = ckpt["config"]["model"]
    backbone = DinoBackbone(tcfg["arch"], None, trainable_blocks=0)
    model = GeometryDINO(backbone, ["road"], tcfg.get("head_dim", 128))
    model.load_state_dict(ckpt["model"], strict=True)
    model.cuda().eval()
    loader = DataLoader(ImageRows(todo), batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)
    t_drv, t_lane, t_curb = cfg["road_conf_drivable"], cfg["road_conf_lane"], cfg["road_conf_curb"]
    for idx, (imgs,) in tqdm(loader, desc="road"):
        x = normalize(imgs.cuda())
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.forward_static(x, ["road"])["road"].float()
            logits_f = model.forward_static(x.flip(-1), ["road"])["road"].float().flip(-1)
        p_drv = 0.5 * (logits[:, :3].softmax(1) + logits_f[:, :3].softmax(1))
        p_lane = 0.5 * (torch.sigmoid(logits[:, 3]) + torch.sigmoid(logits_f[:, 3]))
        p_curb = 0.5 * (torch.sigmoid(logits[:, 4]) + torch.sigmoid(logits_f[:, 4]))
        conf_drv, drv = p_drv.max(1)
        for k, i in enumerate(idx):
            d = drv[k].byte().cpu().numpy()
            cd = conf_drv[k].cpu().numpy()
            d[cd < t_drv] = IGNORE
            lane = (p_lane[k] > 0.5).byte().cpu().numpy()
            pl = p_lane[k].cpu().numpy()
            lane[(pl > 1 - t_lane) & (pl < t_lane)] = IGNORE  # uncertain band
            curb = (p_curb[k] > 0.5).byte().cpu().numpy()
            pc = p_curb[k].cpu().numpy()
            curb[(pc > 1 - t_curb) & (pc < t_curb)] = IGNORE
            atomic_savez(
                label_path(todo[i], "road"), drivable=d, lane=lane, curb=curb,
                drivable_conf=(cd * 255).round().astype(np.uint8),
                teacher=np.asarray("road_teacher_dinov3_vitb16_bdd"),
            )


# ----------------------------------------------------------------------------
# Flow pass
# ----------------------------------------------------------------------------


def run_flow(pairs, args, cfg):
    from stage2.geometry_pretrain.pseudo_labels.teachers import FlowTeacher, fb_consistency

    todo = [p for p in pairs if not flow_path(p).is_file()]
    print(f"flow: {len(todo)}/{len(pairs)} pairs to process")
    if not todo:
        return
    guard = DiskGuard(CACHE, args.min_free_gb)
    teacher = FlowTeacher(batch_size=args.batch_size)
    loader = DataLoader(ImageRows(todo, ("image_t", "image_tplus")), batch_size=args.batch_size,
                        num_workers=args.workers, collate_fn=collate)
    ih, iw = teacher.infer_hw
    for step, (idx, (a, b)) in enumerate(tqdm(loader, desc="flow")):
        if step % 200 == 0:
            guard.check("during flow pseudo-labeling")
        f_fw, f_bw, c_fw, _ = teacher(a.cuda(), b.cuda())
        consistent = fb_consistency(f_fw, f_bw).float()
        conf = c_fw.clamp(0, 1) * consistent
        # Stride-4 storage: area-resample vectors, rescale to stride-4 px units.
        flow_lr = F.interpolate(f_fw, LABEL_HW, mode="area")
        flow_lr[:, 0] *= LABEL_HW[1] / iw
        flow_lr[:, 1] *= LABEL_HW[0] / ih
        conf_lr = F.interpolate(conf[:, None], LABEL_HW, mode="area")[:, 0]
        cons_lr = F.interpolate(consistent[:, None], LABEL_HW, mode="area")[:, 0]
        for k, i in enumerate(idx):
            atomic_savez(
                flow_path(todo[i]),
                flow=flow_lr[k].half().cpu().numpy(),
                conf=(conf_lr[k] * 255).round().byte().cpu().numpy(),
                consistent_frac=np.float32(cons_lr[k].mean().item()),
                teacher=np.asarray(f"sea-raft-s(spring-S)@{ih}x{iw} fw/bw"),
            )


# ----------------------------------------------------------------------------
# Contact derivation (CPU)
# ----------------------------------------------------------------------------


def derive_contact(row, cfg):
    out = label_path(row, "contact")
    if out.is_file():
        return None
    obj_p, road_p = label_path(row, "objects"), label_path(row, "road")
    if not obj_p.is_file():
        return "no_objects"
    obj = np.load(obj_p)
    inst, cls, ok, info = obj["inst"], obj["cls"], obj["ok"], obj["info"]
    drivable = np.load(road_p)["drivable"] if road_p.is_file() else np.full(LABEL_HW, IGNORE, np.uint8)
    road_up = cv2.resize(drivable, (INST_HW[1], INST_HW[0]), interpolation=cv2.INTER_NEAREST)
    contact = np.zeros(INST_HW, np.uint8)
    ignore = np.zeros(INST_HW, bool)
    stats = Counter()
    Hh, Wh = INST_HW
    for iid in range(1, len(cls)):
        if cls[iid] not in VEHICLE_CLASSES:
            continue
        m = inst == iid
        if not m.any():
            continue
        x1, y1, x2, y2 = (info[iid - 1][:4] / 2)
        bh = y2 - y1
        lower_region = (slice(int(max(y1 + 0.5 * bh, 0)), int(min(y2 + 0.1 * bh + 2, Hh))), slice(int(max(x1, 0)), int(min(x2, Wh))))
        reason = None
        if not ok[iid]:
            reason = "mask_rejected"
        elif bh < cfg["contact_min_height"]:
            reason = "too_small"
        elif info[iid - 1][6] > 0 or y2 >= Hh - 2 or x1 <= 1 or x2 >= Wh - 2:
            reason = "truncated"
        if reason is None:
            cols = np.nonzero(m.any(0))[0]
            bottoms = np.array([np.nonzero(m[:, c])[0].max() for c in cols])
            low = bottoms >= (y2 - cfg["contact_lower_frac"] * bh)
            cols, bottoms = cols[low], bottoms[low]
            # Road support: pixel(s) just below the contour must be road / not another object.
            below_y = np.clip(bottoms + 2, 0, Hh - 1)
            other = (inst[below_y, cols] != 0) & (inst[below_y, cols] != iid)  # occluded from below
            win = cfg["contact_road_window"]
            road_ok = np.zeros(len(cols), bool)
            unknown = np.zeros(len(cols), bool)
            for k in range(1, win + 1):
                yy = np.clip(bottoms + k, 0, Hh - 1)
                v = road_up[yy, cols]
                road_ok |= np.isin(v, (1, 2))
                unknown |= v == IGNORE
            support = (road_ok | (unknown & cfg["contact_allow_unknown_road"])) & ~other
            if len(cols) < 3 or support.mean() < cfg["contact_min_support"]:
                reason = "no_road_support"
            else:
                t = max(1, int(round(cfg["contact_band_frac"] * bh)))
                for c, yb, sp in zip(cols, bottoms, support):
                    if sp:
                        contact[max(yb - t, 0) : min(yb + t + 1, Hh), c] = 1
                stats["accepted"] += 1
                continue
        stats[reason] += 1
        ignore[lower_region] = True
    small_c = (cv2.resize(contact.astype(np.float32), (LABEL_HW[1], LABEL_HW[0]), interpolation=cv2.INTER_AREA) > 0.2).astype(np.uint8)
    small_i = cv2.resize(ignore.astype(np.uint8), (LABEL_HW[1], LABEL_HW[0]), interpolation=cv2.INTER_NEAREST) > 0
    target = small_c.copy()
    target[small_i & (small_c == 0)] = IGNORE
    atomic_savez(out, contact=target)
    return dict(stats)


def _contact_worker(task):
    row, cfg = task
    return row["source"], derive_contact(row, cfg)


def run_contact(rows, args, cfg):
    totals = defaultdict(Counter)
    with Pool(args.workers) as pool:
        for src, st in tqdm(pool.imap_unordered(_contact_worker, [(r, cfg) for r in rows], chunksize=64), total=len(rows), desc="contact"):
            if isinstance(st, dict):
                totals[src].update(st)
            elif st:
                totals[src][st] += 1
    atomic_json(CACHE / "stats" / f"contact_stats_{int(time.time())}.json", {k: dict(v) for k, v in totals.items()})
    print({k: dict(v) for k, v in totals.items()})


# ----------------------------------------------------------------------------
# Cache manifest / statistics
# ----------------------------------------------------------------------------


def run_summary(rows, pairs, args, cfg):
    per = defaultdict(lambda: defaultdict(Counter))
    for r in tqdm(rows, desc="summary-frames"):
        c = per[r["source"]][r["split"]]
        c["frames"] += 1
        for kind in ("road", "objects", "depth", "contact"):
            c[f"has_{kind}"] += int(label_path(r, kind).is_file())
        p = label_path(r, "objects")
        if p.is_file():
            o = np.load(p)
            c["instances"] += len(o["cls"]) - 1
            c["instances_accepted"] += int(o["ok"][1:].sum())
        p = label_path(r, "depth")
        if p.is_file():
            c["depth_conf_mean_x1000"] += int(np.load(p)["conf"].mean() / 255 * 1000)
    for p in tqdm(pairs, desc="summary-pairs"):
        c = per[p["source"]][p["split"]]
        c["pairs"] += 1
        fp = flow_path(p)
        if fp.is_file():
            z = np.load(fp)
            c["has_flow"] += 1
            frac = float(z["consistent_frac"])
            c["flow_pairs_accepted"] += int(frac >= cfg["flow_min_consistent"])
    out = {s: {sp: dict(c) for sp, c in v.items()} for s, v in per.items()}
    sizes = {s: dir_size_gb(CACHE / "labels" / s) for s in SOURCES if (CACHE / "labels" / s).exists()}
    atomic_json(CACHE / "cache_manifest.json", {"counts": out, "label_cache_gb": sizes, "config": cfg, "time": time.ctime()})
    print(json.dumps({"counts": out, "label_cache_gb": sizes}, indent=1))


DEFAULT_CFG = {
    "det_score": 0.5,
    "min_box_side": 6,
    "sam_iou_human": 0.70,
    "sam_iou_det": 0.80,
    "box_consistency": 0.70,
    "min_fill": 0.25,
    "road_conf_drivable": 0.70,
    "road_conf_lane": 0.70,
    "road_conf_curb": 0.70,
    "contact_min_height": 8,  # px at 224x400
    "contact_lower_frac": 0.25,
    "contact_band_frac": 0.04,
    "contact_min_support": 0.3,
    "contact_road_window": 10,  # rows at 224x400 (~20 px at 448x800) below the contour
    "contact_allow_unknown_road": True,  # masked (unknown) road does not veto; background does
    "flow_min_consistent": 0.3,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["static", "road", "flow", "contact", "summary"])
    parser.add_argument("--manifest-dir", default=str(CACHE / "manifests"))
    parser.add_argument("--sources", nargs="+", default=list(SOURCES))
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--min-free-gb", type=float, default=12.0)
    parser.add_argument("--road-teacher", default="/workspace/outputs/geometry_pretrain/road_teacher/best.pt")
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--gpu-mem-fraction", type=float, default=None, help="cap this process on a shared GPU")
    args = parser.parse_args()
    if args.gpu_mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    cfg = dict(DEFAULT_CFG)
    if args.config_json:
        cfg.update(json.loads(Path(args.config_json).read_text()))
    md = Path(args.manifest_dir)
    rows = load_frames(md, args.sources, args.splits)
    pairs = load_pairs(md, args.sources, args.splits)
    if args.limit:
        rows, pairs = rows[: args.limit], pairs[: args.limit]
    atomic_json(CACHE / "stats" / f"generate_{args.stage}_config.json", {"cfg": cfg, "args": vars(args)})
    if args.stage == "static":
        run_static(rows, args, cfg)
    elif args.stage == "road":
        run_road(rows, args, cfg)
    elif args.stage == "flow":
        run_flow(pairs, args, cfg)
    elif args.stage == "contact":
        run_contact(rows, args, cfg)
    else:
        run_summary(rows, pairs, args, cfg)


if __name__ == "__main__":
    main()
