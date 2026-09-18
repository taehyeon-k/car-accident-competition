"""Frozen teacher models. Every teacher is local-only, frozen and in eval mode.

TEACHERS records the exact checkpoints for the report and cache metadata.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from stage2.geometry_pretrain.common import INPUT_HW, LABEL_HW

TEACHERS = {
    "detector": {
        "model": "RF-DETR Small (COCO, 91 logits)",
        "checkpoint": "/workspace/pretrained/rfdetr_small/rf-detr-small.pth",
        "sha256": "d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33",
        "used_for": "vehicle/pedestrian boxes on footage without human boxes",
    },
    "segmenter": {
        "model": "SAM 2.1 Hiera-Small (box-prompted)",
        "checkpoint": "facebook/sam2.1-hiera-small -> /workspace/pretrained/sam2.1_hiera_small",
        "used_for": "instance masks from human (BDD) or detector boxes",
    },
    "depth": {
        "model": "Depth Anything V2 Small (relative inverse depth)",
        "checkpoint": "/workspace/pretrained/depth_anything_v2_small",
        "sha256": "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70",
        "used_for": "affine-invariant relative depth, flip-TTA consistency confidence",
    },
    "flow": {
        "model": "SEA-RAFT-S (spring-S eval config)",
        "checkpoint": "/workspace/pretrained/sea_raft/model.safetensors",
        "used_for": "optical flow + mixture-Laplace uncertainty, fw/bw consistency",
    },
    "road": {
        "model": "DINOv3 ViT-B/16 + road head trained on BDD100K human drivable/lane/curb labels",
        "checkpoint": "/workspace/outputs/geometry_pretrain/road_teacher/best.pt",
        "used_for": "drivable (ego/alt/bg), lane-marking and curb pseudo labels on accident/BATON footage",
    },
}

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
COCO_TO_OURS = {1: 5, 2: 4, 3: 1, 4: 4, 6: 3, 7: 2, 8: 2}


def normalize(images_uint8: torch.Tensor) -> torch.Tensor:
    x = images_uint8.float() / 255.0
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class Detector:
    def __init__(self, device="cuda", score_threshold: float = 0.5):
        from stage2.model.backbones import load_local

        wrapper = load_local("stage2.model.local_assets:detector", TEACHERS["detector"]["checkpoint"])
        self.network = wrapper.network.to(device).eval()
        self.postprocess = wrapper.postprocess
        self.device = device
        self.score_threshold = score_threshold

    @torch.inference_mode()
    def __call__(self, images_uint8: torch.Tensor):
        """images: B,3,H,W uint8 RGB at INPUT_HW -> list of box dicts (input px)."""
        x = F.interpolate(normalize(images_uint8), (512, 512), mode="bilinear", align_corners=False)
        sizes = torch.tensor([INPUT_HW] * len(x), device=x.device)
        preds = self.postprocess(self.network(x), sizes)
        out = []
        for p in preds:
            keep = p["scores"] >= self.score_threshold
            boxes = []
            for box, score, label in zip(p["boxes"][keep].tolist(), p["scores"][keep].tolist(), p["labels"][keep].tolist()):
                cls = COCO_TO_OURS.get(int(label))
                if cls is not None:
                    boxes.append({"cls": cls, "box": box, "score": float(score), "source": "rfdetr"})
            out.append(boxes)
        return out


class BoxSegmenter:
    def __init__(self, device="cuda", path="/workspace/pretrained/sam2.1_hiera_small"):
        from transformers import Sam2Model

        self.model = Sam2Model.from_pretrained(path).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, images_uint8: torch.Tensor, boxes_per_image: list[list[list[float]]], out_hw=(224, 400)):
        """Per image: (masks[N,h,w] bool at out_hw, iou[N], tight_box[N,4] input px, area[N] input px).

        One padded prompt-decoder call covers the whole batch; mask statistics
        used for acceptance are computed on the GPU.
        """
        x = F.interpolate(normalize(images_uint8), (1024, 1024), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb = self.model.get_image_embeddings(x)
        H, W = INPUT_HW
        h, w = out_hw
        n_max = max((len(b) for b in boxes_per_image), default=0)
        empty = (torch.zeros(0, h, w, dtype=torch.bool), torch.zeros(0), torch.zeros(0, 4), torch.zeros(0))
        if n_max == 0:
            return [empty for _ in boxes_per_image]
        B = len(boxes_per_image)
        padded = torch.zeros(B, n_max, 4, device=self.device)
        for i, boxes in enumerate(boxes_per_image):
            if boxes:
                b = torch.tensor(boxes, device=self.device, dtype=torch.float32)
                padded[i, : len(b)] = b
                padded[i, len(b) :] = b[0]
        scaled = padded * padded.new_tensor([1024 / W, 1024 / H, 1024 / W, 1024 / H])
        masks, ious = [], []
        for g in range(0, B, 4):  # decode 4 images at a time to bound peak memory
            gm, gi = [], []
            for chunk in range(0, n_max, 64):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self.model(image_embeddings=[e[g : g + 4] for e in emb],
                                     input_boxes=scaled[g : g + 4, chunk : chunk + 64], multimask_output=False)
                low = out.pred_masks[:, :, 0]  # b,N,256,256 (bf16)
                gm.append(F.interpolate(low, (h, w), mode="bilinear", align_corners=False) > 0)
                gi.append(out.iou_scores[:, :, 0].float())
                del out, low
            masks.append(torch.cat(gm, 1))
            ious.append(torch.cat(gi, 1))
        masks, ious = torch.cat(masks, 0), torch.cat(ious, 0)
        rows_any, cols_any = masks.any(3), masks.any(2)  # B,N,h / B,N,w
        ar_h = torch.arange(h, device=masks.device).float()
        ar_w = torch.arange(w, device=masks.device).float()
        big = 1e9
        y0 = torch.where(rows_any, ar_h, big).amin(-1)
        y1 = torch.where(rows_any, ar_h, -big).amax(-1) + 1
        x0 = torch.where(cols_any, ar_w, big).amin(-1)
        x1 = torch.where(cols_any, ar_w, -big).amax(-1) + 1
        tight = torch.stack([x0 * W / w, y0 * H / h, x1 * W / w, y1 * H / h], -1)
        area = masks.sum((2, 3)).float() * (H / h) * (W / w)
        masks, ious, tight, area = masks.cpu(), ious.cpu(), tight.cpu(), area.cpu()
        return [
            (masks[i, : len(b)], ious[i, : len(b)], tight[i, : len(b)], area[i, : len(b)])
            for i, b in enumerate(boxes_per_image)
        ]


class DepthTeacher:
    def __init__(self, device="cuda", path=TEACHERS["depth"]["checkpoint"], size=(518, 924)):
        from transformers import AutoModelForDepthEstimation

        self.model = AutoModelForDepthEstimation.from_pretrained(path).to(device).eval()
        self.size = size

    @torch.inference_mode()
    def __call__(self, images_uint8: torch.Tensor):
        """Return disparity (B,h,w) at LABEL_HW and flip-consistency confidence."""
        x = F.interpolate(normalize(images_uint8), self.size, mode="bicubic", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d = self.model(pixel_values=torch.cat([x, x.flip(-1)])).predicted_depth.float()
        d = F.interpolate(d[:, None], LABEL_HW, mode="area")[:, 0]
        d1, d2 = d.chunk(2)
        d2 = d2.flip(-1)
        # Least-squares affine alignment of the flipped pass onto the direct pass.
        B = d1.shape[0]
        a, b = d1.reshape(B, -1), d2.reshape(B, -1)
        am, bm = a.mean(1, keepdim=True), b.mean(1, keepdim=True)
        s = ((a - am) * (b - bm)).sum(1, keepdim=True) / ((b - bm) ** 2).sum(1, keepdim=True).clamp_min(1e-6)
        b_al = (b - bm) * s + am
        spread = (a - a.median(1, keepdim=True).values).abs().mean(1, keepdim=True).clamp_min(1e-6)
        rel = ((a - b_al).abs() / spread).reshape_as(d1)
        conf = torch.exp(-rel / 0.25)
        disp = 0.5 * (d1 + b_al.reshape_as(d1))
        return disp, conf, rel.reshape(B, -1).mean(1)


class FlowTeacher:
    """SEA-RAFT-S run at ``infer_hw`` (flow is only stored at stride 4 anyway)."""

    def __init__(self, device="cuda", batch_size: int = 8, infer_hw=(288, 512)):
        from stage3.flow.sea_raft import SeaRaftS

        self.raft = SeaRaftS("/workspace/pretrained/sea_raft/source", TEACHERS["flow"]["checkpoint"], device, batch_size)
        self.infer_hw = tuple(infer_hw)

    @torch.inference_mode()
    def __call__(self, img_t: torch.Tensor, img_tp: torch.Tensor):
        """uint8 B,3,H,W pairs -> fw flow, bw flow (in infer_hw px), confidences at infer_hw."""
        a = F.interpolate(img_t.float(), self.infer_hw, mode="bilinear", align_corners=False, antialias=True)
        b = F.interpolate(img_tp.float(), self.infer_hw, mode="bilinear", align_corners=False, antialias=True)
        fw, cfw = self.raft.estimate_batch(torch.cat([a, b]), torch.cat([b, a]))  # fp32 for accuracy
        f_fw, f_bw = fw.float().chunk(2)
        c_fw, c_bw = cfw.float().chunk(2)
        return f_fw, f_bw, c_fw, c_bw


def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample x (B,C,H,W) at p + flow(p)."""
    B, _, H, W = flow.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=flow.device), torch.arange(W, device=flow.device), indexing="ij")
    gx = (xs[None] + flow[:, 0]) / (W - 1) * 2 - 1
    gy = (ys[None] + flow[:, 1]) / (H - 1) * 2 - 1
    return F.grid_sample(x, torch.stack([gx, gy], -1), align_corners=True, padding_mode="border")


def fb_consistency(f_fw: torch.Tensor, f_bw: torch.Tensor, alpha1: float = 0.01, alpha2: float = 0.5):
    """UnFlow forward/backward occlusion test; 1 = consistent (non-occluded, in view)."""
    bw_at = warp(f_bw, f_fw)
    lhs = (f_fw + bw_at).pow(2).sum(1)
    rhs = alpha1 * (f_fw.pow(2).sum(1) + bw_at.pow(2).sum(1)) + alpha2
    H, W = f_fw.shape[-2:]
    ys, xs = torch.meshgrid(torch.arange(H, device=f_fw.device), torch.arange(W, device=f_fw.device), indexing="ij")
    tx, ty = xs[None] + f_fw[:, 0], ys[None] + f_fw[:, 1]
    inside = (tx >= 0) & (tx <= W - 1) & (ty >= 0) & (ty <= H - 1)
    return (lhs < rhs) & inside
