"""Assemble the DACON code-submission package and zip it.

    submit/
    ├── inference.py
    ├── requirements.txt
    └── model/
        ├── stage1/best.pt                       (from the Stage 1 zip, unchanged)
        ├── stage2/joint_model.pt                (config + weights only, float32)
        ├── stage2/code/stage2/...               (the modules inference imports)
        ├── stage2/pretrained/{dinov3,vjepa2}-source  (architectures + licenses)
        ├── stage2/pretrained/rfdetr_small/rf-detr-small.pth
        └── stage3/{best.pt,code/stage3,pretrained/sea_raft}

joint_model.pt already holds every frozen backbone weight plus LoRA and the head,
so the multi-GB DINOv3/V-JEPA checkpoints are never shipped.
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
# Measured import closure of the inference path; keep in sync if imports change.
STAGE2_MODULES = (
    "__init__.py",
    "data/__init__.py",
    "data/augment.py",
    "data/cache_geometry.py",
    "data/joint.py",
    "data/joint_sampling.py",
    "data/sampling.py",
    "data/transforms.py",
    "model/__init__.py",
    "model/backbones.py",
    "model/joint.py",
    "model/joint_system.py",
    "model/joint_tracking.py",
    "model/local_assets.py",
    "model/lora.py",
    "model/modules.py",
    "model/tracking.py",
    "utils/__init__.py",
    "utils/joint_losses.py",
    "utils/utils.py",
)
STORED_SUFFIXES = {".pt", ".pth", ".safetensors"}  # dense; deflating only costs time


def export_checkpoint(source: Path, target: Path) -> dict:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("format_version") != 2
        or checkpoint["config"]["stage"] != "joint"
    ):
        raise ValueError(f"{source} is not a Stage 2 joint checkpoint")
    exported = {
        "format_version": 2,
        "config": checkpoint["config"],
        "model": checkpoint["model"],
        "epoch": checkpoint.get("epoch"),
        "validation_metrics": checkpoint.get("validation_metrics", {}),
        "source_checkpoint": str(source),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, target)
    return exported


def export_stage3_checkpoint(source: Path, target: Path) -> dict:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1 or checkpoint["config"].get("stage") != "stage3":
        raise ValueError(f"{source} is not a Stage 3 checkpoint")
    exported = {
        key: checkpoint[key]
        for key in ("format_version", "config", "model", "ema_model", "physics_center", "physics_scale", "epoch", "validation_metrics")
        if key in checkpoint
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, target)
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", default="/workspace/car-accident/runs/joint_online_lora/best.pt"
    )
    parser.add_argument(
        "--stage1-zip", default="/workspace/global_g1_threshold_0_50.zip"
    )
    parser.add_argument(
        "--stage3-checkpoint", default="/workspace/car-accident/runs/stage3/baseline_v1_2/best.pt"
    )
    parser.add_argument("--pretrained", default="/workspace/pretrained")
    parser.add_argument("--output", default="/workspace/outputs/submit")
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output directory"
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        if not args.force:
            raise SystemExit(f"{output} exists; pass --force to rebuild it")
        shutil.rmtree(output)
    model = output / "model"

    for name in ("inference.py", "requirements.txt"):
        (output / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / "submission" / name, output / name)

    with zipfile.ZipFile(args.stage1_zip) as archive:
        (model / "stage1").mkdir(parents=True)
        with (
            archive.open("model/stage1/best.pt") as src,
            open(model / "stage1" / "best.pt", "wb") as dst,
        ):
            shutil.copyfileobj(src, dst)

    exported = export_checkpoint(
        Path(args.checkpoint), model / "stage2" / "joint_model.pt"
    )
    for relative in STAGE2_MODULES:
        target = model / "stage2" / "code" / "stage2" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / "stage2" / relative, target)
    ignore = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".github")
    for source in ("dinov3-source", "vjepa2-source"):
        shutil.copytree(
            Path(args.pretrained) / source,
            model / "stage2" / "pretrained" / source,
            ignore=ignore,
        )
    rfdetr = model / "stage2" / "pretrained" / "rfdetr_small"
    rfdetr.mkdir(parents=True)
    shutil.copy2(
        Path(args.pretrained) / "rfdetr_small" / "rf-detr-small.pth",
        rfdetr / "rf-detr-small.pth",
    )
    stage3_exported = export_stage3_checkpoint(
        Path(args.stage3_checkpoint), model / "stage3" / "best.pt"
    )
    for path in sorted((REPO / "stage3").rglob("*.py")):
        if "tests" in path.parts or "scripts" in path.parts:
            continue
        target = model / "stage3" / "code" / "stage3" / path.relative_to(REPO / "stage3")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    sea_root = model / "stage3" / "pretrained" / "sea_raft"
    shutil.copytree(
        Path(args.pretrained) / "sea_raft" / "source",
        sea_root / "source",
        ignore=ignore,
    )
    sea_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(args.pretrained) / "sea_raft" / "model.safetensors", sea_root / "model.safetensors")

    zip_path = output.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", allowZip64=True) as archive:
        for path in sorted(p for p in output.rglob("*") if p.is_file()):
            compress = (
                zipfile.ZIP_STORED
                if path.suffix in STORED_SUFFIXES
                else zipfile.ZIP_DEFLATED
            )
            archive.write(path, path.relative_to(output), compress_type=compress)
    unpacked = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    print(
        f"stage 2 checkpoint: epoch {exported['epoch']} | "
        f"val score {exported['validation_metrics'].get('competition_score', float('nan')):.4f}"
    )
    print(
        f"stage 3 checkpoint: epoch {stage3_exported.get('epoch')} | "
        f"acceleration F1 {stage3_exported.get('validation_metrics', {}).get('acceleration_macro_f1', float('nan')):.4f}"
    )
    print(f"package: {output} ({unpacked / 2**30:.2f} GiB unpacked)")
    print(
        f"zip: {zip_path} ({zip_path.stat().st_size / 2**30:.2f} GiB; limit 10 GiB zipped / 32 GiB unpacked)"
    )


if __name__ == "__main__":
    main()
