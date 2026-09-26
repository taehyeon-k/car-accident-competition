"""Merge unchanged BATON split with drive-disjoint DriveDNA segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stage3.data.adapters.drivedna import DriveDnaAdapter
from stage3.scripts.build_manifest import segment_record
from stage3.utils.config import read_jsonl


def build(baton_dir: Path, drivedna_raw: Path, filter_report: Path, selection_report: Path,
          cache_dir: Path, output: Path, seed: int = 42, validation_fraction: float = .2,
          segment_seconds: float = 30.) -> dict:
    if output.exists() and list(output.glob("*.jsonl")):
        raise FileExistsError(f"Refusing to replace existing manifests: {output}")
    if not 0 < validation_fraction < 1 or segment_seconds <= 0:
        raise ValueError("Invalid validation fraction or segment duration")
    selection = json.loads(selection_report.read_text())
    records = DriveDnaAdapter(drivedna_raw, filter_report, selection_report).discover()
    if len(records) != selection["accepted_drives"]:
        raise ValueError("Selected DriveDNA drives do not match selection report")
    records.sort(key=lambda record: record.clip_id)
    rng = np.random.default_rng(seed)
    val_ids = {records[i].clip_id for i in rng.permutation(len(records))[:max(1, round(len(records) * validation_fraction))]}
    parts = {"drivedna_train": [], "drivedna_val": [], "baton_train": [], "baton_val": []}
    for record in records:
        part = "drivedna_val" if record.clip_id in val_ids else "drivedna_train"
        parts[part].extend(segment_record(record, cache_dir, segment_seconds))
    for part in ("train", "val"):
        rows = read_jsonl(baton_dir / f"{part}.jsonl")
        for row in rows:
            row["metadata"] = {**row["metadata"], "source": "baton"}
            cache_path = Path(row["cache_path"])
            if not cache_path.is_file():
                alternate = cache_path.parent.parent / "motion" / cache_path.name
                if not alternate.is_file():
                    raise FileNotFoundError(f"BATON cache missing: {cache_path} or {alternate}")
                row["cache_path"] = str(alternate)
        parts[f"baton_{part}"] = rows
    train_ids = {row["metadata"]["source_clip_id"] for row in parts["drivedna_train"]}
    val_ids_seen = {row["metadata"]["source_clip_id"] for row in parts["drivedna_val"]}
    if train_ids & val_ids_seen:
        raise ValueError("Original DriveDNA drive leaked between train and validation")
    parts["drivedna_all"] = parts["drivedna_train"] + parts["drivedna_val"]
    parts["baton_all"] = parts["baton_train"] + parts["baton_val"]
    parts["train"] = parts["baton_train"] + parts["drivedna_train"]
    parts["val"] = parts["baton_val"] + parts["drivedna_val"]
    parts["all"] = parts["train"] + parts["val"]
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in parts.items():
        (output / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = {
        name: {"segments": len(rows), "drives": len({row["metadata"]["source_clip_id"] for row in rows}),
               "hours": sum(row["metadata"]["segment_end_time"] - row["metadata"]["segment_start_time"] for row in rows) / 3600}
        for name, rows in parts.items()
    }
    (output / "composition.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baton-dir", type=Path, required=True)
    parser.add_argument("--drivedna-raw", type=Path, required=True)
    parser.add_argument("--filter-report", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    args = parser.parse_args()
    print(json.dumps(build(args.baton_dir, args.drivedna_raw, args.filter_report, args.selection_report,
                           args.cache_dir, args.output, args.seed, args.validation_fraction), indent=2))


if __name__ == "__main__":
    main()
