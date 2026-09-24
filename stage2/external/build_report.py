"""Build the executed Stage-2 external-data collection report from artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
from typing import Any


DEFAULT_ROOT = Path(os.environ.get("STAGE2_EXTERNAL_ROOT", "/workspace/data/stage2_external"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def gib(value: int) -> str:
    return f"{value / 2**30:.2f} GiB"


def counts_table(decisions: dict[str, dict[str, int]]) -> str:
    return "\n".join(
        [
            "| Source | Total | High confidence | Borderline | Auto-reject |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {source} | {sum(values.values()):,} | {values.get('HIGH_CONFIDENCE_LABEL_QUEUE', 0):,} | {values.get('BORDERLINE_REVIEW', 0):,} | {values.get('AUTO_REJECT', 0):,} |"
                for source, values in decisions.items()
            ],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    arguments = parser.parse_args()
    root = arguments.root
    summary = json.loads((root / "reports/filter_summary.json").read_text(encoding="utf-8"))
    rejected = read_jsonl(root / "manifests/auto_rejected.jsonl")
    reasons = {
        source: Counter(reason for row in rejected if row["source"] == source for reason in row.get("filter_reason", []))
        for source in ("mmau", "causalcrash")
    }
    downloaded = read_jsonl(root / "manifests/high_confidence_label_queue.jsonl")
    queue = read_jsonl(root / "labeling/queue.jsonl")
    validation = read_jsonl(root / "manifests/video_validation.jsonl")
    dedup = read_jsonl(root / "manifests/dedup_report.jsonl")
    ledgers = {
        source: read_jsonl(root / f"manifests/{source}_download_ledger.jsonl")
        for source in ("mmau", "causalcrash")
    }
    download_counts = {source: Counter(row["download_status"] for row in rows) for source, rows in ledgers.items()}
    archive_limited = Counter(
        row.get("shard", "unknown")
        for row in ledgers["mmau"]
        if row.get("download_status") == "NOT_SELECTED_ARCHIVE_LIMIT"
    )
    queue_counts = Counter(row["source"] for row in queue)
    free = shutil.disk_usage(root).free
    used = directory_bytes(root)
    reason_lines = []
    for source, counter in reasons.items():
        reason_lines.append(f"### {source}")
        reason_lines.extend(f"- {reason}: {count:,}" for reason, count in counter.most_common())
    download_lines = []
    for source, counter in download_counts.items():
        download_lines.append(f"- {source}: " + ", ".join(f"{status}={count:,}" for status, count in sorted(counter.items())))
    report = f"""# Stage-2 external dataset collection report

Generated from the executed pipeline artifacts. Official sources were pinned to MM-AU dataset revision `540cb1277cb70e91a7022abe852decb3ee9adb0a`, MM-AU code/metadata commit `2b205a48d81f78b0c09d387d04a333aca9fc5949`, and CausalCrash revision `117221526dcabe23b55333457e3b49e0654d85d6`.

- MM-AU: [official repository](https://github.com/jeffreychou777/LOTVS-MM-AU), [Hugging Face release](https://huggingface.co/datasets/JeffreyChou/MM-AU)
- CausalCrash: [Hugging Face release](https://huggingface.co/datasets/meet2008/CausalCrash)

## Outcome

{counts_table(summary['decisions'])}

The validated manual-labeling queue currently contains **{len(queue):,} unique videos**: MM-AU **{queue_counts.get('mmau', 0):,}** and CausalCrash **{queue_counts.get('causalcrash', 0):,}** ({len(downloaded):,} downloaded high-confidence videos before removing {len(downloaded) - len(queue):,} duplicates). Full-decode validation passed for **{sum(row.get('validation_status') == 'VALID' for row in validation):,}/{len(validation):,}** downloaded files.

## Downloads

{chr(10).join(download_lines)}

MM-AU is a 526 GB frame release. It does not expose per-sample files: CAP types 1–10 are one 42.42 GiB split gzip/tar stream, type 11 is 92.71 GiB, DADA-2000 is 115.20 GiB, and the remaining type groups are 28–43 GiB. The pipeline streams a selected archive directly through `tar`, extracts only selected members, and never stores the compressed archive. Archives that exceed the local safety envelope remain `NOT_SELECTED_ARCHIVE_LIMIT` rather than being treated as metadata rejects.

Deferred high-confidence MM-AU rows by archive: {', '.join(f'`{archive}`={count:,}' for archive, count in sorted(archive_limited.items())) or 'none'}.

CausalCrash provides URLs rather than redistributed video. Unavailable/private/removed sources are isolated in the ledger and do not stop other downloads.

## Filtering rules actually used

MM-AU uses the official accident taxonomy from `cap_text_annotations.xls`, the `accident occurred` flag, `t_ai`, `t_co`, `total_frames`, description, cause, and prevention text. Ego-involved car/truck types are 7–12. The primary queue requires an actual crash, ego involvement, a road car/truck counterpart, affirmative ego lane-keeping evidence, explicit crossing/lane-change/turning intrusion evidence, usable pre-collision temporal context, and no strong ego-turning, rear-end, loss-of-control, or ego-fault text. `t_ai` remains an accident-window anchor and is never copied into `entry_frame`.

CausalCrash uses hazard/outcome labels, explicit ego agent identity, victim role, ego `keeping_lane` intent, a road-vehicle initiator, lateral/crossing/swerving intent, a causal-chain collision naming both agents, valid pre-collision time, and camera-description exclusions. CCTV, inward-facing cabin footage, ego observers/initiators, non-car/truck ego agents, near misses, and invalid collision times are rejected. Its schema has no dedicated camera/source-type field, so the evidence trail records that uncertainty.

## Auto-rejection reasons

{chr(10).join(reason_lines)}

## Validation and deduplication

Every downloaded video was probed for codec, dimensions, average FPS, frame count, time base, start time, and duration, then fully decoded with FFmpeg. Collision anchors were checked against actual duration. Exact SHA-256, source ID/URL, and seven sampled-frame perceptual hashes were compared within the new set and against 251 existing Stage-2 videos. Deduplication emitted **{len(dedup):,} reports**: exact file matches **{sum(row.get('kind') == 'EXACT_FILE' for row in dedup):,}**, duplicate source references **{sum(row.get('kind') == 'DUPLICATE_SOURCE_REFERENCE' for row in dedup):,}**, confident visual near-duplicates removed from the queue **{sum(row.get('kind') == 'CONFIDENT_NEAR_DUPLICATE' for row in dedup):,}** (≥6/7 sampled pHashes within 8 bits and median ≤3 bits; sample pairs visually confirmed as the same source footage re-trimmed; clips matching existing Stage-2 videos are dropped in favour of the existing clip), weaker potential visual matches **{sum(row.get('kind') == 'POTENTIAL_NEAR_DUPLICATE' for row in dedup):,}**, and MM-AU duplicate release names **{sum(row.get('kind') == 'DUPLICATE_RELEASE_VIDEO_NAME' for row in dedup):,}**. Weaker potential matches are report-only.

A deterministic metadata sanity sample of 12 MM-AU rows from each decision class was reviewed before the archive transfer. The high-confidence descriptions consistently combined ego lane-keeping with crossing/cut-in evidence; borderline rows lacked explicit intrusion or temporal support; rejects showed the intended non-ego, wrong-counterpart, rear-end, or ego-fault evidence. A visual anchor review was performed for every strict CausalCrash candidate and 12 downloaded MM-AU candidates. The retained visual samples show ego-view pre-impact motion, lateral intrusion, and visible impact. Metadata anchors are approximate, as expected; the UI requires manual exact-frame labels.

## Available fields and deviations

- MM-AU fields: `video_name`, global `id`, numeric accident `type`, weather, light, scene, road layout, accident flag, `t_ai`, `t_co`, `t_ae`, total frames, description, causes, and measures. Original source URLs and FPS are absent from the metadata. The release's CAP loader uses 30 FPS, which is recorded explicitly in every generated sidecar. The official raw release contains image sequences rather than video files; selected sequences are converted frame-for-frame to H.264 with a sidecar mapping every video position to the release's numeric frame ID, then the temporary JPEG tree is removed.
- MM-AU archive members are keyed by the numeric suffix of `video_name`, padded to the release's directory width; the metadata's global `id` is not the archive directory identifier. The normalized table preserves both values and the exact resolved member prefix.
- CausalCrash fields: source URL, declared FPS/duration, hazard labels, severity/confidence, agents with roles/types/motion/lane/intent/location, environment, temporal windows, critical point, predicted outcome, causal chain, preventive actions, and counterfactuals. A dedicated camera-type field is absent.
- MM-AU `t_co` and CausalCrash collision events are stored only as candidates. Manual `entry_frame`, `collision_frame`, `entry_side`, and `evasion_space` begin empty.
- `/workspace` is not volume-backed. R2 is therefore the durable copy.

## Paths and storage

- Root: `{root}`
- Normalized metadata: `{root}/metadata/mmau/normalized.jsonl`, `{root}/metadata/causalcrash/normalized.jsonl`
- High-confidence queue: `{root}/manifests/high_confidence_label_queue.jsonl`
- Borderline: `{root}/manifests/borderline_review.jsonl`
- Auto-rejected: `{root}/manifests/auto_rejected.jsonl`
- Download failures: `{root}/manifests/download_failures.jsonl`
- Dedup report: `{root}/manifests/dedup_report.jsonl`
- Filter sanity sample: `{root}/reports/filter_sanity_sample.json`
- MM-AU visual review sheet: `{root}/reports/mmau_visual_review.jpg`
- Videos: `{root}/raw_selected/mmau`, `{root}/raw_selected/causalcrash`
- Persistent annotations: `{root}/labeling/annotations.jsonl`
- Local footprint: {gib(used)}; free filesystem space: {gib(free)}
- R2: `r2:car-accident-dataset/stage2/external/`

## Labeling

Launch or restart with:

```bash
bash /workspace/car-accident/stage2/external/start_stage2_external_labeling.sh
```

Open the **Stage 2 External Labeling** application in the Vast portal (external port 10100). The interface supports video playback, jump-to-anchor, exact original-frame decoding, ±1/±5/±10 stepping, ENTRY/COLLISION, LEFT/RIGHT, evasion 0/1, usable/reject reason, autosave, and resume. Labels use original numeric frame IDs; playback proxies never define ground truth.
"""
    destination = root / "reports/stage2_external_dataset_report.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    repository_report = Path("/workspace/car-accident/reports/stage2_external_dataset_report.md")
    repository_report.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(destination), "queue": len(queue), "disk_used": gib(used), "free": gib(free)}, indent=2))


if __name__ == "__main__":
    main()
