# Stage-2 external dataset collection report

Generated from the executed pipeline artifacts. Official sources were pinned to MM-AU dataset revision `540cb1277cb70e91a7022abe852decb3ee9adb0a`, MM-AU code/metadata commit `2b205a48d81f78b0c09d387d04a333aca9fc5949`, and CausalCrash revision `117221526dcabe23b55333457e3b49e0654d85d6`.

- MM-AU: [official repository](https://github.com/jeffreychou777/LOTVS-MM-AU), [Hugging Face release](https://huggingface.co/datasets/JeffreyChou/MM-AU)
- CausalCrash: [Hugging Face release](https://huggingface.co/datasets/meet2008/CausalCrash)

## Outcome

| Source | Total | High confidence | Borderline | Auto-reject |
|---|---:|---:|---:|---:|
| mmau | 11,730 | 806 | 2,198 | 8,726 |
| causalcrash | 273 | 8 | 98 | 167 |

The validated manual-labeling queue currently contains **324 unique videos**: MM-AU **316** and CausalCrash **8** (363 downloaded high-confidence videos before removing 39 duplicates). Full-decode validation passed for **363/363** downloaded files.

## Downloads

- mmau: AVAILABLE_AND_DOWNLOADED=355, NOT_SELECTED=10,924, NOT_SELECTED_ARCHIVE_LIMIT=451
- causalcrash: AVAILABLE_AND_DOWNLOADED=8, NOT_SELECTED=265

MM-AU is a 526 GB frame release. It does not expose per-sample files: CAP types 1–10 are one 42.42 GiB split gzip/tar stream, type 11 is 92.71 GiB, DADA-2000 is 115.20 GiB, and the remaining type groups are 28–43 GiB. The pipeline streams a selected archive directly through `tar`, extracts only selected members, and never stores the compressed archive. Archives that exceed the local safety envelope remain `NOT_SELECTED_ARCHIVE_LIMIT` rather than being treated as metadata rejects.

Deferred high-confidence MM-AU rows by archive: `CAP-DATA_chunks/11`=372, `DADA-2000_chunks`=79.

CausalCrash provides URLs rather than redistributed video. Unavailable/private/removed sources are isolated in the ledger and do not stop other downloads.

## Filtering rules actually used

MM-AU uses the official accident taxonomy from `cap_text_annotations.xls`, the `accident occurred` flag, `t_ai`, `t_co`, `total_frames`, description, cause, and prevention text. Ego-involved car/truck types are 7–12. The primary queue requires an actual crash, ego involvement, a road car/truck counterpart, affirmative ego lane-keeping evidence, explicit crossing/lane-change/turning intrusion evidence, usable pre-collision temporal context, and no strong ego-turning, rear-end, loss-of-control, or ego-fault text. `t_ai` remains an accident-window anchor and is never copied into `entry_frame`.

CausalCrash uses hazard/outcome labels, explicit ego agent identity, victim role, ego `keeping_lane` intent, a road-vehicle initiator, lateral/crossing/swerving intent, a causal-chain collision naming both agents, valid pre-collision time, and camera-description exclusions. CCTV, inward-facing cabin footage, ego observers/initiators, non-car/truck ego agents, near misses, and invalid collision times are rejected. Its schema has no dedicated camera/source-type field, so the evidence trail records that uncertainty.

## Auto-rejection reasons

### mmau
- official taxonomy says ego vehicle is not involved: 5,337
- strong negative: rear_end: 1,431
- collision counterpart is not a road car/truck: 1,327
- strong negative: ego_fault_speed_or_attention: 1,212
- strong negative: ego_loss_of_control: 466
- strong negative: ego_turning: 416
- strong negative: ego_changed_lane: 72
- free-text counterpart conflicts with car/truck taxonomy: 49
- metadata says no accident occurred: 17
- collision timestamp is outside the released frame range: 17
### causalcrash
- metadata does not establish an actual collision outcome: 61
- non-forward/non-ego camera evidence: cctv: 46
- initiator is not a road car/truck: 44
- ego role is incompatible: observer: 44
- ego role is incompatible: initiator: 23
- ego agent is not a road car/truck: 9
- non-forward/non-ego camera evidence: surveillance: 6
- ego intent is not lane keeping: braking: 6
- ego intent is not lane keeping: unknown: 3
- metadata says no hazard: 3
- collision anchor is absent or outside video duration: 3
- ego intent is not lane keeping: overtaking: 2
- non-forward/non-ego camera evidence: cctv, surveillance: 2
- ego intent is not lane keeping: swerving: 2
- non-forward/non-ego camera evidence: interior cabin: 1
- ego intent is not lane keeping: accelerating: 1
- non-forward/non-ego camera evidence: third-person: 1
- non-forward/non-ego camera evidence: interior cabin, inward-facing: 1
- non-forward/non-ego camera evidence: cctv, third-person: 1
- ego intent is not lane keeping: changing_left: 1
- non-forward/non-ego camera evidence: inward-facing: 1
- ego intent is not lane keeping: stopped: 1

## Validation and deduplication

Every downloaded video was probed for codec, dimensions, average FPS, frame count, time base, start time, and duration, then fully decoded with FFmpeg. Collision anchors were checked against actual duration. Exact SHA-256, source ID/URL, and seven sampled-frame perceptual hashes were compared within the new set and against 251 existing Stage-2 videos. Deduplication emitted **53 reports**: exact file matches **0**, duplicate source references **0**, confident visual near-duplicates removed from the queue **39** (≥6/7 sampled pHashes within 8 bits and median ≤3 bits; sample pairs visually confirmed as the same source footage re-trimmed; clips matching existing Stage-2 videos are dropped in favour of the existing clip), weaker potential visual matches **7**, and MM-AU duplicate release names **7**. Weaker potential matches are report-only.

A deterministic metadata sanity sample of 12 MM-AU rows from each decision class was reviewed before the archive transfer. The high-confidence descriptions consistently combined ego lane-keeping with crossing/cut-in evidence; borderline rows lacked explicit intrusion or temporal support; rejects showed the intended non-ego, wrong-counterpart, rear-end, or ego-fault evidence. A visual anchor review was performed for every strict CausalCrash candidate and 12 downloaded MM-AU candidates. The retained visual samples show ego-view pre-impact motion, lateral intrusion, and visible impact. Metadata anchors are approximate, as expected; the UI requires manual exact-frame labels.

## Available fields and deviations

- MM-AU fields: `video_name`, global `id`, numeric accident `type`, weather, light, scene, road layout, accident flag, `t_ai`, `t_co`, `t_ae`, total frames, description, causes, and measures. Original source URLs and FPS are absent from the metadata. The release's CAP loader uses 30 FPS, which is recorded explicitly in every generated sidecar. The official raw release contains image sequences rather than video files; selected sequences are converted frame-for-frame to H.264 with a sidecar mapping every video position to the release's numeric frame ID, then the temporary JPEG tree is removed.
- MM-AU archive members are keyed by the numeric suffix of `video_name`, padded to the release's directory width; the metadata's global `id` is not the archive directory identifier. The normalized table preserves both values and the exact resolved member prefix.
- CausalCrash fields: source URL, declared FPS/duration, hazard labels, severity/confidence, agents with roles/types/motion/lane/intent/location, environment, temporal windows, critical point, predicted outcome, causal chain, preventive actions, and counterfactuals. A dedicated camera-type field is absent.
- MM-AU `t_co` and CausalCrash collision events are stored only as candidates. Manual `entry_frame`, `collision_frame`, `entry_side`, and `evasion_space` begin empty.
- `/workspace` is not volume-backed. R2 is therefore the durable copy.

## Paths and storage

- Root: `/workspace/data/stage2_external`
- Normalized metadata: `/workspace/data/stage2_external/metadata/mmau/normalized.jsonl`, `/workspace/data/stage2_external/metadata/causalcrash/normalized.jsonl`
- High-confidence queue: `/workspace/data/stage2_external/manifests/high_confidence_label_queue.jsonl`
- Borderline: `/workspace/data/stage2_external/manifests/borderline_review.jsonl`
- Auto-rejected: `/workspace/data/stage2_external/manifests/auto_rejected.jsonl`
- Download failures: `/workspace/data/stage2_external/manifests/download_failures.jsonl`
- Dedup report: `/workspace/data/stage2_external/manifests/dedup_report.jsonl`
- Filter sanity sample: `/workspace/data/stage2_external/reports/filter_sanity_sample.json`
- MM-AU visual review sheet: `/workspace/data/stage2_external/reports/mmau_visual_review.jpg`
- Videos: `/workspace/data/stage2_external/raw_selected/mmau`, `/workspace/data/stage2_external/raw_selected/causalcrash`
- Persistent annotations: `/workspace/data/stage2_external/labeling/annotations.jsonl`
- Local footprint: 1.55 GiB; free filesystem space: 69.63 GiB
- R2: `r2:car-accident-dataset/stage2/external/`

## Labeling

Launch or restart with:

```bash
bash /workspace/car-accident/stage2/external/start_stage2_external_labeling.sh
```

Open the **Stage 2 External Labeling** application in the Vast portal (external port 10100). The interface supports video playback, jump-to-anchor, exact original-frame decoding, ±1/±5/±10 stepping, ENTRY/COLLISION, LEFT/RIGHT, evasion 0/1, usable/reject reason, autosave, and resume. Labels use original numeric frame IDs; playback proxies never define ground truth.
