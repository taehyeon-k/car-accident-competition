"""Normalize official metadata and create precision-first Stage-2 decisions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd


DEFAULT_ROOT = Path(os.environ.get("STAGE2_EXTERNAL_ROOT", "/workspace/data/stage2_external"))
CONFIG = Path(__file__).with_name("filter_config.json")
DECISIONS = ("HIGH_CONFIDENCE_LABEL_QUEUE", "BORDERLINE_REVIEW", "AUTO_REJECT")


def normalized_text(*values: Any) -> str:
    text = " ".join(str(value or "") for value in values).lower()
    text = text.replace("\xa0", " ").replace("[cls]", " ").replace("[sep]", " ")
    return re.sub(r"\s+", " ", text).strip()


def matches(patterns: Iterable[str], text: str) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, re.IGNORECASE)]


def integer(value: Any) -> int | None:
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        return None
    return result


def base_row(source: str, source_id: str, source_url: str) -> dict[str, Any]:
    return {
        "sample_id": f"{source}_{source_id}",
        "source": source,
        "source_id": source_id,
        "source_url": source_url,
        "video_path": "",
        "fps": "",
        "num_frames": "",
        "duration_s": "",
        "width": "",
        "height": "",
        "metadata_accident_type": "",
        "metadata_accident_start": "",
        "metadata_collision_time": "",
        "metadata_collision_frame_candidate": "",
        "ego_view_evidence": [],
        "ego_involved_evidence": [],
        "ego_lane_keep_evidence": [],
        "other_vehicle_intrusion_evidence": [],
        "entry_visible_evidence": [],
        "collision_visible_evidence": [],
        "positive_evidence": [],
        "negative_evidence": [],
        "filter_score": 0,
        "filter_decision": "",
        "filter_reason": [],
        "dedup_status": "NOT_CHECKED",
        "usable": "",
        "entry_frame": "",
        "collision_frame": "",
        "entry_side": "",
        "evasion_space": "",
        "reject_reason": "",
        "notes": "",
    }


def mmau_taxonomy(path: Path) -> dict[int, str]:
    frame = pd.read_excel(path, sheet_name="Sheet1")
    return {
        int(row.iloc[0]): str(row.iloc[1]).strip()
        for _, row in frame.dropna(how="any").iterrows()
    }


def mmau_archive(record_id: int, accident_type: int, video_name: str) -> tuple[str, str]:
    serial = int(video_name.split("_", 1)[1])
    if record_id <= 1962:
        return (
            "DADA-2000_chunks",
            f"Origin/DADA2000/DADA2000/{accident_type}/{serial:03d}/images/",
        )
    if accident_type <= 10:
        group = "1-10"
    elif accident_type == 11:
        group = "11"
    elif accident_type <= 42:
        group = "12-42"
    elif accident_type == 43:
        group = "43"
    else:
        group = "44-62"
    return (
        f"CAP-DATA_chunks/{group}",
        f"CAP-DATA/{group}/{accident_type}/{serial:06d}/images/",
    )


def normalize_mmau(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = root / "metadata/mmau"
    raw = json.loads((metadata / "video_metadata.json").read_text(encoding="utf-8"))
    taxonomy = mmau_taxonomy(metadata / "cap_text_annotations.xls")
    rules = config["mmau"]
    rows = []
    for hashcode, source in raw.items():
        record_id = int(source["id"])
        accident_type = int(source["type"])
        t_ai = integer(source.get("t_ai"))
        t_co = integer(source.get("t_co"))
        t_ae = integer(source.get("t_ae"))
        total = integer(source.get("total_frames"))
        archive, member_prefix = mmau_archive(record_id, accident_type, source["video_name"])
        row = base_row("mmau", f"{record_id:06d}", "https://huggingface.co/datasets/JeffreyChou/MM-AU")
        row.update(
            mmau_hashcode=hashcode,
            mmau_video_name=source["video_name"],
            mmau_type=accident_type,
            mmau_archive=archive,
            mmau_member_prefix=member_prefix,
            weather=source.get("weather", ""),
            light=source.get("light", ""),
            scenes=source.get("scenes", ""),
            linear=source.get("linear", ""),
            metadata_accident_type=taxonomy.get(accident_type, f"type {accident_type}"),
            metadata_accident_start=t_ai if t_ai is not None else "",
            metadata_collision_time="",
            metadata_collision_frame_candidate=t_co if t_co is not None else "",
            metadata_accident_end=t_ae if t_ae is not None else "",
            num_frames=total if total is not None else "",
            mmau_text=source.get("texts", ""),
            mmau_causes=source.get("causes", ""),
            mmau_measures=source.get("measures", ""),
        )
        text = normalized_text(source.get("texts"), source.get("causes"), source.get("measures"))
        actual = str(source.get("accident occurred")) == "1"
        ego_involved = accident_type in rules["ego_involved_types"]
        road_vehicle = accident_type in rules["road_vehicle_types"]
        structured_intrusion = accident_type in rules["structured_intrusion_types"]
        intrusion_hits = matches(rules["intrusion_patterns"], normalized_text(source.get("texts")))
        lane_hits = matches(rules["ego_lane_keep_patterns"], text)
        negative_hits = {
            reason: matches(patterns, text)
            for reason, patterns in rules["negative_patterns"].items()
        }
        negative_hits = {reason: values for reason, values in negative_hits.items() if values}
        counterpart_mismatch = bool(
            road_vehicle
            and re.search(r"\b(pedestrian|cyclist|motorbike|motorcycle)\b", normalized_text(source.get("texts")))
        )
        temporal_valid = all(value is not None for value in (t_ai, t_co, total)) and 0 <= t_ai < t_co <= total
        pre_visible = bool(
            temporal_valid
            and t_ai >= rules["minimum_pre_event_frames"]
            and t_co - t_ai >= rules["minimum_entry_to_collision_frames"]
        )
        collision_visible = bool(t_co is not None and total is not None and 0 <= t_co <= total)

        row["ego_view_evidence"] = ["MM-AU release is documented as ego-view"]
        if ego_involved:
            row["ego_involved_evidence"] = [f"official taxonomy type {accident_type}: {row['metadata_accident_type']}"]
        if lane_hits:
            row["ego_lane_keep_evidence"] = lane_hits
        if structured_intrusion:
            row["other_vehicle_intrusion_evidence"].append("official crossing-vehicle taxonomy")
        row["other_vehicle_intrusion_evidence"].extend(intrusion_hits)
        if temporal_valid:
            row["entry_visible_evidence"].append(f"t_ai={t_ai} precedes t_co={t_co}; t_ai is only an accident-window anchor")
        if pre_visible:
            row["entry_visible_evidence"].append(f"{t_ai} frames precede t_ai and {t_co - t_ai} frames span t_ai→t_co")
        if collision_visible:
            row["collision_visible_evidence"].append(f"t_co={t_co} lies within total_frames={total}")

        score = int(actual) * 2 + 1 + int(ego_involved) * 2 + int(road_vehicle) * 2
        score += int(structured_intrusion) * 3 + int(bool(intrusion_hits)) * 2
        score += int(bool(lane_hits)) + int(pre_visible) + int(collision_visible)
        score -= 6 * len(negative_hits)
        row["filter_score"] = score
        row["positive_evidence"] = (
            row["ego_view_evidence"]
            + row["ego_involved_evidence"]
            + row["ego_lane_keep_evidence"]
            + row["other_vehicle_intrusion_evidence"]
            + row["entry_visible_evidence"]
            + row["collision_visible_evidence"]
        )
        row["negative_evidence"] = [f"{reason}: {pattern}" for reason, patterns in negative_hits.items() for pattern in patterns]

        hard_reasons = []
        if not actual:
            hard_reasons.append("metadata says no accident occurred")
        if not ego_involved:
            hard_reasons.append("official taxonomy says ego vehicle is not involved")
        if ego_involved and not road_vehicle:
            hard_reasons.append("collision counterpart is not a road car/truck")
        if not collision_visible:
            hard_reasons.append("collision timestamp is outside the released frame range")
        if negative_hits:
            hard_reasons.extend(f"strong negative: {reason}" for reason in negative_hits)
        if counterpart_mismatch:
            hard_reasons.append("free-text counterpart conflicts with car/truck taxonomy")

        explicit_intrusion = structured_intrusion or bool(intrusion_hits)
        if hard_reasons:
            row["filter_decision"] = "AUTO_REJECT"
            row["filter_reason"] = hard_reasons
        elif (
            road_vehicle
            and explicit_intrusion
            and bool(lane_hits)
            and pre_visible
            and accident_type not in rules["ambiguous_overtaking_types"]
            and score >= 10
        ):
            row["filter_decision"] = "HIGH_CONFIDENCE_LABEL_QUEUE"
            row["filter_reason"] = ["ego-involved road-vehicle collision with explicit intrusion and valid temporal context"]
        else:
            row["filter_decision"] = "BORDERLINE_REVIEW"
            reasons = []
            if not explicit_intrusion:
                reasons.append("metadata does not explicitly establish lateral intrusion")
            if not pre_visible:
                reasons.append("pre-entry visibility cannot be established from temporal metadata")
            if accident_type in rules["ambiguous_overtaking_types"]:
                reasons.append("overtaking taxonomy does not prove the other vehicle entered ego's path")
            row["filter_reason"] = reasons or ["metadata support is incomplete"]
        rows.append(row)
    return rows


def infer_ego_agents(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        agent
        for agent in agents
        if agent.get("spatial_location") == "ego"
        or "ego" in str(agent.get("id", "")).lower()
        or agent.get("role") == "ego"
    ]


def collision_step(record: dict[str, Any], ego_ids: set[str], initiator_ids: set[str]) -> tuple[float | None, list[str]]:
    hits = []
    for step in record.get("level_3_explanation_counterfactual", {}).get("causal_chain", []):
        event = normalized_text(step.get("event"))
        actors = {str(actor) for actor in step.get("actors", [])}
        collision_words = re.search(r"\b(collid|impact|hits?|strikes?|clips?|contact|rear-ends?)", event)
        involves_both = bool(actors & ego_ids) and bool(actors & initiator_ids)
        if collision_words and involves_both:
            time = step.get("time_sec")
            if isinstance(time, (int, float)) and math.isfinite(float(time)):
                hits.append((float(time), event))
    if not hits:
        return None, []
    hits.sort()
    return hits[0][0], [event for _, event in hits]


def normalize_causalcrash(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = root / "metadata/causalcrash"
    annotations = json.loads((metadata / "annotations.json").read_text(encoding="utf-8"))
    rules = config["causalcrash"]
    rows = []
    for source in annotations:
        source_id = str(source.get("video_id") or source["video_metadata"]["video_id"]).removesuffix(".mp4")
        url = source.get("youtube_url", "")
        video = source.get("video_metadata", {})
        perception = source.get("level_1_perception", {})
        predictive = source.get("level_2_predictive", {})
        agents = perception.get("agents", [])
        ego = infer_ego_agents(agents)
        ego_ids = {str(agent.get("id")) for agent in ego}
        initiators = [agent for agent in agents if agent.get("role") == "initiator" and str(agent.get("id")) not in ego_ids]
        initiator_ids = {str(agent.get("id")) for agent in initiators}
        ego_intents = {agent.get("initial_state", {}).get("intent") for agent in ego}
        ego_roles = {agent.get("role") for agent in ego}
        initiator_intents = {agent.get("initial_state", {}).get("intent") for agent in initiators}
        initiator_types = {agent.get("type") for agent in initiators}
        hazard = set(video.get("hazard_type", []))
        outcome = predictive.get("predicted_outcome", {}).get("label")
        fps = float(video.get("fps") or 0)
        duration = float(video.get("duration_sec") or 0)
        temporal = predictive.get("temporal_window", {})
        active = temporal.get("active_phase_sec") or []
        active_start = float(active[0]) if active and isinstance(active[0], (int, float)) else None
        chain_collision, chain_events = collision_step(source, ego_ids, initiator_ids)
        critical = predictive.get("critical_point_time_sec")
        critical = float(critical) if isinstance(critical, (int, float)) else None
        collision_time = chain_collision if chain_collision is not None else critical
        descriptive_text = normalized_text(
            video.get("notes"),
            perception.get("notes"),
            predictive.get("notes"),
            predictive.get("predicted_outcome", {}).get("summary"),
        )
        invalid_camera_hits = matches(rules["invalid_camera_patterns"], descriptive_text)

        row = base_row("causalcrash", source_id, url)
        row.update(
            fps=fps or "",
            num_frames=round(fps * duration) if fps > 0 and duration > 0 else "",
            duration_s=duration or "",
            metadata_accident_type=";".join(sorted(hazard)),
            metadata_accident_start=active_start if active_start is not None else "",
            metadata_collision_time=collision_time if collision_time is not None else "",
            metadata_collision_frame_candidate=round(collision_time * fps) if collision_time is not None and fps > 0 else "",
            causalcrash_outcome=outcome or "",
            causalcrash_agents=agents,
            causalcrash_causal_chain=source.get("level_3_explanation_counterfactual", {}).get("causal_chain", []),
            causalcrash_notes=" | ".join(
                normalized_text(value)
                for value in (
                    video.get("notes"),
                    perception.get("notes"),
                    predictive.get("notes"),
                    predictive.get("predicted_outcome", {}).get("summary"),
                )
                if value
            ),
        )
        actual_collision = bool(hazard & set(rules["collision_hazards"])) and outcome in rules["collision_labels"]
        ego_victim = bool(ego_roles & set(rules["ego_positive_roles"])) and "initiator" not in ego_roles and "observer" not in ego_roles
        ego_lane_keep = bool(ego_intents & set(rules["ego_intents"]))
        other_vehicle = bool(initiator_types & set(rules["road_vehicle_types"]))
        ego_vehicle_type = bool({agent.get("type") for agent in ego} & set(rules["road_vehicle_types"]))
        lateral_intent = bool(initiator_intents & set(rules["intrusion_intents"]))
        intrusion_hazard = bool(hazard & set(rules["intrusion_hazards"]))
        temporal_valid = collision_time is not None and 0 < collision_time < duration
        pre_visible = bool(
            temporal_valid
            and active_start is not None
            and active_start < collision_time
            and collision_time - active_start >= rules["minimum_pre_collision_seconds"]
        )
        collision_involves_both = bool(chain_events)

        if ego:
            row["ego_view_evidence"] = [f"explicit ego agent(s): {', '.join(sorted(ego_ids))}; camera field is absent"]
        if ego_victim:
            row["ego_involved_evidence"] = ["explicit ego agent has victim role", *chain_events]
        if ego_lane_keep:
            row["ego_lane_keep_evidence"] = [f"ego intent: {', '.join(sorted(str(x) for x in ego_intents if x))}"]
        if other_vehicle and lateral_intent:
            row["other_vehicle_intrusion_evidence"] = [
                f"initiator type(s): {', '.join(sorted(str(x) for x in initiator_types if x))}",
                f"initiator intent(s): {', '.join(sorted(str(x) for x in initiator_intents if x))}",
            ]
        if intrusion_hazard:
            row["other_vehicle_intrusion_evidence"].append(f"hazard label(s): {', '.join(sorted(hazard & set(rules['intrusion_hazards'])))}")
        if pre_visible:
            row["entry_visible_evidence"] = [f"active hazard starts {collision_time - active_start:.2f}s before collision anchor"]
        if temporal_valid:
            row["collision_visible_evidence"] = [f"collision anchor {collision_time:.3f}s is within duration {duration:.3f}s"]

        score = int(actual_collision) * 2 + int(bool(ego)) + int(ego_victim) * 2
        score += int(ego_lane_keep) * 2 + int(other_vehicle) + int(lateral_intent) * 2
        score += int(intrusion_hazard) + int(collision_involves_both) * 2 + int(pre_visible) + int(temporal_valid)
        row["filter_score"] = score
        row["positive_evidence"] = (
            row["ego_view_evidence"]
            + row["ego_involved_evidence"]
            + row["ego_lane_keep_evidence"]
            + row["other_vehicle_intrusion_evidence"]
            + row["entry_visible_evidence"]
            + row["collision_visible_evidence"]
        )
        hard_reasons = []
        if not video.get("hazard_present"):
            hard_reasons.append("metadata says no hazard")
        if not actual_collision:
            hard_reasons.append("metadata does not establish an actual collision outcome")
        if ego and ("observer" in ego_roles or "initiator" in ego_roles):
            hard_reasons.append(f"ego role is incompatible: {', '.join(sorted(str(x) for x in ego_roles))}")
        if ego and not ego_vehicle_type:
            hard_reasons.append("ego agent is not a road car/truck")
        if ego and not other_vehicle:
            hard_reasons.append("initiator is not a road car/truck")
        if ego and ego_intents and not ego_lane_keep:
            hard_reasons.append(f"ego intent is not lane keeping: {', '.join(sorted(str(x) for x in ego_intents if x))}")
        if not temporal_valid:
            hard_reasons.append("collision anchor is absent or outside video duration")
        if invalid_camera_hits:
            hard_reasons.append(f"non-forward/non-ego camera evidence: {', '.join(invalid_camera_hits)}")
        row["negative_evidence"] = hard_reasons.copy()

        if hard_reasons:
            row["filter_decision"] = "AUTO_REJECT"
            row["filter_reason"] = hard_reasons
        elif all((ego, ego_vehicle_type, ego_victim, ego_lane_keep, other_vehicle, lateral_intent, collision_involves_both, pre_visible)) and score >= 14:
            row["filter_decision"] = "HIGH_CONFIDENCE_LABEL_QUEUE"
            row["filter_reason"] = ["explicit ego victim keeps lane while road-vehicle initiator moves laterally; causal chain includes their collision"]
        else:
            row["filter_decision"] = "BORDERLINE_REVIEW"
            reasons = []
            if not ego:
                reasons.append("no explicit ego/camera agent in metadata")
            if not ego_victim:
                reasons.append("ego collision involvement is not explicit")
            if not ego_lane_keep:
                reasons.append("ego lane-keeping intent is not explicit")
            if not other_vehicle or not lateral_intent:
                reasons.append("road-vehicle initiator with lateral intent is not explicit")
            if not collision_involves_both:
                reasons.append("causal chain does not explicitly name a collision between ego and initiator")
            if not pre_visible:
                reasons.append("metadata cannot establish visible pre-entry time")
            row["filter_reason"] = reasons or ["metadata support is incomplete"]
        rows.append(row)
    return rows


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.with_suffix(".jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))

    mmau = normalize_mmau(arguments.root, config)
    causalcrash = normalize_causalcrash(arguments.root, config)
    write_rows(arguments.root / "metadata/mmau/normalized", mmau)
    write_rows(arguments.root / "metadata/causalcrash/normalized", causalcrash)
    combined = mmau + causalcrash
    names = {
        "HIGH_CONFIDENCE_LABEL_QUEUE": "high_confidence_metadata",
        "BORDERLINE_REVIEW": "borderline_review",
        "AUTO_REJECT": "auto_rejected",
    }
    for decision, name in names.items():
        selected = [row for row in combined if row["filter_decision"] == decision]
        write_rows(arguments.root / "manifests" / name, selected)
    summary = {
        "records": {"mmau": len(mmau), "causalcrash": len(causalcrash)},
        "decisions": {
            source: {
                decision: sum(row["source"] == source and row["filter_decision"] == decision for row in combined)
                for decision in DECISIONS
            }
            for source in ("mmau", "causalcrash")
        },
    }
    (arguments.root / "reports").mkdir(parents=True, exist_ok=True)
    (arguments.root / "reports/filter_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
