"""Visual labels and source-region reporting for the active recognition stage."""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np

from .common import COLOR_INFO, Component
from .mesh import boundary_loops, build_local_mesh
from .project import color_resolution_status


_CONFIDENCE = {
    "VERYLOW": 0.1,
    "LOW": 0.35,
    "MED": 0.65,
    "MEDIUM": 0.65,
    "HIGH": 0.9,
    "UNKNOWN": 0.0,
}


def confidence_score(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return _CONFIDENCE.get(str(value).strip().upper(), default)


def _part_index(value: object) -> int | None:
    if value is None:
        return None
    match = re.search(r"p?0*([0-9]+)", str(value).strip(), re.IGNORECASE)
    return int(match.group(1)) if match else None


def load_visual_semantics(path: Path | None) -> dict:
    if path is None:
        return {"source": None, "parts": {}, "physical_partitions": []}
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_parts = raw.get("parts", {})
    if isinstance(raw_parts, dict):
        entries = raw_parts.items()
    elif isinstance(raw_parts, list):
        entries = (
            (item.get("part") or item.get("part_id") or item.get("part_index"), item)
            for item in raw_parts if isinstance(item, dict)
        )
    else:
        entries = ()
    parts = {}
    for key, value in entries:
        index = _part_index(key)
        if index is None or not isinstance(value, dict):
            continue
        confidence = value.get("confidence", "UNKNOWN")
        parts[index] = {
            "part_index": index,
            "label": str(value.get("label") or value.get("name") or value.get("type") or value.get("semantic_label") or ""),
            "description": str(value.get("description") or value.get("role") or value.get("semantic_description") or ""),
            "confidence": confidence,
            "confidence_score": confidence_score(confidence),
            "visual_evidence": value.get("visual_evidence") or value.get("evidence") or "",
        }
    partitions = []
    for item in raw.get("physical_partitions", []):
        if not isinstance(item, dict):
            continue
        scope = str(item.get("scope", "part")).strip().lower()
        index = _part_index(item.get("part") or item.get("part_index"))
        plane = item.get("plane", item)
        if (scope != "all_components" and index is None) or not isinstance(plane, dict):
            continue
        confidence = item.get("confidence", "UNKNOWN")
        partitions.append({
            "part_index": index,
            "scope": scope,
            "label": str(item.get("label") or item.get("semantic_boundary") or ""),
            "origin": plane.get("origin"),
            "normal": plane.get("normal"),
            "confidence": confidence,
            "confidence_score": confidence_score(confidence),
            "user_confirmed": bool(item.get("user_confirmed", False)),
            "source_views": item.get("source_views", []),
        })
    return {"source": str(path), "parts": parts, "physical_partitions": partitions}


def annotate_recognition_with_visual_semantics(
    records: list[dict], part_semantics: dict[int, dict]
) -> list[dict]:
    annotated = []
    for record in records:
        updated = dict(record)
        semantic = part_semantics.get(int(record["part_index"]))
        if semantic:
            updated.update({
                "visual_semantic_label": semantic.get("label", ""),
                "visual_semantic_description": semantic.get("description", ""),
                "visual_semantic_confidence": semantic.get("confidence", "UNKNOWN"),
                "visual_semantic_confidence_score": semantic.get("confidence_score", 0.0),
                "visual_semantic_evidence": semantic.get("visual_evidence", ""),
            })
        annotated.append(updated)
    return annotated


def component_recognition_records(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    source_colors: list[str],
) -> list[dict]:
    records = []
    for index, component in enumerate(components, start=1):
        color_info = COLOR_INFO.get(component.color_code, {"name": component.color_code, "hex": ""})
        resolution_status = color_resolution_status(color_info)
        _, local_faces, _, _ = build_local_mesh(vertices, faces, component)
        token_counts = collections.Counter(
            str(source_colors[int(face_id)]) for face_id in component.global_faces
        )
        tokens = [
            {"token": token, "faces": int(count)}
            for token, count in sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        records.append({
            "part_index": index,
            "role": "part",
            "color_code": component.color_code,
            "raw_color_token": "|".join(item["token"] for item in tokens),
            "raw_color_tokens": tokens,
            "color_name": color_info.get("name", component.color_code),
            "color_hex": color_info.get("hex", ""),
            "filament_slot_index": color_info.get("filament_slot"),
            "color_mapping_source": color_info.get("mapping_source", "unmapped"),
            "color_resolution_status": resolution_status,
            "color_is_fallback": resolution_status == "fallback_estimate",
            "faces": component.face_count,
            "area_mm2": component.area,
            "bbox_min": component.bbox_min.round(6).tolist(),
            "bbox_max": component.bbox_max.round(6).tolist(),
            "bbox_size_mm": (component.bbox_max - component.bbox_min).round(6).tolist(),
            "center": component.center.round(6).tolist(),
            "boundary_loops": len(boundary_loops(local_faces)),
        })
    return records


def print_recognition(records: list[dict]) -> None:
    print("recognized_parts:", flush=True)
    for record in records:
        sx, sy, sz = record["bbox_size_mm"]
        cx, cy, cz = record["center"]
        slot_index = record.get("filament_slot_index")
        slot_text = "unmapped" if slot_index is None else f"{int(slot_index) + 1}(index={int(slot_index)})"
        line = (
            "  P{part_index:02d} role={role} color_name={color_name} color_hex={color_hex} "
            "raw_token={raw_color_token} filament_slot={slot_text} "
            "mapping_source={color_mapping_source} resolution={color_resolution_status} "
            "faces={faces} size_mm=({sx:.2f},{sy:.2f},{sz:.2f}) center=({cx:.2f},{cy:.2f},{cz:.2f}) "
            "loops={boundary_loops} processing={processing}".format(
                sx=sx, sy=sy, sz=sz, cx=cx, cy=cy, cz=cz,
                slot_text=slot_text,
                processing=record.get("selected_processing_mode", "unclassified"),
                **record,
            )
        )
        if record.get("visual_semantic_label"):
            line += f" semantic={record['visual_semantic_label']}({record.get('visual_semantic_confidence', 'UNKNOWN')})"
        print(line, flush=True)
        print("recognized_part_json=" + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
