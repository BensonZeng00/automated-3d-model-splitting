from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..common import Component


RECOGNITION_REVIEW_SCHEMA_VERSION = 1


def component_fingerprint(components: list[Component]) -> str:
    digest = hashlib.sha256()
    for index, component in enumerate(components, start=1):
        digest.update(np.asarray([index], dtype=np.int64).tobytes())
        digest.update(str(component.color_code).encode("utf-8"))
        digest.update(np.asarray(component.global_faces, dtype=np.int64).tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_recognition_review(
    path: Path | None,
    *,
    source_path: Path,
    source_face_count: int,
    components: list[Component],
) -> dict:
    fingerprint = component_fingerprint(components)
    if path is None:
        return {
            "source": str(source_path.resolve()),
            "source_face_count": int(source_face_count),
            "recognition_fingerprint": fingerprint,
            "actions": [],
            "reference_regions": _region_records(components),
            "user_confirmed": False,
            "expected_result_fingerprint": None,
        }
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recognition review JSON must contain an object")
    if int(payload.get("schema_version", -1)) != RECOGNITION_REVIEW_SCHEMA_VERSION:
        raise ValueError("recognition review JSON has an unsupported schema_version")
    if Path(str(payload.get("source", ""))).resolve() != source_path.resolve():
        raise ValueError("recognition review JSON belongs to a different source file")
    if int(payload.get("source_face_count", -1)) != int(source_face_count):
        raise ValueError("recognition review JSON source_face_count does not match the source")
    if payload.get("recognition_fingerprint") != fingerprint:
        raise ValueError("recognition review JSON is stale for the current recognized regions")
    actions = payload.get("actions", [])
    if not isinstance(actions, list):
        raise ValueError("recognition review JSON actions must be an array")
    return {
        "source": str(source_path.resolve()),
        "source_face_count": int(source_face_count),
        "recognition_fingerprint": fingerprint,
        "actions": _validate_actions(actions, len(components)),
        "reference_regions": _region_records(components),
        "user_confirmed": payload.get("user_confirmed") is True,
        "expected_result_fingerprint": payload.get("expected_result_fingerprint"),
    }


def apply_recognition_actions(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    actions: list[dict],
) -> list[Component]:
    if not actions:
        return list(components)
    by_part: dict[int, dict] = {}
    for action in actions:
        for part_index in action["parts"]:
            by_part[part_index] = action

    result: list[Component] = []
    for original_index, component in enumerate(components, start=1):
        action = by_part.get(original_index)
        if action is None:
            result.append(component)
            continue
        if action["action"] == "delete":
            continue
        if original_index != action["keep"]:
            continue
        merged_faces = np.unique(np.concatenate([
            np.asarray(components[index - 1].global_faces, dtype=np.int64)
            for index in action["parts"]
        ]))
        expected_count = sum(components[index - 1].face_count for index in action["parts"])
        if len(merged_faces) != expected_count:
            raise ValueError("recognition merge contains overlapping source faces")
        selected_faces = np.asarray(faces, dtype=np.int64)[merged_faces]
        points = np.asarray(vertices, dtype=np.float64)[selected_faces].reshape((-1, 3))
        normals = np.cross(
            np.asarray(vertices)[selected_faces[:, 1]] - np.asarray(vertices)[selected_faces[:, 0]],
            np.asarray(vertices)[selected_faces[:, 2]] - np.asarray(vertices)[selected_faces[:, 0]],
        )
        result.append(Component(
            color_code=str(components[action["keep"] - 1].color_code),
            global_faces=merged_faces,
            face_count=int(len(merged_faces)),
            area=float((np.linalg.norm(normals, axis=1) * 0.5).sum()),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            center=points.mean(axis=0),
        ))
    if not result:
        raise ValueError("recognition review cannot delete every region")
    return result


def result_fingerprint(components: list[Component], boundaries) -> str:
    digest = hashlib.sha256(component_fingerprint(components).encode("ascii"))
    digest.update(str(boundaries.fingerprint).encode("ascii"))
    return digest.hexdigest()


def write_recognition_review_template(
    path: Path,
    *,
    review: dict,
    result_fingerprint_value: str,
    components: list[Component],
    reference_regions: list[dict] | None = None,
    user_confirmed: bool = False,
) -> None:
    payload = {
        "schema_version": RECOGNITION_REVIEW_SCHEMA_VERSION,
        "source": review["source"],
        "source_face_count": int(review["source_face_count"]),
        "recognition_fingerprint": review["recognition_fingerprint"],
        "expected_result_fingerprint": str(result_fingerprint_value),
        "user_confirmed": bool(user_confirmed),
        "actions": list(review["actions"]),
        "instruction": (
            "Review the recognition report and preview. Use actions=[] if correct. "
            'For corrections, use {"action":"delete","parts":["P08"]} or '
            '{"action":"merge","parts":["P02","P03"],"keep":"P02"}. After applying '
            "corrections, review the regenerated report, then set user_confirmed=true."
        ),
        "action_reference_regions": list(reference_regions or review.get("reference_regions", [])),
        "revised_regions": _region_records(components),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _validate_actions(actions: list, component_count: int) -> list[dict]:
    normalized = []
    touched: set[int] = set()
    for position, raw_action in enumerate(actions, start=1):
        if not isinstance(raw_action, dict):
            raise ValueError(f"recognition action {position} must be an object")
        kind = str(raw_action.get("action", "")).strip().lower()
        if kind not in {"delete", "merge"}:
            raise ValueError(f"recognition action {position} must be delete or merge")
        raw_parts = raw_action.get("parts")
        if not isinstance(raw_parts, list):
            raise ValueError(f"recognition action {position} parts must be an array")
        indices = [_part_index(value) for value in raw_parts]
        if len(indices) != len(set(indices)) or not indices:
            raise ValueError(f"recognition action {position} must list unique part ids")
        if kind == "delete" and len(indices) < 1:
            raise ValueError("delete action requires at least one part")
        if kind == "merge" and len(indices) < 2:
            raise ValueError("merge action requires at least two parts")
        invalid = [index for index in indices if index > component_count]
        if invalid:
            raise ValueError("recognition action references unknown parts: " + ", ".join(
                f"P{index:02d}" for index in invalid
            ))
        overlap = touched.intersection(indices)
        if overlap:
            raise ValueError("a part cannot appear in multiple recognition actions")
        touched.update(indices)
        action = {"action": kind, "parts": indices}
        if kind == "merge":
            keep = _part_index(raw_action.get("keep", f"P{indices[0]:02d}"))
            if keep not in indices:
                raise ValueError("merge keep must be one of the merged parts")
            action["keep"] = keep
        reason = str(raw_action.get("reason", "")).strip()
        if reason:
            action["reason"] = reason
        normalized.append(action)
    deleted_count = sum(len(action["parts"]) for action in normalized if action["action"] == "delete")
    merged_count_reduction = sum(
        len(action["parts"]) - 1 for action in normalized if action["action"] == "merge"
    )
    if component_count - deleted_count - merged_count_reduction < 1:
        raise ValueError("recognition actions must leave at least one region")
    return normalized


def _region_records(components: list[Component]) -> list[dict]:
    return [
        {
            "part": f"P{index:02d}",
            "source_min_face_index": int(np.min(component.global_faces)),
            "face_count": int(component.face_count),
            "color_code": str(component.color_code),
        }
        for index, component in enumerate(components, start=1)
    ]


def _part_index(value) -> int:
    token = str(value).strip().upper()
    if token.startswith("P"):
        token = token[1:]
    if not token.isdigit() or int(token) < 1:
        raise ValueError(f"invalid part id: {value!r}")
    return int(token)
