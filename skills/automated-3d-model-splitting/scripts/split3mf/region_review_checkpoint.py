"""Persist the expensive recognition work completed before source-region review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .application.stage_artifact_reader import read_stage_arrays, read_stage_json
from .application.stage_artifacts import StageArtifactStore
from .stage_cache import sha256_file
from .stage_reuse_provenance import upstream_code_fingerprint


SCHEMA = "region-review-checkpoint/v1"
_OPTIONS = (
    "model_entry", "format_profile", "noise_review_max_faces",
    "small_region_review_max_faces", "max_region_review_candidates",
    "exterior_view_count", "exterior_depth_map_resolution",
    "exterior_depth_tolerance_mm", "region_review_resolution",
)


def _options(args: Any) -> dict:
    options = {name: getattr(args, name) for name in _OPTIONS}
    for name in ("color_map_json", "boundary_review_json"):
        value = getattr(args, name, None)
        if value:
            path = Path(value).expanduser().resolve()
            options[name] = {"path": str(path), "sha256": sha256_file(path)}
        else:
            options[name] = None
    return options


def _flatten(groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([len(group) for group in groups], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    faces = np.concatenate(groups).astype(np.int64, copy=False) if groups else np.empty(0, dtype=np.int64)
    return faces, offsets


def _restore(faces: np.ndarray, offsets: np.ndarray, face_count: int) -> list[np.ndarray]:
    if (offsets.ndim != 1 or len(offsets) < 1 or int(offsets[0]) != 0
            or int(offsets[-1]) != len(faces) or np.any(np.diff(offsets) < 0)
            or (len(faces) and (int(faces.min()) < 0 or int(faces.max()) >= face_count))):
        raise ValueError("region-review checkpoint has invalid group offsets or face IDs")
    return [faces[int(start):int(end)] for start, end in zip(offsets[:-1], offsets[1:])]


@dataclass(frozen=True)
class RegionReviewCheckpoint:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray
    project_settings: dict
    source_sha256: str
    visible_faces: np.ndarray
    recognition_token_colors: np.ndarray
    recognition_colors: np.ndarray
    groups: list[np.ndarray]
    raw_group_count: int
    automatic_noise_exclusions: list[dict]
    review_candidates: list[tuple[np.ndarray, str]]
    review_limit_noise_exclusions: list[dict]
    review_records: list[dict]


def write_region_review_checkpoint(
    store: StageArtifactStore,
    *,
    source_sha256: str,
    args: Any,
    visible_faces: np.ndarray,
    recognition_token_colors: list[str] | np.ndarray,
    recognition_colors: list[str] | np.ndarray,
    groups: list[np.ndarray],
    raw_group_count: int,
    automatic_noise_exclusions: list[dict],
    review_candidates: list[tuple[np.ndarray, str]],
    review_limit_noise_exclusions: list[dict],
    review_records: list[dict],
) -> None:
    group_faces, group_offsets = _flatten(groups)
    candidate_faces, candidate_offsets = _flatten([group for group, _ in review_candidates])
    store.write_arrays(
        "03_region_review_checkpoint",
        visible_faces=np.asarray(visible_faces, dtype=np.bool_),
        recognition_token_colors=np.asarray(recognition_token_colors, dtype="U"),
        recognition_colors=np.asarray(recognition_colors, dtype="U"),
        group_faces=group_faces,
        group_offsets=group_offsets,
        candidate_faces=candidate_faces,
        candidate_offsets=candidate_offsets,
    )
    store.write_json("03_region_review_checkpoint", {
        "schema": SCHEMA,
        "source_sha256": source_sha256,
        "options": _options(args),
        "implementation_sha256": upstream_code_fingerprint(),
        "raw_group_count": raw_group_count,
        "automatic_noise_exclusions": automatic_noise_exclusions,
        "candidate_categories": [category for _, category in review_candidates],
        "review_limit_noise_exclusions": review_limit_noise_exclusions,
        "review_records": review_records,
    })


def read_region_review_checkpoint(run_dir: Path, source: Path, args: Any) -> RegionReviewCheckpoint:
    run_dir = Path(run_dir).expanduser().resolve()
    summary, loaded = read_saved_stage02(run_dir, source)
    source_sha256 = summary["source_sha256"]
    record = read_stage_json(run_dir, "03_region_review_checkpoint")
    if record.get("schema") != SCHEMA or record.get("source_sha256") != source_sha256:
        raise ValueError("region-review checkpoint schema or source hash mismatch")
    if record.get("options") != _options(args):
        raise ValueError("region-review checkpoint options changed; rerun recognition")
    if record.get("implementation_sha256") != upstream_code_fingerprint():
        raise ValueError("region-review checkpoint producer code changed; rerun recognition")
    saved = read_stage_arrays(run_dir, "03_region_review_checkpoint")
    vertices, faces, colors = loaded["vertices"], loaded["faces"], loaded["face_color_tokens"]
    face_count = len(faces)
    if (len(colors) != face_count or len(saved["visible_faces"]) != face_count
            or len(saved["recognition_token_colors"]) != face_count
            or len(saved["recognition_colors"]) != face_count):
        raise ValueError("region-review checkpoint face arrays do not match Stage 02")
    groups = _restore(saved["group_faces"], saved["group_offsets"], face_count)
    candidates = _restore(saved["candidate_faces"], saved["candidate_offsets"], face_count)
    categories = record.get("candidate_categories", [])
    records = record.get("review_records", [])
    if len(categories) != len(candidates) or len(records) != len(candidates):
        raise ValueError("region-review checkpoint candidates are inconsistent")
    if any(not len(group) or int(group.min()) != int(item.get("source_min_face_index", -1))
           for group, item in zip(candidates, records)):
        raise ValueError("region-review checkpoint candidate identity mismatch")
    return RegionReviewCheckpoint(
        vertices=vertices, faces=faces, colors=colors,
        project_settings=summary["project_settings"], source_sha256=source_sha256,
        visible_faces=saved["visible_faces"],
        recognition_token_colors=saved["recognition_token_colors"],
        recognition_colors=saved["recognition_colors"], groups=groups,
        raw_group_count=int(record["raw_group_count"]),
        automatic_noise_exclusions=record["automatic_noise_exclusions"],
        review_candidates=list(zip(candidates, categories)),
        review_limit_noise_exclusions=record["review_limit_noise_exclusions"],
        review_records=records,
    )


def read_saved_stage02(run_dir: Path, source: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Validate and load the source mesh even when an old run has no review checkpoint."""
    run_dir = Path(run_dir).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    summary = read_stage_json(run_dir, "02_loaded_project_summary")
    if Path(str(summary.get("source", ""))).resolve() != source:
        raise ValueError("saved Stage 02 mesh belongs to a different source path")
    if summary.get("source_sha256") != sha256_file(source):
        raise ValueError("source 3MF changed since Stage 02")
    loaded = read_stage_arrays(run_dir, "02_loaded_project")
    if len(loaded["faces"]) != int(summary.get("face_count", -1)):
        raise ValueError("saved Stage 02 face count does not match its summary")
    return summary, loaded
