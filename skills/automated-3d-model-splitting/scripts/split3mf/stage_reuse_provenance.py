"""Audit the inputs that produced frozen Stage 02-04 artifacts."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from .stage_cache import sha256_file


_UPSTREAM_MODULES = (
    "project.py",
    "recognition.py",
    "recognition_metadata.py",
    "region_review.py",
    "semantic_partition.py",
    "source_region_review.py",
    "boundary_review.py",
    "application/boundary_snapshot_builder.py",
    "application/recognized_boundaries.py",
    "application/recognition_review.py",
    "contact_interface_planner.py",
    "boundary_matching.py",
)

_UPSTREAM_OPTIONS = (
    "model_entry", "format_profile", "color_map_json",
    "boundary_review_json", "region_review_json", "recognition_review_json",
    "noise_review_max_faces", "small_region_review_max_faces",
    "max_region_review_candidates",
    "exterior_view_count", "exterior_depth_map_resolution",
    "exterior_depth_tolerance_mm", "visual_semantics_json",
    "visual_semantic_min_confidence", "region_review_resolution",
)


def upstream_code_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _UPSTREAM_MODULES:
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    # The orchestration file also contains Stage 05. Its prefix contains the
    # source read, recognition and assembly planner, so Stage 05 edits do not
    # invalidate a frozen Stage 04 plan.
    from .pipeline import SplitPipeline

    stage_prefix, separator, _ = inspect.getsource(SplitPipeline.run).partition(
        "        from .interface_assembly import ("
    )
    if not separator:
        raise ValueError("cannot locate Stage 05 boundary in pipeline source")
    digest.update(stage_prefix.encode("utf-8"))
    return digest.hexdigest()


def _option_value(name: str, value: Any) -> Any:
    if isinstance(value, Path):
        value = str(value)
    if name.endswith("_json") and value:
        path = Path(value).expanduser().resolve()
        return {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None}
    return value


def upstream_options(namespace: Any) -> dict[str, Any]:
    return {
        name: _option_value(name, getattr(namespace, name, None))
        for name in _UPSTREAM_OPTIONS
    }


def provenance_record(namespace: Any, source_sha256: str) -> dict[str, Any]:
    options = upstream_options(namespace)
    encoded = json.dumps(options, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {
        "schema": "stage-02-04-provenance/v1",
        "source_sha256": source_sha256,
        "upstream_options": options,
        "upstream_options_sha256": hashlib.sha256(encoded).hexdigest(),
        "upstream_code_sha256": upstream_code_fingerprint(),
    }


def validate_provenance(record: dict, source_sha256: str) -> None:
    if record.get("schema") != "stage-02-04-provenance/v1":
        raise ValueError("unsupported Stage 02-04 provenance schema")
    if record.get("source_sha256") != source_sha256:
        raise ValueError("Stage 02-04 provenance source hash mismatch")
    options = record.get("upstream_options")
    if not isinstance(options, dict):
        raise ValueError("Stage 02-04 provenance options are missing")
    encoded = json.dumps(options, sort_keys=True, ensure_ascii=False).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != record.get("upstream_options_sha256"):
        raise ValueError("Stage 02-04 provenance options hash mismatch")
    for name, value in options.items():
        if name.endswith("_json") and isinstance(value, dict):
            path = Path(str(value.get("path", "")))
            if not path.is_file() or sha256_file(path) != value.get("sha256"):
                raise ValueError(f"Stage 02-04 input changed: {name}")
    if upstream_code_fingerprint() != record.get("upstream_code_sha256"):
        raise ValueError("Stage 02-04 producer code changed; regenerate upstream artifacts")
