"""Validated readers for persisted split-pipeline stage artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .recognized_boundaries import RecognizedBoundaries


def read_stage_json(run_dir: Path, stage: str) -> dict:
    path = Path(run_dir) / f"{stage}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing required stage artifact: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("stage") != stage or record.get("status") != "completed":
        raise ValueError(f"stage artifact is not completed: {path}")
    return record.get("result", {})


def read_stage_arrays(run_dir: Path, stage: str) -> dict[str, np.ndarray]:
    run_dir = Path(run_dir)
    archive_path = run_dir / f"{stage}.npz"
    manifest = read_stage_json(run_dir, f"{stage}_manifest")
    if manifest.get("artifact") != archive_path.name or not archive_path.is_file():
        raise ValueError(f"stage artifact manifest does not match {archive_path}")
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != manifest.get("sha256"):
        raise ValueError(f"stage artifact checksum mismatch: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    declared = manifest.get("arrays", {})
    if set(arrays) != set(declared):
        raise ValueError(f"stage artifact arrays do not match manifest: {archive_path}")
    for name, value in arrays.items():
        record = declared[name]
        if list(value.shape) != record.get("shape") or str(value.dtype) != record.get("dtype"):
            raise ValueError(f"stage artifact array metadata mismatch: {archive_path}:{name}")
    return arrays


def restore_boundaries(
    arrays: dict[str, np.ndarray], summary: dict
) -> RecognizedBoundaries:
    required = {
        "boundary_vertex_ids", "boundary_points", "boundary_loop_offsets",
        "component_loop_offsets",
    }
    if not required.issubset(arrays):
        raise ValueError("03 boundary artifact is missing required arrays")
    ids = arrays["boundary_vertex_ids"]
    points = arrays["boundary_points"]
    loop_offsets = arrays["boundary_loop_offsets"]
    component_offsets = arrays["component_loop_offsets"]
    if len(ids) != len(points) or points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("03 boundary artifact has inconsistent point and vertex arrays")
    if len(loop_offsets) == 0 or len(component_offsets) == 0:
        raise ValueError("03 boundary artifact has empty offsets")
    loops: list[tuple[tuple[int, ...], ...]] = []
    point_loops: list[tuple[tuple[tuple[float, float, float], ...], ...]] = []
    for component_index in range(len(component_offsets) - 1):
        component_loops = []
        component_points = []
        for loop_index in range(
            int(component_offsets[component_index]),
            int(component_offsets[component_index + 1]),
        ):
            start, end = map(int, loop_offsets[loop_index : loop_index + 2])
            component_loops.append(tuple(map(int, ids[start:end])))
            component_points.append(tuple(tuple(map(float, point)) for point in points[start:end]))
        loops.append(tuple(component_loops))
        point_loops.append(tuple(component_points))
    boundary_summary = summary.get("recognized_boundaries", {})
    return RecognizedBoundaries(
        component_loops=tuple(loops),
        component_loop_points=tuple(point_loops),
        source_vertex_count=int(boundary_summary["source_vertex_count"]),
        source_face_count=int(boundary_summary["source_face_count"]),
        fingerprint=str(boundary_summary["fingerprint"]),
        simplification_records=tuple(boundary_summary.get("simplification_records", [])),
        excluded_components=tuple(summary.get("recognition_excluded_regions", [])),
    )
