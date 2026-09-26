from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .application import RecognizedBoundaries, StageArtifactStore
from .common import Component
from .interface_assembly import build_pairwise_interface_surfaces


def _read_stage_json(run_dir: Path, stage: str) -> dict:
    path = run_dir / f"{stage}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing required stage artifact: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("stage") != stage or record.get("status") != "completed":
        raise ValueError(f"stage artifact is not completed: {path}")
    return record.get("result", {})


def _read_stage_arrays(run_dir: Path, stage: str) -> dict[str, np.ndarray]:
    archive_path = run_dir / f"{stage}.npz"
    manifest = _read_stage_json(run_dir, f"{stage}_manifest")
    expected_name = manifest.get("artifact")
    expected_digest = manifest.get("sha256")
    if expected_name != archive_path.name or not archive_path.is_file():
        raise ValueError(f"stage artifact manifest does not match {archive_path}")
    actual_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if actual_digest != expected_digest:
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


def _restore_boundaries(
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


def resume_interface_stage(
    source_run_dir: Path,
    output_root: Path,
    *,
    scale_ratio: float = 0.50,
    clearance_mm: float = 0.20,
) -> Path:
    """Run Stage 05 from completed 02/03/04 artifacts without rereading the 3MF."""
    source_run_dir = Path(source_run_dir).expanduser().resolve()
    loaded_summary = _read_stage_json(source_run_dir, "02_loaded_project_summary")
    recognition_summary = _read_stage_json(source_run_dir, "03_recognition_summary")
    assembly_record = _read_stage_json(source_run_dir, "04_assembly_plan")
    loaded_arrays = _read_stage_arrays(source_run_dir, "02_loaded_project")
    region_arrays = _read_stage_arrays(source_run_dir, "03_recognition_regions")
    boundary_arrays = _read_stage_arrays(source_run_dir, "03_recognized_boundaries")

    vertices = loaded_arrays.get("vertices")
    faces = loaded_arrays.get("faces")
    if vertices is None or faces is None:
        raise ValueError("02 loaded-project artifact must contain vertices and faces")
    regions = sorted(
        recognition_summary.get("regions", []),
        key=lambda item: int(item["part_index"]),
    )
    components: list[Component] = []
    for record in regions:
        index = int(record["part_index"])
        if index != len(components) + 1:
            raise ValueError("03 region IDs must be contiguous and one-based for Stage 04")
        key = f"region_{index:04d}_source_face_ids"
        if key not in region_arrays:
            raise ValueError(f"03 region artifact is missing {key}")
        global_faces = np.asarray(region_arrays[key], dtype=np.int64)
        components.append(Component(
            color_code=str(record.get("color_code", "DEFAULT")),
            global_faces=global_faces,
            face_count=int(record.get("faces", len(global_faces))),
            area=float(record.get("area_mm2", 0.0)),
            bbox_min=np.asarray(record["bbox_min"], dtype=np.float64),
            bbox_max=np.asarray(record["bbox_max"], dtype=np.float64),
            center=np.asarray(record["center"], dtype=np.float64),
        ))
    if len(components) != len(regions):
        raise ValueError("03 recognition summary and region arrays are inconsistent")
    boundaries = _restore_boundaries(boundary_arrays, recognition_summary)
    interface_plan = assembly_record
    if interface_plan.get("schema") != "contact-interface-plan/v1":
        raise ValueError("04 assembly artifact is not a contact-interface-plan/v1 plan")
    if interface_plan.get("recognized_boundary_fingerprint") != boundaries.fingerprint:
        raise ValueError("04 plan and 03 frozen-boundary fingerprints do not match")

    store = StageArtifactStore(Path(output_root).expanduser())
    arrays, summary = build_pairwise_interface_surfaces(
        interface_plan,
        vertices=vertices,
        faces=faces,
        components=components,
        scale_ratio=scale_ratio,
        clearance_mm=clearance_mm,
    )
    if arrays:
        store.write_arrays("05_interface_surfaces", **arrays)
    summary.update({
        "stage_status": "interface_surfaces_constructed",
        "part_mesh_build_status": "pending_direct_frozen_boundary_integration",
        "frozen_boundary_fingerprint": boundaries.fingerprint,
        "quality_gates_blocking": False,
        "resumed_from_stage_artifacts": str(source_run_dir),
    })
    store.write_json(
        "05_interface_assembly_summary",
        summary,
        inputs={
            "source_run_dir": str(source_run_dir),
            "interface_plan_schema": interface_plan["schema"],
            "recognized_boundary_fingerprint": boundaries.fingerprint,
            "scale_ratio": float(scale_ratio),
            "clearance_mm": float(clearance_mm),
        },
    )
    return store.run_dir
