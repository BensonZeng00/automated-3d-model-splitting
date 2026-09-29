from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .application import StageArtifactStore
from .common import Component
from .common import COLOR_CATALOG
from .interface_assembly import build_pairwise_interface_surfaces
from .interface_assembly import build_pairwise_part_meshes
from .interface_package_validation import validate_reloaded_interface_parts
from .package_io import ThreeMFWriter
from .project import build_color_info_map
from .source_face_ownership import complete_component_face_ownership, confirmed_part_ids
from .validation import ValidationService
from .stage_cache import sha256_file
from .stage_reuse_provenance import validate_provenance
from .assembly_review import ensure_assembly_review
from .stage_table_report import interface_table, write_markdown_table
from .semantic_review_artifacts import ensure_semantic_review_confirmed
from .application.stage_artifact_reader import read_stage_arrays, read_stage_json, restore_boundaries


def resume_interface_stage(
    source_run_dir: Path,
    output_root: Path,
    *,
    scale_ratio: float = 0.50,
    clearance_mm: float = 0.20,
    expected_source: Path | None = None,
    assembly_review_path: Path | None = None,
    semantic_review_path: Path | None = None,
    remove_detached_micro_shells_part_ids: frozenset[str] = frozenset(),
) -> Path:
    """Run Stage 05 from completed 02/03/04 artifacts without rereading the 3MF."""
    source_run_dir = Path(source_run_dir).expanduser().resolve()
    loaded_summary = read_stage_json(source_run_dir, "02_loaded_project_summary")
    provenance_path = source_run_dir / "04_reuse_provenance.json"
    if provenance_path.is_file():
        validate_provenance(
            read_stage_json(source_run_dir, "04_reuse_provenance"),
            str(loaded_summary.get("source_sha256", "")),
        )
    if expected_source is not None:
        expected_source = Path(expected_source).expanduser().resolve()
        recorded_source = Path(str(loaded_summary.get("source", ""))).resolve()
        if expected_source != recorded_source:
            raise ValueError(
                f"stage artifacts belong to {recorded_source}, not {expected_source}"
            )
        recorded_sha256 = loaded_summary.get("source_sha256")
        if not recorded_sha256 or sha256_file(expected_source) != recorded_sha256:
            raise ValueError("source 3MF changed since Stage 02; rerun recognition")
    recognition_summary = read_stage_json(source_run_dir, "03_recognition_summary")
    assembly_record = read_stage_json(source_run_dir, "04_assembly_plan")
    table_path = source_run_dir / "04_interface_table.md"
    if not table_path.is_file():
        write_markdown_table(
            table_path, "04 榫卯接口表",
            assembly_record.get("interface_table") or interface_table(assembly_record),
        )
    ensure_assembly_review(
        assembly_record,
        source_sha256=str(loaded_summary.get("source_sha256", "")),
        template_path=source_run_dir / "04_assembly_review.json",
        table_path=table_path,
        review_path=assembly_review_path,
    )
    ensure_semantic_review_confirmed(
        source_run_dir, semantic_review_path, recognition_summary
    )
    loaded_arrays = read_stage_arrays(source_run_dir, "02_loaded_project")
    region_arrays = read_stage_arrays(source_run_dir, "03_recognition_regions")
    boundary_arrays = read_stage_arrays(source_run_dir, "03_recognized_boundaries")

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
    boundaries = restore_boundaries(boundary_arrays, recognition_summary)
    interface_plan = assembly_record
    if interface_plan.get("schema") != "contact-interface-plan/v1":
        raise ValueError("04 assembly artifact is not a contact-interface-plan/v1 plan")
    if interface_plan.get("recognized_boundary_fingerprint") != boundaries.fingerprint:
        raise ValueError("04 plan and 03 frozen-boundary fingerprints do not match")
    project_settings = loaded_summary.get("project_settings", {})
    face_color_tokens = loaded_arrays.get("face_color_tokens", np.asarray([], dtype="U"))
    color_info, color_order = build_color_info_map(project_settings, face_color_tokens, None)
    COLOR_CATALOG.replace(color_info, color_order)

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
    output_components, source_face_ownership = complete_component_face_ownership(
        vertices, faces, components,
        protected_part_ids=confirmed_part_ids(interface_plan),
    )
    completed_parts, part_records = build_pairwise_part_meshes(
        interface_plan, arrays, summary,
        vertices=vertices, faces=faces, components=output_components,
        recognized_components=components,
        face_color_tokens=face_color_tokens,
        remove_detached_micro_shells_part_ids=remove_detached_micro_shells_part_ids,
    )
    store.write_arrays(
        "05_interface_assembly_meshes",
        **{
            f"{part['part_id'].lower()}_{field}": np.asarray(
                getattr(part["mesh"], field),
                dtype=np.float64 if field == "vertices" else np.int64,
            )
            for part in completed_parts
            for field in ("vertices", "faces")
        },
    )
    output_3mf = store.run_dir / "05_complete_parts.3mf"
    temporary_output = output_3mf.with_name(output_3mf.name + ".tmp")
    source_filament_colors = project_settings.get("filament_colour") or []
    if not isinstance(source_filament_colors, list):
        source_filament_colors = []
    writer = ThreeMFWriter()
    validator = ValidationService()
    try:
        export_summary = writer.write(
            temporary_output,
            completed_parts,
            title="Stage 05 complete mortise and tenon parts",
            source_application=project_settings.get("_source_application"),
            source_filament_colors=source_filament_colors,
            source_project_settings=project_settings,
            output_layout="assembly",
        )
        package_validation = validator.validate_package(
            temporary_output,
            completed_parts,
            source_filament_colors=source_filament_colors,
            source_application=project_settings.get("_source_application"),
            source_project_settings=project_settings,
            output_layout="assembly",
        )
        if not package_validation.get("valid"):
            raise ValueError("Stage 05 3MF validation failed: " + json.dumps(
                package_validation.get("errors", []), ensure_ascii=False
            ))
        reloaded_part_audits = validate_reloaded_interface_parts(
            temporary_output, completed_parts, part_records
        )
        temporary_output.replace(output_3mf)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    summary.update({
        "stage_status": "complete_parts_exported",
        "part_mesh_build_status": "complete_from_frozen_boundaries",
        "frozen_boundary_fingerprint": boundaries.fingerprint,
        "quality_gates_blocking": False,
        "resumed_from_stage_artifacts": str(source_run_dir),
        "output_3mf": str(output_3mf),
        "exported_part_count": len(completed_parts),
        "parts": part_records,
        "source_face_ownership": source_face_ownership,
        "export": export_summary,
        "package_validation": package_validation,
        "reloaded_part_audits": reloaded_part_audits,
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
