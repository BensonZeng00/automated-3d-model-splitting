"""Rebuild Stage 04 from validated Stage 02/03 artifacts without rereading 3MF."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from .application.stage_artifact_reader import read_stage_arrays, read_stage_json, restore_boundaries
from .application.stage_artifacts import StageArtifactStore
from .assembly_review import (
    AssemblyReviewRequired,
    ensure_assembly_review,
    load_direction_policy_corrections,
)
from .common import Component
from .contact_interface_planner import plan_contact_interfaces, DEFAULT_AREA_PRIORITY_RATIO
from .recognition_metadata import require_recognition_semantics
from .stage_cache import sha256_file
from .stage_table_report import interface_table, recognition_table, write_markdown_table


def resume_assembly_stage(
    source_run_dir: Path,
    *,
    expected_source: Path | None = None,
    visual_semantics_path: Path | None = None,
    recognition_review_path: Path | None = None,
    prior_assembly_review_path: Path | None = None,
    replace_unconfirmed_plan: bool = False,
    area_priority_ratio: float = DEFAULT_AREA_PRIORITY_RATIO,
) -> Path:
    """Plan interfaces from frozen 02/03 artifacts and preserve approved labels."""
    run_dir = Path(source_run_dir).expanduser().resolve()
    existing_plan = run_dir / "04_assembly_plan.json"
    if existing_plan.exists():
        if not replace_unconfirmed_plan:
            raise FileExistsError(
                "Stage 04 already exists; pass --replace-unconfirmed-plan only to refresh a pending plan"
            )
        review_path = run_dir / "04_assembly_review.json"
        if not review_path.is_file():
            raise ValueError("cannot replace Stage 04 without its review record")
        previous_review = json.loads(review_path.read_text(encoding="utf-8"))
        if previous_review.get("user_confirmed") is not False:
            raise ValueError("only an unconfirmed Stage 04 plan may be replaced")
        if previous_review.get("correction_requests"):
            raise ValueError("resolve recorded Stage 04 correction requests before refreshing the plan")
        stale_path = run_dir / f"04_assembly_review_stale_{uuid.uuid4().hex[:8]}.json"
        shutil.copy2(review_path, stale_path)
        review_path.unlink()
    loaded_summary = read_stage_json(run_dir, "02_loaded_project_summary")
    source_hash = str(loaded_summary.get("source_sha256", ""))
    if not source_hash:
        raise ValueError("Stage 02 summary has no source SHA-256")
    if expected_source is not None:
        source = Path(expected_source).expanduser().resolve()
        if Path(str(loaded_summary.get("source", ""))).resolve() != source:
            raise ValueError("Stage 02 artifacts belong to a different source path")
        if sha256_file(source) != source_hash:
            raise ValueError("source 3MF changed since Stage 02; rerun recognition")

    recognition_status = read_stage_json(run_dir, "03_recognition_review_status")
    if recognition_status.get("status") != "user_confirmed":
        raise ValueError("confirmed Stage 03 recognition review is required")
    recognition_summary = read_stage_json(run_dir, "03_recognition_summary")
    region_review = recognition_summary.get("region_review", {})
    if region_review.get("status") not in {"not_required", "user_confirmed"}:
        raise ValueError("confirmed source-region classifications are required")
    if visual_semantics_path is not None:
        from .recognition_metadata import annotate_recognition_with_visual_semantics, load_visual_semantics

        semantics = load_visual_semantics(Path(visual_semantics_path).expanduser())
        recognition_summary["regions"] = annotate_recognition_with_visual_semantics(
            recognition_summary.get("regions", []), semantics.get("parts", {})
        )
        require_recognition_semantics(recognition_summary["regions"])
        (run_dir / "03_visual_semantics.json").write_text(
            json.dumps({"source": str(Path(visual_semantics_path).resolve()), "parts": semantics.get("parts", {})}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        require_recognition_semantics(recognition_summary.get("regions", []))

    if recognition_review_path is not None:
        review = json.loads(Path(recognition_review_path).expanduser().read_text(encoding="utf-8"))
        if review.get("user_confirmed") is not True:
            raise ValueError("recognition review JSON is not confirmed")
        if Path(str(review.get("source", ""))).resolve() != Path(str(loaded_summary.get("source", ""))).resolve():
            raise ValueError("recognition review belongs to a different source 3MF")
        if int(review.get("source_face_count", -1)) != int(loaded_summary.get("face_count", -2)):
            raise ValueError("recognition review source face count does not match Stage 02")
        if review.get("expected_result_fingerprint") != recognition_status.get("result_fingerprint"):
            raise ValueError("recognition review does not match the saved Stage 03 result")

    loaded_arrays = read_stage_arrays(run_dir, "02_loaded_project")
    region_arrays = read_stage_arrays(run_dir, "03_recognition_regions")
    boundary_arrays = read_stage_arrays(run_dir, "03_recognized_boundaries")
    vertices = loaded_arrays.get("vertices")
    faces = loaded_arrays.get("faces")
    if vertices is None or faces is None:
        raise ValueError("02 loaded-project artifact must contain vertices and faces")

    regions = sorted(recognition_summary.get("regions", []), key=lambda item: int(item["part_index"]))
    components = []
    for record in regions:
        index = int(record["part_index"])
        if index != len(components) + 1:
            raise ValueError("03 region IDs must be contiguous and one-based")
        key = f"region_{index:04d}_source_face_ids"
        if key not in region_arrays:
            raise ValueError(f"03 region artifact is missing {key}")
        source_faces = np.asarray(region_arrays[key], dtype=np.int64)
        components.append(Component(
            color_code=str(record.get("color_code", "DEFAULT")),
            global_faces=source_faces,
            face_count=int(record.get("faces", len(source_faces))),
            area=float(record.get("area_mm2", 0.0)),
            bbox_min=np.asarray(record["bbox_min"], dtype=np.float64),
            bbox_max=np.asarray(record["bbox_max"], dtype=np.float64),
            center=np.asarray(record["center"], dtype=np.float64),
        ))
    boundaries = restore_boundaries(boundary_arrays, recognition_summary)
    model_center = vertices.mean(axis=0)
    plan = plan_contact_interfaces(
        area_priority_ratio=area_priority_ratio,
        vertices=vertices,
        faces=faces,
        components=components,
        recognized_boundaries=boundaries,
        model_center=model_center,
        recognition_records=recognition_summary["regions"],
        direction_policy_overrides=load_direction_policy_corrections(prior_assembly_review_path),
    )
    if not recognition_status.get("result_fingerprint"):
        raise ValueError("Stage 03 confirmation is missing its result fingerprint")
    plan["interface_table"] = interface_table(plan)

    store = StageArtifactStore(run_dir.parent, run_id=run_dir.name)
    table_path = store.run_dir / "04_interface_table.md"
    review_path = store.run_dir / "04_assembly_review.json"
    summary_record = json.loads((run_dir / "03_recognition_summary.json").read_text(encoding="utf-8"))
    store.write_json("03_recognition_summary", recognition_summary, inputs=summary_record.get("inputs"))
    recognition_summary["region_table"] = recognition_table(recognition_summary)
    store.write_json("03_recognition_table", recognition_summary["region_table"])
    write_markdown_table(
        run_dir / "03_recognition_table.md", "03 识别零件表", recognition_summary["region_table"]
    )
    if visual_semantics_path is not None:
        semantics_path = Path(visual_semantics_path).expanduser().resolve()
        store.write_json("03_visual_semantics", {
            "source": str(semantics_path),
            "sha256": sha256_file(semantics_path),
        })
    store.write_json("04_assembly_plan", plan)
    store.write_json("04_interface_table", plan["interface_table"])
    write_markdown_table(table_path, "04 榫卯接口表", plan["interface_table"])
    store.write_json("04_artifact_reuse", {
        "schema": "stage-04-artifact-reuse/v1",
        "source_sha256": source_hash,
        "recognized_boundary_fingerprint": boundaries.fingerprint,
        "source_run_dir": str(run_dir),
    })
    try:
        ensure_assembly_review(
            plan,
            source_sha256=source_hash,
            template_path=review_path,
            table_path=table_path,
            review_path=None,
        )
    except AssemblyReviewRequired:
        store.write_json("04_assembly_review_status", {
            "status": "needs_user_confirmation",
            "review_json": str(review_path),
            "table": str(table_path),
        })
    return store.run_dir
