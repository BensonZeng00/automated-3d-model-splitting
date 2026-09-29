from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .common import COLOR_CATALOG, COLOR_INFO, progress
from .project import ThreeMFReader, build_color_info_map, material_connectivity_labels
from .recognition import (
    classify_review_groups_semantically,
    connected_components_by_color,
    exterior_visible_face_mask,
    filter_groups_below_area,
    material_identity,
    recognition_colors_from_exterior,
    summarize_components,
)
from .recognition import _merge_partitioned_groups_into_components
from .package_io import ThreeMFWriter, default_output_3mf, prepare_output_3mf
from .validation import ValidationService
from .reporting import runtime_log
from .source_region_review import (
    build_source_region_review,
    default_region_review_dir,
    load_confirmed_region_decisions,
)
from .recognition_metadata import (
    annotate_recognition_with_visual_semantics,
    component_recognition_records,
    confidence_score,
    load_visual_semantics,
    print_recognition,
    require_recognition_semantics,
)
from .domain import SplitConfig
from .boundary_review import BoundaryReviewService
from .stage_cache import sha256_file
from .semantic_partition import apply_semantic_partitions
from .application import BoundarySnapshotBuilder, StageArtifactStore
from .application.boundary_review_artifacts import write_boundary_review_artifacts
from .application.recognition_review import (
    apply_recognition_actions,
    load_recognition_review,
    result_fingerprint as recognition_result_fingerprint,
    write_recognition_review_template,
)
from .stage_table_report import (
    interface_table, markdown_table, recognition_table, write_markdown_table,
)
from .assembly_review import (
    AssemblyReviewRequired,
    ensure_assembly_review,
    load_direction_policy_corrections,
)

class SplitPipeline:
    """Orchestrate one deterministic recognition, planning, build, and export run."""

    def __init__(self, config: SplitConfig, parser) -> None:
        self.config = config
        self.parser = parser
        self.reader = ThreeMFReader()
        self.writer = ThreeMFWriter()
        self.validator = ValidationService()
        self.boundary_snapshot_builder = BoundarySnapshotBuilder()

    def run(self) -> None:
        args = self.config.namespace
        input_path = self.config.input_path
        parser = self.parser
        output_path = default_output_3mf(input_path, args.output)
        artifact_root = (
            Path(getattr(args, "stage_artifacts_dir", None)).expanduser()
            if getattr(args, "stage_artifacts_dir", None)
            else output_path.with_name(output_path.stem + "_stages")
        )
        stage_artifacts = StageArtifactStore(
            artifact_root, run_id=getattr(args, "stage_artifacts_run_id", None)
        )
        runtime_log(
            "输入",
            "source_read_start",
            "开始读取并标准化源 3MF",
            input=str(input_path),
            model_entry=args.model_entry,
            format_profile=args.format_profile,
        )
        try:
            prepare_output_3mf(output_path, args.overwrite)
            vertices, faces, colors, project_settings = self.reader.read(
                input_path,
                model_entry=args.model_entry,
                format_profile=args.format_profile,
            )
        except (OSError, ValueError, zipfile.BadZipFile, ET.ParseError) as exc:
            parser.error(str(exc))
        runtime_log(
            "输入",
            "source_read_done",
            "源 3MF 已读取并转换到毫米空间",
            vertices=int(len(vertices)),
            faces=int(len(faces)),
            source_unit=project_settings.get("_source_unit", "millimeter"),
            selected_model_entry=project_settings.get("_selected_model_entry"),
        )
        stage_artifacts.write_arrays(
            "02_loaded_project",
            vertices=np.asarray(vertices),
            faces=np.asarray(faces),
            face_color_tokens=np.asarray(colors, dtype="U"),
        )
        source_sha256 = sha256_file(input_path) if input_path.is_file() else None
        stage_artifacts.write_json(
            "02_loaded_project_summary",
            {"source": str(input_path.resolve()),
             "source_sha256": source_sha256,
             "vertex_count": len(vertices), "face_count": len(faces),
             "project_settings": project_settings},
        )
        if getattr(args, "stop_after_stage", None) == "load":
            return
        new_color_info, new_color_order = build_color_info_map(
            project_settings,
            colors,
            Path(args.color_map_json).expanduser() if args.color_map_json else None,
        )
        COLOR_CATALOG.replace(new_color_info, new_color_order)
        boundary_review = BoundaryReviewService(
            output_path.with_name(output_path.stem + "_boundary_review"),
            getattr(args, "boundary_review_json", None),
        )
        source_owners, _ = material_connectivity_labels(colors)
        boundary_display_colors = {
            owner: COLOR_INFO.get(color, {}).get("hex", "#aaaaaa")
            for owner, color in zip(source_owners, colors)
        }
        clarified_owners, clarity_record = boundary_review.prepare(
            vertices, faces, source_owners, context="input",
            display_colors=boundary_display_colors,
        )
        ownership_changed = np.flatnonzero(np.asarray(source_owners) != clarified_owners)
        if getattr(args, "boundary_check_only", False):
            print("boundary_check=" + json.dumps(clarity_record, ensure_ascii=False), flush=True)
            return
        source_filament_colors = project_settings.get("filament_colour") or []
        if not isinstance(source_filament_colors, list):
            source_filament_colors = []
        progress(
            "颜色",
            "已按厂商 paint_color 槽位编码解析材料颜色",
            mapping={code: {"hex": info.get("hex"), "slot": info.get("filament_slot"), "source": info.get("mapping_source")} for code, info in COLOR_INFO.items()},
        )
        runtime_log(
            "识别",
            "exterior_visibility_start",
            "开始外表面多视角深度识别",
            views=int(args.exterior_view_count),
            resolution=int(args.exterior_depth_map_resolution),
        )
        progress(
            "识别",
            "正在从模型外部多方向采样可见表面颜色",
            views=args.exterior_view_count,
            resolution=args.exterior_depth_map_resolution,
            depth_tolerance_mm=args.exterior_depth_tolerance_mm,
        )
        visible_faces, exterior_visibility = exterior_visible_face_mask(
            vertices,
            faces,
            view_count=args.exterior_view_count,
            depth_map_resolution=args.exterior_depth_map_resolution,
            depth_tolerance_mm=args.exterior_depth_tolerance_mm,
        )
        recognition_token_colors, exterior_color_filter = recognition_colors_from_exterior(
            colors,
            visible_faces,
            faces=faces,
        )
        runtime_log(
            "识别",
            "exterior_visibility_done",
            "外表面可见性识别完成",
            visible_faces=int(exterior_visibility["visible_faces"]),
            occluded_faces=int(exterior_visibility["occluded_faces"]),
            profile=str(exterior_visibility["profile"]),
        )
        exterior_visibility["color_filter"] = exterior_color_filter
        print(
            "exterior_surface_recognition="
            + json.dumps(exterior_visibility, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        recognition_colors, material_connectivity = material_connectivity_labels(recognition_token_colors)
        if len(ownership_changed):
            recognition_colors = np.asarray(recognition_colors, dtype=object)
            recognition_colors[ownership_changed] = clarified_owners[ownership_changed]
            recognition_colors = recognition_colors.tolist()
        if material_connectivity["merged_material_groups"]:
            progress(
                "颜色",
                "已按源文件解析后的实际耗材槽合并等价 paint_color token 的连通边界",
                merged_groups=material_connectivity["merged_material_groups"],
            )
        runtime_log(
            "识别",
            "component_connectivity_start",
            "开始按材料和共享边识别连通部件",
            source_faces=int(len(faces)),
            small_region_review_max_faces=int(args.small_region_review_max_faces),
            max_region_review_candidates=int(args.max_region_review_candidates),
        )
        groups = connected_components_by_color(faces, recognition_colors)
        raw_group_count = len(groups)
        groups, automatic_noise_exclusions = filter_groups_below_area(
            vertices,
            faces,
            recognition_token_colors,
            groups,
            minimum_area_mm2=1.0,
        )
        if automatic_noise_exclusions:
            runtime_log(
                "识别",
                "subthreshold_regions_filtered",
                "已将面积小于 1 mm² 的连通区域过滤为噪声",
                minimum_area_mm2=1.0,
                filtered_regions=len(automatic_noise_exclusions),
                filtered_faces=sum(item["faces"] for item in automatic_noise_exclusions),
                filtered_area_mm2=sum(item["area_mm2"] for item in automatic_noise_exclusions),
            )
        if len(groups) > 1000:
            # Dense triangle-selector paint commonly leaves thousands of
            # microscopic same-material islands.  Reviewing each island is
            # quadratic in practice and cannot produce thousands of printable
            # parts.  Retain every substantial island plus the largest island
            # of each material, then attach only <= noise-threshold fragments
            # to a same-material component by shared edge or nearest bbox.
            largest_by_material = {}
            for group in groups:
                identity = material_identity(str(recognition_token_colors[int(group[0])]))
                if identity not in largest_by_material or len(group) > len(largest_by_material[identity]):
                    largest_by_material[identity] = group
            consolidation_limit = int(args.small_region_review_max_faces)
            anchors = [group for group in groups
                       if len(group) > consolidation_limit
                       or any(group is anchor for anchor in largest_by_material.values())]
            anchor_ids = {id(group) for group in anchors}
            fragments = [group for group in groups if id(group) not in anchor_ids]
            merged, ignored_fragments, merged_records = _merge_partitioned_groups_into_components(
                vertices, faces, recognition_colors, anchors, fragments,
                display_colors=recognition_token_colors)
            groups = [component.global_faces for component in merged]
            runtime_log('识别', 'dense_paint_fragments_consolidated',
                        '密集涂色微小岛已按同材料邻接或距离合并',
                        raw_groups=len(anchors) + len(fragments),
                        effective_groups=len(groups), fragments=len(fragments),
                        merged_fragments=len(merged_records),
                        ignored_fragments=len(ignored_fragments),
                        maximum_fragment_faces=consolidation_limit)
        region_review = None
        normalized_groups = [np.asarray(group, dtype=np.int64) for group in groups]
        from .region_review import (
            limit_region_review_candidates,
            select_region_review_groups,
        )
        _, review_candidates = select_region_review_groups(
            vertices, faces, normalized_groups, args.noise_review_max_faces,
            args.small_region_review_max_faces, visible_faces)
        review_candidates, review_limit_noise_exclusions = limit_region_review_candidates(
            vertices, faces, review_candidates,
            args.max_region_review_candidates,
        )
        review_groups = [group for group, _ in review_candidates]
        review_records = classify_review_groups_semantically(
            vertices=vertices, faces=faces, connectivity_colors=recognition_colors,
            display_colors=recognition_token_colors, all_groups=normalized_groups,
            review_groups=review_groups, min_faces=args.small_region_review_max_faces + 1,
            visible_faces=visible_faces, view_count=args.exterior_view_count,
            depth_map_resolution=args.exterior_depth_map_resolution)
        if review_groups and not args.region_review_json:
            review_dir = (Path(args.region_review_dir).expanduser()
                          if args.region_review_dir else default_region_review_dir(input_path))
            region_review = build_source_region_review(
                input_path=input_path, review_dir=review_dir, vertices=vertices,
                faces=faces, connectivity_colors=recognition_colors,
                display_colors=recognition_token_colors, groups=normalized_groups,
                noise_max_faces=args.noise_review_max_faces,
                small_region_max_faces=args.small_region_review_max_faces,
                visible_faces=visible_faces, view_count=args.exterior_view_count,
                depth_map_resolution=args.exterior_depth_map_resolution,
                image_resolution=args.region_review_resolution,
                max_review_candidates=args.max_region_review_candidates,
                selected_candidates=review_candidates,
                automatic_noise_candidates=review_limit_noise_exclusions,
                classified_records=review_records)
            print("region_review_required=" + json.dumps(
                region_review, ensure_ascii=False, sort_keys=True), flush=True)
            progress("识别", "检测到噪声、小区域或长细条候选；请完成语义分类",
                     candidates=len(review_groups), manifest=region_review["manifest_path"],
                     decisions=region_review["decision_path"],
                     automatically_classified_noise=len(review_limit_noise_exclusions))
            raise SystemExit(4)
        review_decisions = {}
        if review_groups:
            review_json_path = Path(args.region_review_json).expanduser()
            try:
                review_decisions = load_confirmed_region_decisions(
                    review_json_path, input_path=input_path, source_face_count=len(faces),
                    noise_max_faces=args.noise_review_max_faces,
                    small_region_max_faces=args.small_region_review_max_faces,
                    max_review_candidates=args.max_region_review_candidates,
                    expected_records=review_records)
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
            region_review = {
                "status": "user_confirmed", "source": str(review_json_path),
                "candidate_count": len(review_groups),
                "total_candidate_count": len(review_groups) + len(review_limit_noise_exclusions),
                "automatic_noise_count": len(review_limit_noise_exclusions),
                "automatic_noise_faces": sum(int(item["faces"]) for item in review_limit_noise_exclusions),
                "automatic_noise_area_mm2": sum(float(item["area_mm2"]) for item in review_limit_noise_exclusions),
                "classifications": {
                    value: sum(item["classification"] == value for item in review_decisions.values())
                    + (len(review_limit_noise_exclusions) if value == "noise" else 0)
                    for value in ("noise", "part", "uncertain")
                },
                "automatic_noise_exclusions": review_limit_noise_exclusions,
            }
        else:
            region_review = {
                "status": "not_required", "candidate_count": 0,
                "total_candidate_count": len(review_limit_noise_exclusions),
                "automatic_noise_count": len(review_limit_noise_exclusions),
                "automatic_noise_faces": sum(int(item["faces"]) for item in review_limit_noise_exclusions),
                "automatic_noise_area_mm2": sum(float(item["area_mm2"]) for item in review_limit_noise_exclusions),
                "automatic_noise_exclusions": review_limit_noise_exclusions,
            }
        confirmed_part_source_ids = {
            int(record["source_min_face_index"])
            for record in review_decisions.values()
            if record["classification"] == "part"
        }
        confirmed_noise_source_ids = {
            int(record["source_min_face_index"])
            for record in review_decisions.values()
            if record["classification"] == "noise"
        }
        confirmed_noise_source_ids.update(
            int(item["source_min_face_index"])
            for item in review_limit_noise_exclusions
        )
        components, _ = summarize_components(
            vertices, faces, recognition_colors, groups, 101,
            display_colors=recognition_token_colors,
            retained_source_min_face_ids=confirmed_part_source_ids,
            excluded_source_min_face_ids=confirmed_noise_source_ids)
        visual_semantics = load_visual_semantics(
            Path(args.visual_semantics_json).expanduser()
            if args.visual_semantics_json else None
        )
        visual_semantic_min_confidence = confidence_score(
            args.visual_semantic_min_confidence, default=0.65
        )
        components, semantic_partition_records = apply_semantic_partitions(
            vertices,
            faces,
            components,
            visual_semantics.get("physical_partitions", []),
            visual_semantic_min_confidence,
        )
        if visual_semantics.get("physical_partitions"):
            runtime_log(
                "识别",
                "visual_semantic_physical_partitions_reviewed",
                "已审核同材料物理分割平面并应用通过安全门的拆分",
                applied=semantic_partition_records["applied"],
                rejected=semantic_partition_records["rejected"],
            )
        try:
            components, recognized_boundaries, recognition_exclusions = (
                self.boundary_snapshot_builder.build_with_component_filter(
                    vertices, faces, components
                )
            )
        except ValueError as exc:
            parser.error(str(exc))
        original_recognition_components = list(components)
        try:
            recognition_review = load_recognition_review(
                Path(args.recognition_review_json).expanduser()
                if args.recognition_review_json else None,
                source_path=input_path,
                source_face_count=len(faces),
                components=original_recognition_components,
            )
            components = apply_recognition_actions(
                vertices, faces, original_recognition_components,
                recognition_review["actions"],
            )
            if recognition_review["actions"]:
                components, recognized_boundaries, action_exclusions = (
                    self.boundary_snapshot_builder.build_with_component_filter(
                        vertices, faces, components
                    )
                )
                recognition_exclusions.extend(action_exclusions)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        source_region_classifications = list(review_decisions.values())
        if not components:
            raise SystemExit("No source components found")
        # Recognition ownership is source data.  Boundary planning may report
        # an ambiguous seam later, but it must not reassign source faces.
        # Ownership and semantic classification must never repaint source.
        runtime_log(
            "识别",
            "component_connectivity_done",
            "连通 source 区域识别与语义审核完成",
            raw_groups=int(raw_group_count),
            effective_components=int(len(components)),
            boundary_filtered_components=int(len(recognition_exclusions)),
            reviewed_source_regions=int(len(source_region_classifications)),
            source_geometry_changed=False,
        )
        print("region_review=" + json.dumps({
            "policy": "preserve-source",
            "review": region_review,
            "classifications": source_region_classifications,
            "automatic_noise_filter": {
                "minimum_area_mm2": 1.0,
                "excluded_region_count": len(automatic_noise_exclusions),
                "excluded_face_count": sum(item["faces"] for item in automatic_noise_exclusions),
                "excluded_area_mm2": sum(item["area_mm2"] for item in automatic_noise_exclusions),
            },
        }, ensure_ascii=False, sort_keys=True), flush=True)
        if review_limit_noise_exclusions:
            print(
                "region_review_limit_noise="
                + json.dumps({
                    "count": len(review_limit_noise_exclusions),
                    "faces": sum(item["faces"] for item in review_limit_noise_exclusions),
                    "area_mm2": sum(item["area_mm2"] for item in review_limit_noise_exclusions),
                    "details": "03_recognition_summary.json",
                }, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
        if automatic_noise_exclusions:
            print(
                "recognition_noise_exclusions="
                + json.dumps({
                    "count": len(automatic_noise_exclusions),
                    "faces": sum(item["faces"] for item in automatic_noise_exclusions),
                    "area_mm2": sum(item["area_mm2"] for item in automatic_noise_exclusions),
                    "details": "03_recognition_summary.json",
                }, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
        if recognition_exclusions:
            print(
                "recognition_excluded_regions="
                + json.dumps(recognition_exclusions, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
        model_center = vertices.mean(axis=0)
        recognition = component_recognition_records(vertices, faces, components, colors)
        recognition = annotate_recognition_with_visual_semantics(recognition, visual_semantics.get("parts", {}))
        confirmed_tiny_labels = {
            int(record["source_min_face_index"]): record
            for record in source_region_classifications
        }
        for component_index, component in enumerate(components, start=1):
            source_min_face_index = int(np.min(component.global_faces))
            confirmed_tiny = confirmed_tiny_labels.get(source_min_face_index)
            if confirmed_tiny is None:
                continue
            record = recognition[component_index - 1]
            record["region_review_label"] = str(confirmed_tiny["semantic_label"])
            record["source_region_classification"] = str(confirmed_tiny["classification"])
            record["region_review_confidence"] = str(
                confirmed_tiny.get("visual_confidence", "UNKNOWN")
            )
            record["region_review_user_confirmed"] = True
        processing_classifications = {
            index: {
                "part_index": index,
                "selected_processing_mode": "not_applicable_to_interface_planning",
                "suggested_processing_mode": "not_applicable_to_interface_planning",
                "processing_mode_status": "parent_child_processing_modes_removed",
                "processing_mode_confidence": 1.0,
                "processing_mode_evidence": {},
            }
            for index in range(1, len(components) + 1)
        }
        for record in recognition:
            record["recognition_basis"] = "exterior-visible"
            record["occluded_paint_excluded"] = True
            record.update(processing_classifications[int(record["part_index"])])
        # The recognition-stage snapshot already filtered boundaryless regions
        # before review and body selection; later stages consume this object.
        source_boundary_color_map = {
            str(token): str(COLOR_INFO.get(str(token), {}).get("hex", "#A8A8AC"))
            for token in set(str(value) for value in colors)
        }
        boundary_review_artifacts = write_boundary_review_artifacts(
            stage_artifacts.run_dir,
            recognized_boundaries,
            recognition,
            {"status": region_review.get("status", "unknown"),
             "candidate_count": region_review.get("candidate_count", 0),
             "classifications": source_region_classifications},
            components,
            stage_artifacts.run_dir / "03_recognition_review.json",
            source_vertices=vertices,
            source_faces=faces,
            source_face_color_tokens=colors,
            source_color_map=source_boundary_color_map,
        )
        current_result_fingerprint = recognition_result_fingerprint(
            components, recognized_boundaries
        )
        recognition_confirmed = (
            bool(recognition_review["user_confirmed"])
            and recognition_review["expected_result_fingerprint"] == current_result_fingerprint
        )
        review_template_path = stage_artifacts.run_dir / "03_recognition_review.json"
        write_recognition_review_template(
            review_template_path,
            review=recognition_review,
            result_fingerprint_value=current_result_fingerprint,
            components=components,
            reference_regions=[
                {
                    "part": f"P{index:02d}",
                    "source_min_face_index": int(np.min(component.global_faces)),
                    "face_count": int(component.face_count),
                    "color_code": str(component.color_code),
                }
                for index, component in enumerate(original_recognition_components, start=1)
            ],
            user_confirmed=recognition_confirmed,
        )
        stage_artifacts.write_json(
            "03_recognition_review_status",
            {
                "status": "user_confirmed" if recognition_confirmed else "needs_user_confirmation",
                "decision_file": str(review_template_path),
                "recognition_fingerprint": recognition_review["recognition_fingerprint"],
                "result_fingerprint": current_result_fingerprint,
                "actions": recognition_review["actions"],
            },
        )
        progress("识别", f"识别到 {len(recognition)} 个有效部件", region_review="preserve-source", reviewed=len(source_region_classifications), area_filtered_noise=len(automatic_noise_exclusions))
        stage_artifacts.write_arrays(
            "03_recognition_regions",
            **{f"region_{index:04d}_source_face_ids": np.asarray(component.global_faces, dtype=np.int64)
               for index, component in enumerate(components, start=1)},
        )
        stage_artifacts.write_arrays(
            "03_recognized_boundaries",
            **recognized_boundaries.flattened_arrays(),
        )
        recognition_summary = {
            "regions": recognition, "region_review": region_review,
             "source_region_classifications": source_region_classifications,
             "review_limit_noise_exclusions": review_limit_noise_exclusions,
             "automatic_noise_exclusions": automatic_noise_exclusions,
             "recognition_excluded_regions": recognition_exclusions,
             "source_geometry_changed": False,
             "recognized_boundaries": {
                 "fingerprint": recognized_boundaries.fingerprint,
                 "component_count": len(recognized_boundaries.component_loops),
                 "loop_count": sum(len(loops) for loops in recognized_boundaries.component_loops),
                 "filtered_noise_loop_count": sum(
                     record.get("status") == "filtered"
                     for record in recognized_boundaries.simplification_records
                 ),
                 "loop_vertex_counts": [
                     [len(loop) for loop in loops]
                     for loops in recognized_boundaries.component_loops
                 ],
                 "source_vertex_count": recognized_boundaries.source_vertex_count,
                 "source_face_count": recognized_boundaries.source_face_count,
                 "simplification_retained_fraction": 0.05,
                 "simplification_records": list(
                     recognized_boundaries.simplification_records
                 ),
                 "review_artifacts": boundary_review_artifacts,
                 "immutable_after_recognition": True,
             },
        }
        recognition_summary["region_table"] = recognition_table(recognition_summary)
        stage_artifacts.write_json("03_recognition_summary", recognition_summary)
        stage_artifacts.write_json("03_recognition_table", recognition_summary["region_table"])
        recognition_table_text = markdown_table(
            "03 识别零件表", recognition_summary["region_table"]
        )
        recognition_table_path = stage_artifacts.run_dir / "03_recognition_table.md"
        write_markdown_table(
            recognition_table_path, "03 识别零件表", recognition_summary["region_table"]
        )
        print(recognition_table_text, flush=True)
        print("stage03_table=" + str(recognition_table_path), flush=True)
        print_recognition(recognition)
        if not recognition_confirmed and not (
            args.recognize_only or getattr(args, "stop_after_stage", None) == "recognize"
        ):
            print("recognition_review_required=" + json.dumps({
                "status": "needs_user_confirmation",
                "review_json": str(review_template_path),
                "report": boundary_review_artifacts["boundary_review_report"],
                "actions_applied_for_review": recognition_review["actions"],
            }, ensure_ascii=False), flush=True)
            raise SystemExit(4)
        if args.recognize_only or getattr(args, "stop_after_stage", None) == "recognize":
            return
        try:
            require_recognition_semantics(recognition)
        except ValueError as exc:
            print("semantic_review_required=" + json.dumps({
                "status": "needs_image_semantics",
                "reason": str(exc),
                "stage03_run_dir": str(stage_artifacts.run_dir),
                "boundary_preview": str(stage_artifacts.run_dir / "03_recognition_boundaries.png"),
            }, ensure_ascii=False), flush=True)
            raise SystemExit(4) from exc
        from .contact_interface_planner import plan_contact_interfaces

        runtime_log(
            "装配规划",
            "contact_interface_plan_start",
            "根据识别冻结的简化边界规划接触零件间的榫卯关系",
            components=int(len(components)),
            boundary_fingerprint=recognized_boundaries.fingerprint,
        )
        contact_interface_plan = plan_contact_interfaces(
            area_priority_ratio=args.area_priority_ratio,
            vertices=vertices,
            faces=faces,
            components=components,
            recognized_boundaries=recognized_boundaries,
            model_center=model_center,
            recognition_records=recognition,
            direction_policy_overrides=load_direction_policy_corrections(
                getattr(args, "assembly_review_json", None)
            ),
        )
        contact_interface_plan["interface_table"] = interface_table(contact_interface_plan)
        stage_artifacts.write_json("04_assembly_plan", contact_interface_plan)
        stage_artifacts.write_json("04_interface_table", contact_interface_plan["interface_table"])
        interface_table_text = markdown_table(
            "04 榫卯接口表", contact_interface_plan["interface_table"]
        )
        interface_table_path = stage_artifacts.run_dir / "04_interface_table.md"
        write_markdown_table(
            interface_table_path, "04 榫卯接口表", contact_interface_plan["interface_table"]
        )
        print(interface_table_text, flush=True)
        print("stage04_table=" + str(interface_table_path), flush=True)
        from .stage_reuse_provenance import provenance_record

        stage_artifacts.write_json(
            "04_reuse_provenance", provenance_record(args, source_sha256)
        )
        runtime_log(
            "装配规划",
            "contact_interface_plan_done",
            "榫卯关系规划完成",
            interfaces=int(contact_interface_plan["interface_count"]),
        )
        try:
            assembly_review = ensure_assembly_review(
                contact_interface_plan,
                source_sha256=source_sha256,
                template_path=stage_artifacts.run_dir / "04_assembly_review.json",
                table_path=interface_table_path,
                review_path=getattr(args, "assembly_review_json", None),
            )
        except AssemblyReviewRequired:
            review_notice = {
                "status": "needs_user_confirmation",
                "review_json": str(stage_artifacts.run_dir / "04_assembly_review.json"),
                "table": str(interface_table_path),
            }
            stage_artifacts.write_json("04_assembly_review_status", review_notice)
            if getattr(args, "stop_after_stage", None) == "assembly":
                print("assembly_review_required=" + json.dumps(
                    review_notice, ensure_ascii=False
                ), flush=True)
                return
            raise
        stage_artifacts.write_json("04_assembly_review_status", {
            "status": "user_confirmed",
            "plan_fingerprint": assembly_review["plan_fingerprint"],
        })
        if getattr(args, "stop_after_stage", None) == "assembly":
            return
        from .interface_assembly import (
            build_pairwise_interface_surfaces,
            build_pairwise_part_meshes,
        )
        from .source_face_ownership import complete_component_face_ownership, confirmed_part_ids

        interface_arrays, interface_summary = build_pairwise_interface_surfaces(
            contact_interface_plan,
            vertices=vertices,
            faces=faces,
            components=components,
            scale_ratio=args.interface_scale_ratio,
            clearance_mm=args.interface_clearance_mm,
        )
        if interface_arrays:
            stage_artifacts.write_arrays(
                "05_interface_surfaces", **interface_arrays
            )
        output_components, source_face_ownership = complete_component_face_ownership(
            vertices, faces, components,
            protected_part_ids=confirmed_part_ids(contact_interface_plan),
        )
        built_parts, part_records = build_pairwise_part_meshes(
            contact_interface_plan, interface_arrays, interface_summary,
            vertices=vertices, faces=faces, components=output_components,
            recognized_components=components,
            face_color_tokens=colors,
            remove_detached_micro_shells_part_ids=frozenset(
                getattr(args, "remove_detached_micro_shells_part", [])
            ),
        )
        mesh_audits = [
            {"part_id": part["part_id"], **self.validator.validate_mesh(part["mesh"])}
            for part in built_parts
        ]
        invalid_meshes = [
            item for item in mesh_audits
            if not item.get("watertight") or not item.get("winding_consistent")
            or int(item.get("open_edges", 0)) or int(item.get("over_shared_edges", 0))
        ]
        if invalid_meshes:
            raise ValueError(
                "05 produced invalid part meshes: "
                + json.dumps(invalid_meshes, ensure_ascii=False, sort_keys=True)
            )
        part_arrays = {}
        for part in built_parts:
            array_key = re.sub(r"[^a-z0-9_]+", "_", part["part_id"].lower()).strip("_")
            part_arrays[f"{array_key}_vertices"] = np.asarray(part["mesh"].vertices, dtype=np.float64)
            part_arrays[f"{array_key}_faces"] = np.asarray(part["mesh"].faces, dtype=np.int64)
        if part_arrays:
            stage_artifacts.write_arrays("05_interface_assembly_meshes", **part_arrays)
        interface_summary.update({
            "stage_status": "complete_parts_built",
            "part_mesh_build_status": "complete_from_frozen_boundaries",
            "exported_part_count": len(built_parts),
            "parts": part_records,
            "frozen_boundary_fingerprint": recognized_boundaries.fingerprint,
            "source_face_ownership": source_face_ownership,
        })
        stage_artifacts.write_json(
            "05_interface_assembly_summary",
            interface_summary,
            inputs={
                "interface_plan_schema": contact_interface_plan["schema"],
                "recognized_boundary_fingerprint": recognized_boundaries.fingerprint,
                "scale_ratio": args.interface_scale_ratio,
                "clearance_mm": args.interface_clearance_mm,
            },
        )
        if getattr(args, "stop_after_stage", None) == "interface-assembly":
            return
        temporary_output = output_path.with_name(output_path.name + ".tmp")
        try:
            export_summary = self.writer.write(
                temporary_output,
                built_parts,
                title=f"{input_path.stem} complete mortise and tenon parts",
                source_application=project_settings.get("_source_application"),
                source_filament_colors=source_filament_colors,
                source_project_settings=project_settings,
                output_layout=getattr(args, "output_layout", None) or "assembly",
            )
            package_validation = self.validator.validate_package(
                temporary_output,
                built_parts,
                source_filament_colors=source_filament_colors,
                source_application=project_settings.get("_source_application"),
                source_project_settings=project_settings,
                output_layout=getattr(args, "output_layout", None) or "assembly",
            )
            if not package_validation.get("valid"):
                raise ValueError(
                    "08 final 3MF validation failed: "
                    + json.dumps(package_validation.get("errors", []), ensure_ascii=False)
                )
            from .interface_package_validation import validate_reloaded_interface_parts
            reloaded_part_audits = validate_reloaded_interface_parts(
                temporary_output, built_parts, part_records
            )
            temporary_output.replace(output_path)
        except Exception:
            temporary_output.unlink(missing_ok=True)
            raise
        stage_artifacts.write_json(
            "08_validated_publish",
            {
                "output_3mf": str(output_path.resolve()),
                "part_count": len(built_parts),
                "interface_count": int(contact_interface_plan["interface_count"]),
                "mesh_validation": mesh_audits,
                "package_validation": package_validation,
                "reloaded_part_audits": reloaded_part_audits,
                "export": export_summary,
                "recognized_boundary_fingerprint": recognized_boundaries.fingerprint,
            },
        )
        print("split_summary=" + json.dumps({
            "output_3mf": str(output_path),
            "part_count": len(built_parts),
            "output_kind": "complete_parts",
            "interface_count": int(contact_interface_plan["interface_count"]),
            "mesh_validation": mesh_audits,
            "package_validation": package_validation,
        }, ensure_ascii=False), flush=True)
        return
