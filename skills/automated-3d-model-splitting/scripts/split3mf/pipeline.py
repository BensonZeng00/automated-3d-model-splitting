from __future__ import annotations

import hashlib
from dataclasses import replace

from .common import *
from .project import *
from .recognition import *
from .recognition import _merge_partitioned_groups_into_components
from .mesh import *
from .package_io import *
from .selection import *
from .assembly import *
from .validation import *
from .part_geometry import *
from .debug_export import *
from .reporting import *
from .source_region_review import *
from .domain import PlanarArcRetopologyConfig, PlanarArcRetopologyContext, SplitConfig
from .recursive_preflight import FullTreePreflightError, FullTreePreflightService
from .explicit_merge import merge_body_components, parse_part_group
from .interface_retreat import apply_visual_interface_retreats
from .uniform_fit import scale_finished_insert
from .guided_internal_cut import GuidedInternalCutSpec
from .boundary_review import BoundaryReviewService, owners_from_components
from .stage_cache import (
    RecursiveStageCache,
    fingerprint_payload,
    implementation_fingerprint,
    normalized_run_arguments,
    sha256_file,
)
from .semantic_partition import apply_semantic_partitions
from .application import BoundarySnapshotBuilder, StageArtifactStore


def uses_layer_child_cut_references(assembly_mode: str, tree_strategy: str) -> bool:
    """Whether recursive layer planning is the sole cut-reference consumer."""
    return (
        str(assembly_mode) in {"tree", "flat"}
        and str(tree_strategy) == "recursive-minimal"
    )


class SplitPipeline:
    """Orchestrate one deterministic recognition, planning, build, and export run."""

    def __init__(self, config: SplitConfig, parser) -> None:
        self.config = config
        self.parser = parser
        self.reader = ThreeMFReader()
        self.recognizer = PartRecognizer()
        self.body_selector = BodySelector()
        self.assembly_planner = AssemblyPlanner()
        self.direction_planner = InwardDirectionPlanner()
        self.cap_planner = AdaptiveCapPlanner()
        self.triangulator = BoundaryTriangulator()
        self.mesh_builder = PartMeshBuilder()
        self.writer = ThreeMFWriter()
        self.validator = ValidationService()
        self.full_tree_preflight = FullTreePreflightService()
        self.boundary_snapshot_builder = BoundarySnapshotBuilder()

    def run(self) -> None:
        args = self.config.namespace
        input_path = self.config.input_path
        parser = self.parser
        requested_max_extension_mm = float(args.max_extension_mm)
        args.max_extension_mm = max(
            requested_max_extension_mm,
            0.4,
        )
        maximum_planar_travel_mm = min(
            max(float(args.max_planar_travel_mm), args.max_extension_mm),
            MAXIMUM_SAFE_INWARD_DEPTH_MM,
        )
        available_planar_extra_mm = max(0.0, maximum_planar_travel_mm - args.max_extension_mm)
        requested_planar_extra_limit_mm = (
            None if args.planar_extra_limit_mm is None else float(args.planar_extra_limit_mm)
        )
        args.planar_extra_limit_mm = (
            available_planar_extra_mm
            if requested_planar_extra_limit_mm is None
            else min(requested_planar_extra_limit_mm, available_planar_extra_mm)
        )
        planar_travel_policy = {
            "preferred_minimum_inward_depth_mm": float(args.max_extension_mm),
            "global_safety_ceiling_mm": maximum_planar_travel_mm,
            "parent_thickness_clearance_mm": PARENT_THICKNESS_CLEARANCE_MM,
            "parent_thickness_rule": (
                "min(global ceiling, active parent bounding diagonal, "
                "sampled opposing-surface distance - 0.05 mm)"
            ),
            "requested_planar_extra_limit_mm": requested_planar_extra_limit_mm,
            "effective_planar_extra_limit_mm": float(args.planar_extra_limit_mm),
        }
        insert_shrink_mm, socket_overcut_mm = clearance_offsets(args.clearance_mode, args.fit_clearance_mm)
        top_edge_clearance_mm = visible_top_edge_clearance(insert_shrink_mm)
        force_flat_part_indices = parse_part_index_tokens(args.force_flat_parts)
        try:
            part_mode_overrides = parse_part_mode_overrides(args.part_mode_overrides)
        except ValueError as exc:
            parser.error(str(exc))

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
            debug_root = None
            if args.debug_recursive_3mf:
                debug_root = prepare_debug_directory(
                    output_path.with_name(output_path.stem + "_debug"), args.overwrite)
                runtime_log("运行", "debug_directory_reserved", "已分配独立调试目录", path=str(debug_root))
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
        stage_artifacts.write_json(
            "02_loaded_project_summary",
            {"source": str(input_path.resolve()),
             "source_sha256": sha256_file(input_path) if input_path.is_file() else None,
             "vertex_count": len(vertices), "face_count": len(faces),
             "project_settings": project_settings},
        )
        if getattr(args, "stop_after_stage", None) == "load":
            return
        retopology_failure_sink = None
        if bool(args.diagnostic_preview):
            failure_output_dir = output_path.parent / (
                output_path.stem + "_diagnostics"
            )

            def retopology_failure_sink(payload: dict) -> None:
                export_retopology_failure_diagnostics(
                    payload,
                    failure_output_dir,
                )

        interface_retopology = PlanarArcRetopologyContext(
            config=PlanarArcRetopologyConfig.from_namespace(args),
            failure_sink=retopology_failure_sink,
        )
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
        interface_retopology = replace(interface_retopology,
            curve_review_sink=boundary_review.review_curve)
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
        if args.recognition_surface_profile == "exterior-visible":
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
                body_color_override=args.body_color,
                faces=faces,
            )
        else:
            visible_faces = np.ones(len(faces), dtype=bool)
            recognition_token_colors = list(colors)
            exterior_visibility = {
                "profile": "all-faces",
                "method": "legacy_all_faces",
                "view_count": 0,
                "depth_map_resolution": 0,
                "depth_tolerance_mm": 0.0,
                "visible_faces": int(len(faces)),
                "occluded_faces": 0,
                "visible_ratio": 1.0,
                "per_view_visible_faces": [],
            }
            exterior_color_filter = {
                "base_color_token": args.body_color,
                "base_color_source": "not_applied",
                "reassigned_occluded_faces": 0,
                "already_base_occluded_faces": 0,
                "protected_enclosed_occluded_faces": 0,
                "protected_enclosed_by_source_token": [],
                "reassigned_by_source_token": [],
            }
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
        )
        groups = connected_components_by_color(faces, recognition_colors)
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
        from .region_review import select_region_review_groups
        _, review_candidates = select_region_review_groups(
            vertices, faces, normalized_groups, args.noise_review_max_faces,
            args.small_region_review_max_faces, visible_faces)
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
                image_resolution=args.region_review_resolution)
            print("region_review_required=" + json.dumps(
                region_review, ensure_ascii=False, sort_keys=True), flush=True)
            progress("识别", "检测到噪声、小区域或长细条候选；请完成语义分类",
                     candidates=len(review_groups), manifest=region_review["manifest_path"],
                     decisions=region_review["decision_path"])
            raise SystemExit(4)
        review_decisions = {}
        if review_groups:
            review_json_path = Path(args.region_review_json).expanduser()
            try:
                review_decisions = load_confirmed_region_decisions(
                    review_json_path, input_path=input_path, source_face_count=len(faces),
                    noise_max_faces=args.noise_review_max_faces,
                    small_region_max_faces=args.small_region_review_max_faces,
                    expected_records=review_records)
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
            region_review = {
                "status": "user_confirmed", "source": str(review_json_path),
                "candidate_count": len(review_groups),
                "classifications": {
                    value: sum(item["classification"] == value for item in review_decisions.values())
                    for value in ("noise", "part", "uncertain")
                },
            }
        else:
            region_review = {"status": "not_required", "candidate_count": 0}
        components, _ = summarize_components(
            vertices, faces, recognition_colors, groups, 101,
            display_colors=recognition_token_colors)
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
        source_region_classifications = list(review_decisions.values())
        if not components:
            raise SystemExit("No source components found")
        component_owners = owners_from_components(len(faces), components)
        # Recognition ownership is source data.  Boundary planning may report
        # an ambiguous seam later, but it must not reassign source faces.
        # Ownership and semantic classification must never repaint source.
        recursive_face_colors = list(colors)
        explicit_body_merge = None
        explicit_body_index = None
        if args.merge_body_parts:
            try:
                requested_merge_indices = parse_part_group(args.merge_body_parts)
                merge_result = merge_body_components(
                    vertices,
                    faces,
                    components,
                    requested_merge_indices,
                )
            except ValueError as exc:
                parser.error(str(exc))
            components = merge_result.components
            explicit_body_index = int(merge_result.body_index)
            explicit_body_merge = merge_result.record
            runtime_log(
                "主体",
                "explicit_body_merge_done",
                "已将指定识别部件合并为保留逐面的多材料主体",
                requested_parts=explicit_body_merge[
                    "requested_original_part_indices"
                ],
                effective_body_index=explicit_body_index,
                effective_components=int(len(components)),
                per_face_materials_preserved=True,
            )
        runtime_log(
            "识别",
            "component_connectivity_done",
            "连通 source 区域识别与语义审核完成",
            raw_groups=int(len(groups)),
            effective_components=int(len(components)),
            reviewed_source_regions=int(len(source_region_classifications)),
            source_geometry_changed=False,
        )
        # This run-scoped immutable snapshot is the only source-mesh boundary
        # definition consumed by downstream planning stages.
        recognized_boundaries = self.boundary_snapshot_builder.build(
            vertices, faces, components
        )
        print("region_review=" + json.dumps({
            "policy": "preserve-source",
            "review": region_review,
            "classifications": source_region_classifications,
        }, ensure_ascii=False, sort_keys=True), flush=True)
        model_center = vertices.mean(axis=0)
        runtime_log(
            "主体",
            "body_evidence_start",
            "开始测量结构分隔证据并选择根主体",
            components=int(len(components)),
        )
        body_separator_evidence = body_selection_separator_evidence(
            vertices=vertices,
            faces=faces,
            components=components,
            model_center=model_center,
            min_faces=(args.small_region_review_max_faces + 1),
        )
        excluded_auto_body_indices = {
            int(index)
            for index, record in body_separator_evidence.items()
            if record.get("exclude_from_automatic_body")
        }
        body_component = choose_body_component(
            components,
            args.body_strategy,
            args.body_color,
            explicit_body_index if explicit_body_index is not None else args.body_index,
            excluded_auto_indices=excluded_auto_body_indices,
            auto_selection_evidence=body_separator_evidence,
        )
        body_index = component_identity_index(components, body_component)
        runtime_log(
            "主体",
            "body_evidence_done",
            "根主体选择完成",
            body_index=body_index,
            excluded_separator_candidates=sorted(excluded_auto_body_indices),
        )
        progress(
            "主体",
            "已在排除强结构分隔候选后选择根主体",
            body=None if body_index is None else f"P{body_index:02d}",
            body_candidate_score=(
                None
                if body_index is None
                else round(float(body_separator_evidence[body_index]["body_candidate_score"]), 6)
            ),
            excluded_separator_candidates=[f"P{index:02d}" for index in sorted(excluded_auto_body_indices)],
        )
        components, interface_retreat_records = apply_visual_interface_retreats(
            vertices,
            faces,
            components,
            visual_semantics.get("interface_retreats", []),
            visual_semantic_min_confidence,
        )
        if interface_retreat_records["applied"]:
            body_component = components[int(body_index) - 1]
            body_separator_evidence = body_selection_separator_evidence(
                vertices=vertices,
                faces=faces,
                components=components,
                model_center=model_center,
                min_faces=(args.small_region_review_max_faces + 1),
            )
            runtime_log(
                "切面内收",
                "interface_retreat_done",
                "已按视觉语义将局部薄边表皮转移到子件，源材料保持不变",
                applied=interface_retreat_records["applied"],
                rejected=interface_retreat_records["rejected"],
            )
        recognition = component_recognition_records(vertices, faces, components, body_component, colors)
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
        invalid_mode_override_indices = sorted(index for index in part_mode_overrides if index < 1 or index > len(components))
        if invalid_mode_override_indices:
            parser.error(
                "--part-mode-overrides references unknown parts: "
                + ", ".join(f"P{index:02d}" for index in invalid_mode_override_indices)
            )
        runtime_log(
            "分类",
            "processing_classification_start",
            "开始分类主体和内嵌部件处理模式",
            components=int(len(components)),
        )
        processing_classifications = classify_component_processing_modes(
            vertices=vertices,
            faces=faces,
            components=components,
            body_component=body_component,
            model_center=model_center,
            global_mode=args.part_processing_mode,
            overrides=part_mode_overrides,
            accept_ambiguous_inward=args.accept_ambiguous_inward,
            precomputed_structural_evidence=body_separator_evidence,
        )
        runtime_log(
            "分类",
            "processing_classification_done",
            "部件处理模式分类完成",
            body_index=body_index,
            inward_parts=int(
                sum(
                    record["selected_processing_mode"] == "inward"
                    for record in processing_classifications.values()
                )
            ),
        )
        for record in recognition:
            record["recognition_basis"] = args.recognition_surface_profile
            record["occluded_paint_excluded"] = args.recognition_surface_profile == "exterior-visible"
            record.update(processing_classifications[int(record["part_index"])])
        progress("识别", f"识别到 {len(recognition)} 个有效部件", region_review="preserve-source", reviewed=len(source_region_classifications))
        stage_artifacts.write_arrays(
            "03_recognition_regions",
            **{f"region_{index:04d}_source_face_ids": np.asarray(component.global_faces, dtype=np.int64)
               for index, component in enumerate(components, start=1)},
        )
        stage_artifacts.write_arrays(
            "03_recognized_boundaries",
            **recognized_boundaries.flattened_arrays(),
        )
        stage_artifacts.write_json(
            "03_recognition_summary",
            {"regions": recognition, "region_review": region_review,
             "source_region_classifications": source_region_classifications,
             "source_geometry_changed": False,
             "recognized_boundaries": {
                 "fingerprint": recognized_boundaries.fingerprint,
                 "component_count": len(recognized_boundaries.component_loops),
                 "loop_count": sum(len(loops) for loops in recognized_boundaries.component_loops),
                 "loop_vertex_counts": [
                     [len(loop) for loop in loops]
                     for loops in recognized_boundaries.component_loops
                 ],
                 "source_vertex_count": recognized_boundaries.source_vertex_count,
                 "source_face_count": recognized_boundaries.source_face_count,
                 "immutable_after_recognition": True,
             }},
        )
        print_recognition(recognition)
        if args.recognize_only or getattr(args, "stop_after_stage", None) == "recognize":
            return
        processing_mode_by_part = {
            index: str(classification["selected_processing_mode"])
            for index, classification in processing_classifications.items()
        }
        runtime_log(
            "装配",
            "assembly_plan_start",
            "开始构建共享边、边界环和递归装配树",
            strategy=args.assembly_tree_strategy,
            components=int(len(components)),
        )
        component_adjacency = component_shared_edges(faces, components)
        boundary_neighbor_lookup = component_boundary_neighbor_lookup(faces, components)
        boundary_loop_neighbors = [
            record
            for index, component in enumerate(components, start=1)
            for record in component_boundary_loop_neighbors(
                vertices, faces, component, index, boundary_neighbor_lookup,
                recognized_boundaries.loops_for_component(index),
            )
        ]
        recursive_minimal_reparents: list[dict] = []
        recursive_assembly_enabled = args.assembly_mode in {"tree", "flat"}
        if recursive_assembly_enabled:
            assembly_parents, assembly_children, assembly_records = self.assembly_planner.build_tree(
                components,
                body_component,
                component_adjacency,
                args.min_assembly_shared_edges,
                args.assembly_tree_strategy,
                boundary_loop_neighbors,
            )
            (
                assembly_parents,
                assembly_children,
                assembly_records,
                mixed_boundary_reparents,
                boundary_loop_neighbors,
            ) = refine_mixed_boundary_parents(
                boundary_loop_neighbors,
                component_adjacency,
                assembly_parents,
                assembly_records,
                args.min_assembly_shared_edges,
            )
            (
                assembly_parents,
                assembly_children,
                assembly_records,
                cycle_breaks,
            ) = break_assembly_parent_cycles(
                components,
                body_index,
                component_adjacency,
                assembly_parents,
                assembly_records,
                args.min_assembly_shared_edges,
            )
            (
                assembly_parents,
                assembly_children,
                assembly_records,
                shared_loop_reparents,
            ) = reparent_shared_parent_child_loops(
                assembly_parents,
                assembly_records,
                boundary_loop_neighbors,
                component_adjacency,
            )
            mixed_boundary_reparents = mixed_boundary_reparents + shared_loop_reparents
            if args.assembly_tree_strategy == "recursive-minimal":
                (
                    assembly_parents,
                    assembly_children,
                    assembly_records,
                    recursive_minimal_reparents,
                ) = refine_recursive_minimal_parents(
                    components,
                    component_adjacency,
                    assembly_parents,
                    assembly_records,
                    boundary_loop_neighbors,
                    args.min_assembly_shared_edges,
                )
                (
                    assembly_parents,
                    assembly_children,
                    assembly_records,
                    recursive_cycle_breaks,
                ) = break_assembly_parent_cycles(
                    components,
                    body_index,
                    component_adjacency,
                    assembly_parents,
                    assembly_records,
                    args.min_assembly_shared_edges,
                )
                cycle_breaks = cycle_breaks + recursive_cycle_breaks
                (
                    assembly_parents,
                    assembly_children,
                    assembly_records,
                    mixed_parent_loop_groups,
                ) = group_mixed_parent_loop_children(
                    components,
                    component_adjacency,
                    assembly_parents,
                    assembly_records,
                    boundary_loop_neighbors,
                )
                mixed_boundary_reparents = (
                    mixed_boundary_reparents + mixed_parent_loop_groups
                )
                runtime_log(
                    "装配诊断",
                    "mixed_parent_loop_grouping_done",
                    "混合父边界环子装配分组完成",
                    parent_map={
                        str(int(child_index)): (
                            None if parent_index is None else int(parent_index)
                        )
                        for child_index, parent_index in sorted(assembly_parents.items())
                    },
                    grouping_changes=mixed_parent_loop_groups,
                )
        else:
            assembly_parents = {
                index: (None if index == body_index else body_index)
                for index in range(1, len(components) + 1)
            }
            assembly_children = collections.defaultdict(list)
            for child_index, parent_index in assembly_parents.items():
                if parent_index is not None:
                    assembly_children[parent_index].append(child_index)
            assembly_children = {
                parent: sorted(children) for parent, children in assembly_children.items()
            }
            assembly_records = [
                {
                    "part_index": index,
                    "parent_index": parent_index,
                    "shared_edges_to_parent": 0,
                    "reason": "legacy_flat_mode_direct_body_insert" if parent_index is not None else "body_root",
                }
                for index, parent_index in sorted(assembly_parents.items())
            ]
            mixed_boundary_reparents = []
            cycle_breaks = []
        visual_semantic_parent_overrides = {"applied": [], "rejected": []}
        if recursive_assembly_enabled and visual_semantics.get("parent_relations"):
            (
                assembly_parents,
                assembly_children,
                assembly_records,
                visual_semantic_parent_overrides,
            ) = apply_visual_semantic_parent_relations(
                components,
                component_adjacency,
                assembly_parents,
                assembly_records,
                visual_semantics.get("parent_relations", []),
                visual_semantic_min_confidence,
            )
        runtime_log(
            "装配",
            "assembly_plan_done",
            "递归装配树和执行顺序已确定",
            strategy=args.assembly_tree_strategy,
            parent_relations=int(
                sum(parent is not None for parent in assembly_parents.values())
            ),
            internal_steps=int(
                sum(bool(children) for children in assembly_children.values())
            ),
        )
        effective_fit_clearance_by_part: dict[int, float] = {}
        clearance_records: list[dict] = []
        for component_index, component in enumerate(components, start=1):
            effective_clearance, clearance_record = effective_feature_clearance(
                component,
                args.fit_clearance_mm,
                profile=args.clearance_profile,
                feature_ratio=args.clearance_feature_ratio,
                minimum_mm=args.clearance_min_mm,
            )
            effective_fit_clearance_by_part[component_index] = effective_clearance
            clearance_records.append({"part_index": component_index, **clearance_record})
        assembly_depth_by_part = assembly_depths(assembly_parents)
        recursive_minimal_layers = build_recursive_minimal_layers(assembly_parents, assembly_children)
        inward_overrides: dict[int, np.ndarray] = {}
        inward_override_records = []
        body_index_for_overrides = component_identity_index(components, body_component)
        body_root_indices = {int(body_index_for_overrides)} if body_index_for_overrides is not None else set()
        for nested_leaf_index, nested_parent_index in assembly_parents.items():
            is_nested_leaf = (
                nested_parent_index is not None
                and int(nested_parent_index) not in body_root_indices
                and not assembly_children.get(nested_leaf_index)
            )
            if not is_nested_leaf:
                continue
            original_clearance = float(effective_fit_clearance_by_part[nested_leaf_index])
            nested_clearance = min(original_clearance, 0.40)
            effective_fit_clearance_by_part[nested_leaf_index] = nested_clearance
            for clearance_record in clearance_records:
                if int(clearance_record["part_index"]) == int(nested_leaf_index):
                    clearance_record["pre_nested_detail_clearance_mm"] = original_clearance
                    clearance_record["effective_fit_clearance_mm"] = nested_clearance
                    clearance_record["nested_detail_clearance_cap_mm"] = 0.40
                    clearance_record["reason"] = (
                        "nested_detail_oblique_wall_exposure_cap"
                        if nested_clearance < original_clearance - 1e-12
                        else clearance_record["reason"]
                    )
                    break
        for child_index, parent_index in assembly_parents.items():
            if parent_index is None or int(parent_index) in body_root_indices:
                continue
            direction = components[parent_index - 1].center - components[child_index - 1].center
            length = float(np.linalg.norm(direction))
            if length <= 1e-9:
                continue
            parent_direction = direction / length
            original_inward = component_inward_direction(vertices, faces, components[child_index - 1], model_center)
            alignment = float(np.dot(original_inward, parent_direction))
            override_record = {
                "part_index": child_index,
                "parent_index": parent_index,
                "alignment_dot": alignment,
                "threshold": float(args.assembly_direction_override_dot),
                "original_inward": original_inward.round(6).tolist(),
                "parent_direction": parent_direction.round(6).tolist(),
                "overridden": bool(alignment < float(args.assembly_direction_override_dot)),
            }
            if override_record["overridden"]:
                inward_overrides[child_index] = parent_direction
            inward_override_records.append(override_record)

        semantic_direction_overrides = {"applied": [], "rejected": []}
        guided_internal_cuts_by_part: dict[int, GuidedInternalCutSpec] = {}
        for part_index, semantic in sorted(visual_semantics.get("parts", {}).items()):
            part_index = int(part_index)
            requested_vector = semantic.get("force_inward_vector")
            force_parent = bool(semantic.get("force_parent_direction", False))
            guided_mapping = semantic.get("guided_internal_cut")
            confidence = float(semantic.get("confidence_score", 0.0))
            rejection = {
                "part_index": part_index,
                "confidence": semantic.get("confidence", "UNKNOWN"),
                "confidence_score": confidence,
            }
            if requested_vector is None and not force_parent and guided_mapping is None:
                continue
            if confidence < visual_semantic_min_confidence:
                rejection["reject_reason"] = "semantic_confidence_below_threshold"
                semantic_direction_overrides["rejected"].append(rejection)
                continue
            if part_index not in range(1, len(components) + 1) or part_index in body_root_indices:
                rejection["reject_reason"] = "unknown_part_or_body"
                semantic_direction_overrides["rejected"].append(rejection)
                continue
            direction_source = "force_inward_vector"
            if guided_mapping is not None:
                try:
                    guided_spec = GuidedInternalCutSpec.from_mapping(
                        guided_mapping
                    )
                except (TypeError, ValueError) as exc:
                    rejection["reject_reason"] = "invalid_guided_internal_cut"
                    rejection["error"] = str(exc)
                    semantic_direction_overrides["rejected"].append(rejection)
                    continue
                parent_index = assembly_parents.get(part_index)
                if parent_index is None:
                    rejection["reject_reason"] = (
                        "guided_internal_cut_without_parent"
                    )
                    semantic_direction_overrides["rejected"].append(rejection)
                    continue
                guided_internal_cuts_by_part[part_index] = guided_spec
                direction = guided_spec.entry_direction.copy()
                direction_source = "guided_internal_cut"
            elif requested_vector is not None:
                try:
                    direction = np.asarray(requested_vector, dtype=np.float64)
                except (TypeError, ValueError):
                    direction = np.asarray([], dtype=np.float64)
                if direction.shape != (3,) or not np.all(np.isfinite(direction)):
                    rejection["reject_reason"] = "invalid_force_inward_vector"
                    semantic_direction_overrides["rejected"].append(rejection)
                    continue
            else:
                parent_index = assembly_parents.get(part_index)
                if parent_index is None:
                    rejection["reject_reason"] = "force_parent_direction_without_parent"
                    semantic_direction_overrides["rejected"].append(rejection)
                    continue
                direction = components[int(parent_index) - 1].center - components[part_index - 1].center
                direction_source = "force_parent_direction"
            length = float(np.linalg.norm(direction))
            if length <= 1e-9:
                rejection["reject_reason"] = "zero_length_direction"
                semantic_direction_overrides["rejected"].append(rejection)
                continue
            normalized = direction / length
            inward_overrides[part_index] = normalized
            applied = {
                **rejection,
                "direction_source": direction_source,
                "normalized_direction": normalized.round(6).tolist(),
                "parent_index": assembly_parents.get(part_index),
            }
            semantic_direction_overrides["applied"].append(applied)
            inward_override_records.append({**applied, "overridden": True})

        layer_cut_references_only = uses_layer_child_cut_references(
            args.assembly_mode, args.assembly_tree_strategy
        )
        if layer_cut_references_only:
            all_cut_refs = []
            runtime_log(
                "几何",
                "global_cut_references_skipped",
                "递归最小装配将按层建立切割引用，跳过无人消费的全局预计算",
                components=int(len(components)),
                assembly_mode=str(args.assembly_mode),
                assembly_tree_strategy=str(args.assembly_tree_strategy),
            )
        else:
            runtime_log(
                "几何",
                "cut_reference_start",
                "开始建立部件切割边界与方向引用",
                components=int(len(components)),
            )
            all_cut_refs = build_component_cut_references(
                vertices,
                faces,
                components,
                model_center,
                inward_overrides,
                fit_clearance_by_part=effective_fit_clearance_by_part,
                clearance_mode=args.clearance_mode,
                recognized_boundaries=recognized_boundaries,
            )
            runtime_log(
                "几何",
                "cut_reference_done",
                "切割边界与方向引用建立完成",
                cut_references=int(len(all_cut_refs)),
            )
        refs_by_component_index: dict[int, list[dict]] = collections.defaultdict(list)
        for ref in all_cut_refs:
            ref["processing_mode"] = "inward"
            refs_by_component_index[int(ref["component_index"])].append(ref)

        def effective_cap_mode(component_index: int) -> str:
            if component_index in force_flat_part_indices:
                return "flat"
            parent_index = assembly_parents.get(component_index)
            is_nested_insert = (
                recursive_assembly_enabled
                and parent_index is not None
                and int(parent_index) not in body_root_indices
            )
            requested_mode = (
                args.nested_cap_mode
                if is_nested_insert and args.nested_cap_mode != "inherit"
                else args.cap_mode
            )
            return requested_mode

        def effective_planar_extra_limit(component_index: int) -> float:
            return float(args.planar_extra_limit_mm)

        layer_child_context_cache: dict[int, tuple[list[dict], dict[int, Component], dict[int, list[int]]]] = {}

        def layer_child_context(parent_index: int) -> tuple[list[dict], dict[int, Component], dict[int, list[int]]]:
            parent_index = int(parent_index)
            if parent_index not in layer_child_context_cache:
                layer_child_context_cache[parent_index] = build_layer_child_cut_references(
                    vertices=vertices,
                    faces=faces,
                    components=components,
                    parent_index=parent_index,
                    direct_child_indices=assembly_children.get(parent_index, []),
                    assembly_children=assembly_children,
                    model_center=model_center,
                    boundary_neighbor_lookup=boundary_neighbor_lookup,
                    inward_overrides=inward_overrides,
                    guided_internal_cuts_by_part=guided_internal_cuts_by_part,
                    effective_cap_mode=effective_cap_mode,
                    effective_planar_extra_limit=effective_planar_extra_limit,
                    fit_clearance_by_part=effective_fit_clearance_by_part,
                    boundary_reconciliation_tolerance_mm=args.fit_clearance_mm,
                    clearance_mode=args.clearance_mode,
                    interface_retopology=interface_retopology,
                    max_extension_mm=args.max_extension_mm,
                    flat_clearance_mm=args.flat_clearance_mm,
                    bottom_clearance_mm=args.bottom_clearance_mm,
                    lead_in_mm=args.lead_in_mm,
                    interface_geometry=args.interface_geometry,
                    recognized_boundaries=recognized_boundaries,
                )
            return layer_child_context_cache[parent_index]

        def direct_child_refs(parent_index: int) -> list[dict]:
            if layer_cut_references_only:
                refs, _union_by_child, _subtree_by_child = layer_child_context(parent_index)
                results = []
                for ref in refs:
                    ref_with_mode = dict(ref)
                    ref_with_mode["processing_mode"] = "inward"
                    results.append(ref_with_mode)
                return results
            refs: list[dict] = []
            for child_index in assembly_children.get(parent_index, []):
                for ref in refs_by_component_index.get(child_index, []):
                    ref_with_mode = dict(ref)
                    ref_with_mode["cap_mode"] = effective_cap_mode(child_index)
                    ref_with_mode["planar_extra_limit_mm"] = effective_planar_extra_limit(
                        child_index
                    )
                    ref_with_mode["processing_mode"] = "inward"
                    refs.append(ref_with_mode)
            return refs

        stage_artifacts.write_json(
            "04_assembly_plan",
            {"body_index": body_index, "strategy": args.assembly_tree_strategy,
             "assembly": assembly_records, "recursive_layers": recursive_minimal_layers,
             "recognized_boundary_fingerprint": recognized_boundaries.fingerprint},
        )
        if getattr(args, "stop_after_stage", None) == "assembly":
            return

        full_tree_preflight_record = {
            "status": "SKIPPED",
            "reason": "disabled_or_non_recursive",
        }
        if bool(args.full_tree_preflight) and recursive_assembly_enabled:
            runtime_log(
                "递归预检",
                "full_tree_preflight_start",
                "开始在首个大布尔前检查整棵装配树",
                parent_count=int(len(recursive_minimal_layers)),
            )
            try:
                full_tree_preflight_record = self.full_tree_preflight.run(
                    recursive_minimal_layers,
                    layer_child_context,
                ).as_record()
            except FullTreePreflightError as exc:
                full_tree_preflight_record = exc.report.as_record()
                print(
                    "full_tree_preflight="
                    + json.dumps(full_tree_preflight_record, ensure_ascii=False),
                    file=sys.stderr,
                    flush=True,
                )
                raise SystemExit(3)
            runtime_log(
                "递归预检",
                "full_tree_preflight_done",
                "整棵装配树的接口安全预检完成",
                interface_count=int(
                    full_tree_preflight_record.get("interface_count", 0)
                ),
                elapsed_seconds=float(
                    full_tree_preflight_record.get("elapsed_seconds", 0.0)
                ),
            )

        stage_artifacts.write_json(
            "05_interface_plan",
            {"preflight": full_tree_preflight_record,
             "recursive_layer_count": len(recursive_minimal_layers),
             "interface_geometry": args.interface_geometry,
             "boundary_shape": args.boundary_shape,
             "recognized_boundary_fingerprint": recognized_boundaries.fingerprint},
        )
        if getattr(args, "stop_after_stage", None) == "interfaces":
            return

        component_centers = {
            index: component.center
            for index, component in enumerate(components, start=1)
        }

        print("assembly_tree:", flush=True)
        progress("装配", "已推断父子装配关系", strategy=args.assembly_tree_strategy)
        for record in assembly_records:
            parent = record.get("parent_index")
            parent_label = "none" if parent is None else f"P{int(parent):02d}"
            print(
                "  P{part_index:02d} parent={parent} shared_edges={shared_edges_to_parent} reason={reason}".format(
                    parent=parent_label,
                    **record,
                ),
                flush=True,
            )

        report = {
            "source": input_path.name,
            "output_3mf": str(output_path),
            "format_profile_requested": args.format_profile,
            "recognition_surface_profile": args.recognition_surface_profile,
            "exterior_surface_recognition": exterior_visibility,
            "format_support": project_settings.get("_format_support", {}),
            "vendor_paint_decode": project_settings.get("_vendor_paint_decode", {}),
            "source_unit": project_settings.get("_source_unit", "millimeter"),
            "unit_scale_mm": project_settings.get("_unit_scale_mm", 1.0),
            "selected_model_entry": project_settings.get("_selected_model_entry"),
            "available_model_entries": project_settings.get("_available_model_entries", []),
            "default_filament_resolution": project_settings.get("_default_filament", {}),
            "source_vertex_count": int(len(vertices)),
            "source_face_count": int(len(faces)),
            "source_mesh_validation": validate_mesh_in_memory(
                trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            ),
            "region_review_policy": "preserve-source",
            "noise_review_max_faces": args.noise_review_max_faces,
            "small_region_review_max_faces": args.small_region_review_max_faces,
            "region_review": region_review,
            "reviewed_source_region_count": len(source_region_classifications),
            "source_region_classifications": source_region_classifications,
            "material_connectivity": material_connectivity,
            "requested_max_extension_mm": requested_max_extension_mm,
            "minimum_flat_bottom_depth_mm": MINIMUM_INWARD_DEPTH_MM,
            "max_extension_mm": args.max_extension_mm,
            "max_planar_travel_mm": maximum_planar_travel_mm,
            "planar_travel_policy": planar_travel_policy,
            "flat_clearance_mm": args.flat_clearance_mm,
            "fit_clearance_mm": max(float(args.fit_clearance_mm), 0.0),
            "bottom_clearance_mm": max(float(args.bottom_clearance_mm), 0.0),
            "lead_in_mm": max(float(args.lead_in_mm), 0.0),
            "sibling_clearance_mm": max(float(args.sibling_clearance_mm), 0.0),
            "cap_mode": args.cap_mode,
            "nested_cap_mode": args.nested_cap_mode,
            "planar_extra_limit_mm": max(float(args.planar_extra_limit_mm), 0.0),
            "force_flat_parts": sorted(force_flat_part_indices),
            "part_processing_mode": args.part_processing_mode,
            "part_mode_overrides": {f"P{index:02d}": mode for index, mode in sorted(part_mode_overrides.items())},
            "accept_ambiguous_inward": bool(args.accept_ambiguous_inward),
            "processing_classifications": [
                processing_classifications[index] for index in sorted(processing_classifications)
            ],
            "source_recognized_component_count": len(components),
            "final_candidate_component_count": len(components),
            "clearance_mode": args.clearance_mode,
            "insert_shrink_mm": insert_shrink_mm,
            "socket_overcut_mm": socket_overcut_mm,
            "top_edge_clearance_mm": top_edge_clearance_mm,
            "interface_retopology_mode": "planar-arc-retopology",
            "boundary_target_samples": args.boundary_target_samples,
            "boundary_smooth_passes": args.boundary_smooth_passes,
            "visible_interface_simplification_tolerance_mm": float(
                args.visible_interface_simplification_tolerance
            ),
            "boundary_retopology_band_mm": args.boundary_retopology_band_mm,
            "boundary_target_slope_deg": interface_retopology.config.target_slope_degrees,
            "boundary_min_slope_deg": interface_retopology.config.minimum_slope_degrees,
            "boundary_max_slope_deg": interface_retopology.config.maximum_slope_degrees,
            "body_strategy": args.body_strategy,
            "body_color": args.body_color,
            "body_index": args.body_index,
            "explicit_body_merge": explicit_body_merge,
            "selected_body_index": body_index_for_overrides,
            "automatic_body_excluded_separator_indices": sorted(excluded_auto_body_indices),
            "body_selection_separator_evidence": [
                {"part_index": int(index), **record}
                for index, record in sorted(body_separator_evidence.items())
            ],
            "assembly_mode": args.assembly_mode,
            "assembly_tree_strategy": args.assembly_tree_strategy,
            "min_assembly_shared_edges": args.min_assembly_shared_edges,
            "assembly_direction_override_dot": args.assembly_direction_override_dot,
            "semantic_direction_overrides": semantic_direction_overrides,
            "guided_internal_cuts": [
                {
                    "part_index": int(index),
                    **spec.as_record(),
                }
                for index, spec in sorted(guided_internal_cuts_by_part.items())
            ],
            "clearance_profile": args.clearance_profile,
            "clearance_records": clearance_records,
            "output_layout": args.output_layout,
            "visual_semantics_source": visual_semantics.get("source"),
            "visual_semantic_min_confidence": args.visual_semantic_min_confidence,
            "visual_semantic_min_confidence_score": visual_semantic_min_confidence,
            "visual_part_semantics": [
                visual_semantics["parts"][index]
                for index in sorted(visual_semantics.get("parts", {}))
            ],
            "visual_semantic_parent_overrides": visual_semantic_parent_overrides,
            "visual_interface_retreats": interface_retreat_records,
            "inward_direction_overrides": inward_override_records,
            "mixed_boundary_reparents": mixed_boundary_reparents,
            "recursive_minimal_reparents": recursive_minimal_reparents,
            "recursive_minimal_layers": recursive_minimal_layers,
            "cycle_breaks": cycle_breaks,
            "boundary_loop_neighbors": boundary_loop_neighbors,
            "component_adjacency": [
                {
                    "part_a": left,
                    "part_b": right,
                    "shared_edges": int(record["shared_edges"]),
                    "shared_vertex_count": int(record["shared_vertex_count"]),
                }
                for (left, right), record in sorted(component_adjacency.items())
            ],
            "assembly_tree": assembly_records,
            "full_tree_preflight": full_tree_preflight_record,
            "project_filament_colours": project_settings.get("filament_colour", []),
            "color_info": COLOR_INFO,
            "recognition": recognition,
            "body_part_id": None,
            "body_part_ids": [],
            "parts": [],
        }
        if args.debug_recursive_steps is not None:
            if not args.debug_recursive_3mf:
                parser.error("--debug-recursive-steps requires --debug-recursive-3mf")
            if int(args.debug_recursive_steps) <= 0:
                parser.error("--debug-recursive-steps must be positive")
        recursive_execution_steps = list(recursive_minimal_layers)
        partial_recursive_debug = args.debug_recursive_steps is not None
        if partial_recursive_debug:
            recursive_execution_steps = recursive_execution_steps[
                : int(args.debug_recursive_steps)
            ]
        source_artifact_sha256 = sha256_file(input_path)
        component_identity = []
        for component_index, component in enumerate(components, start=1):
            face_indices = np.asarray(component.global_faces, dtype=np.int64)
            component_identity.append(
                {
                    "component_index": int(component_index),
                    "color_code": str(component.color_code),
                    "face_count": int(len(face_indices)),
                    "face_indices_sha256": hashlib.sha256(
                        face_indices.tobytes()
                    ).hexdigest(),
                }
            )
        run_fingerprint = fingerprint_payload(
            {
                "source_sha256": source_artifact_sha256,
                "arguments": normalized_run_arguments(args),
                "components": component_identity,
                "assembly_parents": assembly_parents,
                "assembly_children": assembly_children,
                "recursive_steps": recursive_execution_steps,
                "fit_clearance_by_part": effective_fit_clearance_by_part,
                "inward_overrides": inward_overrides,
                "body_index": int(body_index_for_overrides),
            }
        )
        stage_implementation_fingerprint = implementation_fingerprint(
            Path(__file__).resolve().parent
        )
        recursive_stage_cache = (
            None
            if args.resume == "off"
            else RecursiveStageCache(Path(args.cache_dir), mode=args.resume)
        )
        optimization_metrics = {
            "run_fingerprint": run_fingerprint,
            "source_sha256": source_artifact_sha256,
            "stage_cache_mode": str(args.resume),
            "stage_cache_directory": (
                None
                if recursive_stage_cache is None
                else str(recursive_stage_cache.root)
            ),
            "stage_cache_hits": 0,
            "stage_cache_misses": 0,
            "stage_cache_commits": 0,
            "verified_recursive_parse_hits": 0,
            "verified_recursive_parse_misses": 0,
            "verified_recursive_parse_bytes_avoided": 0,
        }
        report["optimization"] = optimization_metrics
        report["boundary_clarity"] = boundary_review.records
        runtime_log(
            "递归缓存",
            "recursive_stage_cache_context_ready",
            "递归阶段缓存上下文已建立",
            run_fingerprint=str(run_fingerprint),
            implementation_fingerprint=str(stage_implementation_fingerprint),
            cache_enabled=bool(recursive_stage_cache is not None),
        )
        runtime_log(
            "递归",
            "strict_recursion_start",
            "开始严格深度优先递归拆件",
            steps=int(len(recursive_execution_steps)),
            root_body_index=body_index_for_overrides,
            debug_output=bool(args.debug_recursive_3mf),
        )
        try:
            (
                strict_layers_dir,
                strict_execution_stage_records,
                strict_active_parts,
                strict_snapshot_records,
            ) = execute_strict_recursive_split(
                output_dir=debug_root,
                report_path_mode="relative",
                input_stem=input_path.stem,
                vertices=vertices,
                faces=faces,
                colors=recursive_face_colors,
                components=components,
                assembly_parents=assembly_parents,
                assembly_children=assembly_children,
                recursive_steps=recursive_execution_steps,
                boundary_neighbor_lookup=boundary_neighbor_lookup,
                component_centers=component_centers,
                inward_overrides=inward_overrides,
                model_center=model_center,
                max_extension_mm=args.max_extension_mm,
                interface_retopology=interface_retopology,
                flat_clearance_mm=args.flat_clearance_mm,
                fit_clearance_by_part=effective_fit_clearance_by_part,
                boundary_reconciliation_tolerance_mm=args.fit_clearance_mm,
                lead_in_mm=args.lead_in_mm,
                clearance_mode=args.clearance_mode,
                sibling_clearance_mm=args.sibling_clearance_mm,
                bottom_clearance_mm=args.bottom_clearance_mm,
                effective_cap_mode=effective_cap_mode,
                effective_planar_extra_limit=effective_planar_extra_limit,
                layer_child_context=layer_child_context,
                validation_profile=args.validation_profile,
                max_topology_defect_ratio=args.max_topology_defect_ratio,
                source_application=project_settings.get("_source_application"),
                source_filament_colors=project_settings.get(
                    "filament_colour", []
                ),
                source_project_settings=project_settings,
                interface_geometry=args.interface_geometry,
                allow_partial=partial_recursive_debug,
                stage_cache=recursive_stage_cache,
                run_fingerprint=run_fingerprint,
                stage_implementation_fingerprint=(
                    stage_implementation_fingerprint
                ),
                source_artifact_sha256=source_artifact_sha256,
                optimization_metrics=optimization_metrics,
                boundary_review=boundary_review,
            )
        except ValueError as exc:
            print(
                "strict_recursive_execution_failure="
                + json.dumps({"error": str(exc)}, ensure_ascii=False),
                file=sys.stderr,
            )
            raise SystemExit(3)
        runtime_log(
            "递归",
            "strict_recursion_done",
            "严格递归拆件完成，最终活动部件已物化",
            steps=int(len(recursive_execution_steps)),
            final_parts=int(len(strict_active_parts)),
            final_active_indices=sorted(strict_active_parts),
        )
        build_arrays = {}
        built_parts = []
        for position, (part_index, entry) in enumerate(sorted(strict_active_parts.items()), start=1):
            mesh = entry.get("mesh") if isinstance(entry, dict) else None
            if mesh is None:
                continue
            build_arrays[f"part_{position:04d}_vertices"] = np.asarray(mesh.vertices)
            build_arrays[f"part_{position:04d}_faces"] = np.asarray(mesh.faces)
            built_parts.append({"part_index": int(part_index), "part_id": entry.get("part_id"),
                                "vertex_count": len(mesh.vertices), "face_count": len(mesh.faces)})
        if build_arrays:
            stage_artifacts.write_arrays("06_recursive_build_meshes", **build_arrays)
        stage_artifacts.write_json(
            "06_recursive_build_summary",
            {"parts": built_parts, "stage_records": strict_execution_stage_records,
             "completed_steps": len(recursive_execution_steps),
             "recognized_boundary_fingerprint": recognized_boundaries.fingerprint},
        )
        if getattr(args, "stop_after_stage", None) == "build":
            return

        if partial_recursive_debug:
            latest_snapshot = (
                strict_snapshot_records[-1] if strict_snapshot_records else {}
            )
            print(
                "partial_recursive_debug="
                + json.dumps(
                    {
                        "completed_steps": int(len(recursive_execution_steps)),
                        "requested_steps": int(args.debug_recursive_steps),
                        "active_part_indices": sorted(strict_active_parts),
                        "cumulative_3mf": latest_snapshot.get("output_3mf"),
                        "debug_directory": str(strict_layers_dir),
                        "interface_geometry": str(args.interface_geometry),
                        "final_deliverable_written": False,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return

        debug_layers_dir = strict_layers_dir if args.debug_recursive_3mf else None
        debug_stage_records = (
            strict_execution_stage_records if args.debug_recursive_3mf else []
        )
        print(
            "strict_recursive_execution="
            + json.dumps(
                {
                    "execution_order": "strict_depth_first_preorder",
                    "step_count": len(recursive_execution_steps),
                    "final_active_indices": sorted(strict_active_parts),
                    "parent_emitted_part_3mf_is_recursive_input": True,
                    "cumulative_3mf_is_recursive_input": False,
                    "cumulative_debug_snapshots": len(strict_snapshot_records)
                    if args.debug_recursive_3mf
                    else 0,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        report["strict_recursive_execution"] = {
            "execution_order": "strict_depth_first_preorder",
            "step_count": len(recursive_execution_steps),
            "final_active_indices": sorted(strict_active_parts),
            "parent_emitted_part_3mf_is_recursive_input": True,
            "cumulative_3mf_is_recursive_input": False,
            "debug_snapshot_mode": "standalone_colored_parts_plus_cumulative_audit",
            "snapshots": strict_snapshot_records if args.debug_recursive_3mf else [],
        }
        if args.debug_recursive_3mf:
            recursive_stage_cap_audit = []
            for stage_record in debug_stage_records:
                loop_audit = []
                for extension in stage_record.get("loop_extensions", []):
                    loop_audit.append(
                        {
                            "loop_index": extension.get("loop_index"),
                            "cap_mode": extension.get("cap_mode"),
                            "flat_orientation": extension.get("flat_orientation"),
                            "global_flat_extension_max_mm": extension.get("global_flat_extension_max_mm"),
                            "best_fit_flat_extension_max_mm": extension.get("best_fit_flat_extension_max_mm"),
                            "best_fit_flat_normal_dot_global": extension.get("best_fit_flat_normal_dot_global"),
                            "planar_extra_limit_mm": extension.get("planar_extra_limit_mm"),
                            "local_fallback_reason": extension.get("local_fallback_reason"),
                        }
                    )
                if loop_audit:
                    recursive_stage_cap_audit.append(
                        {
                            "part_id": stage_record.get("part_id"),
                            "geometry_cap_mode": stage_record.get("geometry_cap_mode"),
                            "loops": loop_audit,
                        }
                    )
            print(
                "recursive_stage_cap_audit="
                + json.dumps(recursive_stage_cap_audit, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
            invalid_debug_parts = [
                part.get("part_id")
                for part in debug_stage_records
                if not part.get("reload_watertight")
                or part.get("reload_open_edges")
                or part.get("reload_over_shared_edges")
            ]
            max_debug_defect_ratio = max(float(args.max_topology_defect_ratio), 0.0)
            blocking_debug_parts = [
                part.get("part_id")
                for part in debug_stage_records
                if (
                    not part.get("reload_watertight")
                    or part.get("reload_open_edges")
                    or part.get("reload_over_shared_edges")
                )
                and (
                    args.validation_profile == "strict"
                    or float(part.get("reload_topology_defect_ratio", 1.0)) > max_debug_defect_ratio
                )
            ]
            print(
                "debug_recursive_3mf="
                + json.dumps(
                    {
                        "directory": str(debug_layers_dir),
                        "part_count": len(debug_stage_records),
                        "invalid_parts": invalid_debug_parts,
                        "blocking_parts": blocking_debug_parts,
                        "inward_stage_count": len(recursive_minimal_layers),
                        "snapshot_count": len(strict_snapshot_records),
                        "snapshot_mode": "standalone_colored_parts_plus_cumulative_audit",
                        "recursive_input_mode": "parent_emitted_standalone_3mf",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if blocking_debug_parts:
                if not args.diagnostic_preview:
                    raise SystemExit(3)
                progress("验证", "递归 3MF 含超过当前容差的非水密部件；诊断预览模式继续", parts=blocking_debug_parts)
            elif invalid_debug_parts:
                progress(
                    "验证",
                    "递归 3MF 含少量源网格缺陷；比例容差内继续",
                    parts=invalid_debug_parts,
                    max_topology_defect_ratio=max_debug_defect_ratio,
                )

        colored_3mf_parts = []
        for index, component in enumerate(components, start=1):
            raw_token_counts = collections.Counter(str(colors[int(face_id)]) for face_id in component.global_faces)
            raw_color_tokens = [
                {"token": token, "faces": int(count)}
                for token, count in sorted(raw_token_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
            ]
            raw_color_token = "|".join(item["token"] for item in raw_color_tokens)
            color_info = COLOR_INFO.get(component.color_code, {"name": component.color_code, "hex": "", "rgba": [200, 200, 200, 255]})
            mapping_source = color_info.get("mapping_source", "unmapped")
            resolution_status = color_resolution_status(color_info)
            filament_slot_index = color_info.get("filament_slot")
            filament_slot_number = None if filament_slot_index is None else int(filament_slot_index) + 1
            part_id = f"P{index:02d}_{component.color_code}_{sanitize_name(color_info['name'])}"
            part_cap_mode = effective_cap_mode(index)
            part_fit_clearance_mm = float(effective_fit_clearance_by_part[index])
            part_insert_shrink_mm, part_socket_overcut_mm = clearance_offsets(
                args.clearance_mode, part_fit_clearance_mm
            )
            part_top_edge_clearance_mm = visible_top_edge_clearance(
                part_insert_shrink_mm
            )
            selected_processing_mode = processing_mode_by_part[index]
            progress("拆分", f"正在生成 P{index:02d}/{len(components):02d}", color_code=component.color_code, cap_mode=part_cap_mode)
            if index in body_root_indices:
                part_id = f"{part_id}_BODY_CUT"
            strict_entry = strict_active_parts.get(index)
            if strict_entry is None:
                raise RuntimeError(
                    f"strict recursive execution did not materialize P{index:02d}"
                )
            mesh = strict_entry["mesh"]
            stats = dict(strict_entry["stats"])
            stats["part_id"] = part_id
            stats["strict_recursive_origin_step"] = int(
                strict_entry.get("origin_step", 0)
            )
            stats["strict_recursive_final_state_role"] = strict_entry.get(
                "state_role"
            )
            stats["strict_recursive_parent_part_3mf_is_input"] = True
            stats["strict_recursive_cumulative_3mf_is_input"] = False
            classification = processing_classifications[index]
            stats["selected_processing_mode"] = selected_processing_mode
            stats["suggested_processing_mode"] = classification["suggested_processing_mode"]
            stats["processing_mode_status"] = classification["processing_mode_status"]
            stats["processing_mode_confidence"] = classification["processing_mode_confidence"]
            stats["processing_mode_evidence"] = classification["processing_mode_evidence"]
            stats["forced_flat_cap"] = bool(index in force_flat_part_indices)
            stats["clearance_profile"] = args.clearance_profile
            stats["requested_fit_clearance_mm"] = float(args.fit_clearance_mm)
            stats["effective_fit_clearance_mm"] = part_fit_clearance_mm
            stats["raw_color_token"] = raw_color_token
            stats["raw_color_tokens"] = raw_color_tokens
            stats["filament_slot_index"] = filament_slot_index
            stats["filament_slot_number"] = filament_slot_number
            stats["color_mapping_source"] = mapping_source
            stats["color_resolution_status"] = resolution_status
            stats["color_is_fallback"] = resolution_status == "fallback_estimate"
            stats["assembly_parent_index"] = assembly_parents.get(index)
            stats["assembly_child_indices"] = assembly_children.get(index, [])
            stats["assembly_depth"] = int(assembly_depth_by_part.get(index, 0))
            stats["recursive_minimal_role"] = (
                "root_body"
                if assembly_parents.get(index) is None
                else ("subassembly_body" if assembly_children.get(index) else "leaf_insert")
            )
            if selected_processing_mode == "inward":
                mesh, scaling = scale_finished_insert(mesh, args.post_split_uniform_scale)
                stats["post_split_uniform_scaling"] = scaling
                runtime_log("配合", "post_split_uniform_scale", "原尺寸扣除完成，公件 XYZ 等比缩小",
                            part_id=part_id, **scaling)
            mesh_validation = validate_mesh_in_memory(mesh)
            source_validation = report["source_mesh_validation"]
            mesh_validation["source_mesh_had_defects"] = bool(
                source_validation["open_edges"] or source_validation["over_shared_edges"]
            )
            mesh_validation["attribution"] = (
                "source_contains_topology_defects_exact_edge_provenance_unresolved"
                if mesh_validation["source_mesh_had_defects"] and (mesh_validation["open_edges"] or mesh_validation["over_shared_edges"])
                else "no_remaining_defect"
            )
            stats["mesh_validation"] = mesh_validation
            if index in body_root_indices:
                report.setdefault("body_part_ids", []).append(part_id)
                if report.get("body_part_id") is None:
                    report["body_part_id"] = part_id
            report["parts"].append(stats)
            colored_3mf_parts.append(
                {
                    "part_id": part_id,
                    "color_code": component.color_code,
                    "color_name": color_info["name"],
                    "color_hex": color_info["hex"],
                    "filament_slot_index": filament_slot_index,
                    "filament_slot_number": filament_slot_number,
                    "color_mapping_source": mapping_source,
                    "color_resolution_status": resolution_status,
                    "source_surface_face_count": int(
                        max(
                            0,
                            int(stats.get("source_faces", 0))
                            - int(stats.get("dropped_non_parent_source_faces", 0)),
                        )
                    ),
                    "mesh": mesh,
                    **{field: strict_entry[field] for field in
                       ('face_color_hexes', 'face_filament_slot_indices', 'face_color_codes')
                       if field in strict_entry},
                    "annotation": {
                        "recognition_basis": args.recognition_surface_profile,
                        "occluded_paint_excluded": args.recognition_surface_profile == "exterior-visible",
                        "processing_mode": stats["processing_mode"],
                        "selected_processing_mode": selected_processing_mode,
                        "suggested_processing_mode": classification["suggested_processing_mode"],
                        "processing_mode_status": classification["processing_mode_status"],
                        "processing_mode_confidence": classification["processing_mode_confidence"],
                        "processing_mode_evidence": classification["processing_mode_evidence"],
                        "raw_color_token": raw_color_token,
                        "raw_color_tokens": raw_color_tokens,
                        "filament_slot_index": filament_slot_index,
                        "filament_slot_number": filament_slot_number,
                        "color_mapping_source": mapping_source,
                        "color_resolution_status": resolution_status,
                        "color_is_fallback": resolution_status == "fallback_estimate",
                        "semantic_label": stats.get("visual_semantic_label"),
                        "parent_part": None if assembly_parents.get(index) is None else f"P{int(assembly_parents[index]):02d}",
                        "assembly_depth": int(assembly_depth_by_part.get(index, 0)),
                        "strict_recursive_origin_step": int(
                            stats.get("strict_recursive_origin_step", 0)
                        ),
                        "strict_recursive_final_state_role": stats.get(
                            "strict_recursive_final_state_role"
                        ),
                        "strict_recursive_parent_part_3mf_is_input": True,
                        "strict_recursive_cumulative_3mf_is_input": False,
                        "cap_mode": part_cap_mode,
                        "geometry_cap_mode": stats.get("geometry_cap_mode", part_cap_mode),
                        "fixed_inward_depth_mm": stats.get("fixed_inward_depth_mm", args.max_extension_mm),
                        "maximum_generated_inward_travel_mm": stats.get("maximum_generated_inward_travel_mm", 0.0),
                        "interface_retopology_mode": "planar-arc-retopology",
                        "boundary_target_samples": int(args.boundary_target_samples),
                        "boundary_smooth_passes": int(args.boundary_smooth_passes),
                        "visible_interface_simplification_tolerance_mm": float(
                            args.visible_interface_simplification_tolerance
                        ),
                        "boundary_retopology_band_mm": float(args.boundary_retopology_band_mm),
                        "boundary_target_slope_deg": float(interface_retopology.config.target_slope_degrees),
                        "boundary_min_slope_deg": float(interface_retopology.config.minimum_slope_degrees),
                        "boundary_max_slope_deg": float(interface_retopology.config.maximum_slope_degrees),
                        "interface_retopology_records": stats.get("interface_retopology_records", []),
                        "local_inward_direction_records": stats.get("local_inward_direction_records", []),
                        "local_inward_outward_vertices_before": stats.get("local_inward_outward_vertices_before", 0),
                        "local_inward_outward_vertices_after": stats.get("local_inward_outward_vertices_after", 0),
                        "planar_extra_limit_mm": effective_planar_extra_limit(index),
                        "maximum_planar_inward_travel_mm": maximum_planar_travel_mm,
                        "requested_fit_clearance_mm": float(args.fit_clearance_mm),
                        "effective_fit_clearance_mm": part_fit_clearance_mm,
                        "fit_clearance_mm": part_fit_clearance_mm,
                        "fit_strategy": "exact_subtract_then_uniform_scale",
                        "post_split_uniform_scaling": stats.get("post_split_uniform_scaling"),
                        "lead_in_mm": float(args.lead_in_mm),
                        "validation_level": (
                            "strict"
                            if mesh_validation["watertight"]
                            and mesh_validation["winding_consistent"]
                            and not mesh_validation.get("inward_closed_components")
                            and not (
                                mesh_validation["open_edges"]
                                or mesh_validation["over_shared_edges"]
                                or mesh_validation["inconsistent_shared_edges"]
                            )
                            else "diagnostic_invalid"
                        ),
                        "mesh_validation": mesh_validation,
                    },
                }
            )
            runtime_log(
                "部件",
                "final_part_ready",
                f"最终部件 P{index:02d}/{len(components):02d} 已完成拓扑检查",
                part_id=part_id,
                faces=int(len(mesh.faces)),
                vertices=int(len(mesh.vertices)),
                watertight=bool(mesh_validation["watertight"]),
                winding_consistent=bool(mesh_validation["winding_consistent"]),
                open_edges=int(mesh_validation["open_edges"]),
            )
            print(
                f"{part_id}: color_name={stats['color_name']} color_hex={stats['color_hex']} "
                f"raw_token={raw_color_token} filament_slot={filament_slot_number if filament_slot_number is not None else 'unmapped'} "
                f"mapping_source={mapping_source} resolution={resolution_status} faces={stats['source_faces']} "
                f"out={stats['output_faces']} loops={stats['boundary_loops']} "
                f"mode={stats['processing_mode']} watertight={mesh_validation['watertight']} "
                f"winding_consistent={mesh_validation['winding_consistent']} "
                f"open_edges={mesh_validation['open_edges']} over_edges={mesh_validation['over_shared_edges']} "
                f"inconsistent_edges={mesh_validation['inconsistent_shared_edges']}",
                flush=True,
            )

        inward_depth_audit = [
            {
                "part_id": part.get("part_id"),
                "selected_processing_mode": part.get("selected_processing_mode"),
                "cap_mode": part.get("geometry_cap_mode", part.get("cap_mode")),
                "target_depth_mm": float(part.get("fixed_inward_depth_mm", 0.0)),
                "maximum_generated_inward_travel_mm": float(part.get("maximum_generated_inward_travel_mm", 0.0)),
                "local_inward_outward_vertices_before": int(part.get("local_inward_outward_vertices_before", 0)),
                "local_inward_outward_vertices_after": int(part.get("local_inward_outward_vertices_after", 0)),
            }
            for part in report["parts"]
        ]
        print("inward_depth_audit=" + json.dumps(inward_depth_audit, ensure_ascii=False), flush=True)

        runtime_log(
            "视觉验证",
            "multiview_validation_start",
            "开始确定性多视角表面一致性验证",
            profile=args.visual_validation_profile,
            views=int(args.visual_validation_view_count),
            resolution=int(args.visual_validation_resolution),
            generated_parts=int(len(colored_3mf_parts)),
        )
        from .cutting_reference import attach_cutting_references
        attach_cutting_references(colored_3mf_parts, {
            f'P{int(index):02d}': entry['stats'].get('complete_child_boolean_record', {})
            for index, entry in strict_active_parts.items()})
        manual_adjustment_required = False
        seating_validation = {}
        if args.visual_validation_profile == "off":
            visual_surface_validation = {
                "valid": True,
                "skipped": True,
                "profile": "off",
                "requires_computer_use": False,
                "slicer_screenshot_policy": "ask_user_to_open_final_3mf_and_provide_screenshots",
            }
        else:
            source_part_by_face = np.zeros(len(faces), dtype=np.int32)
            for component_index, component in enumerate(components, start=1):
                source_part_by_face[np.asarray(component.global_faces, dtype=np.int64)] = component_index
            from .assembly_visibility import validate_and_seat_assembly
            seating_validation = validate_and_seat_assembly(
                colored_3mf_parts, vertices, faces, source_part_by_face,
                area_budget_mm2=args.micro_defect_area_mm2,
                allow_coupled_seating=args.allow_coupled_seating,
                overlap_tolerance_mm3=args.seating_overlap_tolerance_mm3,
                ignore_overlap_ratio=args.assembly_ignore_overlap_ratio,
                penetration_tolerance_mm=args.seating_penetration_tolerance_mm,
                recovery_dir=args.recovery_dir,
                post_fit_difference=args.post_fit_parent_difference,
                surface_tolerance_mm=args.print_surface_tolerance_mm,
                allow_manual_adjustment=args.assembly_fit_validation == 'manual',
            )
            report['insert_surface_visibility'] = seating_validation
            for part, stats in zip(colored_3mf_parts, report['parts']):
                trimming = part['annotation'].get('post_fit_difference')
                if trimming:
                    stats['post_fit_difference'] = trimming
                    stats['mesh_validation'] = validate_mesh_in_memory(part['mesh'])
                    part['annotation']['mesh_validation'] = stats['mesh_validation']
                    stats['output_faces'] = int(len(part['mesh'].faces))
                    stats['output_vertices'] = int(len(part['mesh'].vertices))
                    stats['bbox_min'] = part['mesh'].bounds[0].tolist()
                    stats['bbox_max'] = part['mesh'].bounds[1].tolist()
                seating = part['annotation'].get('post_fit_seating')
                if seating is not None:
                    stats['post_fit_seating'] = seating
                    stats['bbox_min'] = part['mesh'].bounds[0].tolist()
                    stats['bbox_max'] = part['mesh'].bounds[1].tolist()
                    runtime_log('配合', 'insert_seating_corrected',
                                '按实测干涉校正公件落座位置，形状和槽位保持不变',
                                part_id=part['part_id'], **seating)
            visual_surface_validation = validate_multiview_visual_consistency(
                vertices,
                faces,
                source_part_by_face,
                colored_3mf_parts,
                view_count=args.visual_validation_view_count,
                resolution=args.visual_validation_resolution,
                depth_tolerance_mm=args.visual_depth_tolerance_mm,
                max_intrusion_ratio=args.visual_max_intrusion_ratio,
                max_material_mismatch_ratio=args.visual_max_material_mismatch_ratio,
                max_local_material_mismatch_ratio=args.visual_max_local_material_mismatch_ratio,
                max_local_material_mismatch_pixels=args.visual_max_local_material_mismatch_pixels,
                local_material_mismatch_gate=(
                    "both"
                    if args.surface_band_validation == "advisory"
                    else "either"
                ),
                min_coverage_ratio=args.visual_min_coverage_ratio,
            )
            visual_surface_validation["profile"] = args.visual_validation_profile
            if args.assembly_fit_validation == 'manual':
                from .assembly_review import annotate_manual_adjustment
                manual_adjustment_required = annotate_manual_adjustment(
                    colored_3mf_parts, seating_validation, visual_surface_validation)
                visual_surface_validation['manual_adjustment_required'] = manual_adjustment_required
                if manual_adjustment_required:
                    progress('装配', '拆件继续导出，请观察装配差异并判断是否影响打印，必要时手动调整',
                             affected_parts=seating_validation.get('affected_parts', []),
                             visual_errors=visual_surface_validation.get('errors', []))
        runtime_log(
            "视觉验证",
            "multiview_validation_done",
            "多视角表面一致性验证完成",
            valid=bool(visual_surface_validation.get("valid", False)),
            skipped=bool(visual_surface_validation.get("skipped", False)),
            coverage_ratio=visual_surface_validation.get("coverage_ratio"),
            intrusion_ratio=visual_surface_validation.get(
                "generated_surface_intrusion_ratio"
            ),
            mismatch_ratio=visual_surface_validation.get(
                "front_material_mismatch_ratio"
            ),
        )
        report["visual_surface_validation"] = visual_surface_validation
        fit_arrays = {}
        fit_parts = []
        for position, part in enumerate(colored_3mf_parts, start=1):
            mesh = part.get("mesh")
            if mesh is None:
                continue
            fit_arrays[f"part_{position:04d}_vertices"] = np.asarray(mesh.vertices)
            fit_arrays[f"part_{position:04d}_faces"] = np.asarray(mesh.faces)
            fit_parts.append({"part_id": part.get("part_id"),
                              "vertex_count": len(mesh.vertices),
                              "face_count": len(mesh.faces)})
        if fit_arrays:
            stage_artifacts.write_arrays("07_fitted_assembly_meshes", **fit_arrays)
        stage_artifacts.write_json(
            "07_fitted_assembly_summary",
            {"parts": fit_parts, "seating_validation": seating_validation,
             "visual_surface_validation": visual_surface_validation,
             "recognized_boundary_fingerprint": recognized_boundaries.fingerprint},
        )
        if getattr(args, "stop_after_stage", None) == "fit":
            return
        print(
            "visual_surface_validation="
            + json.dumps(visual_surface_validation, ensure_ascii=False),
            flush=True,
        )
        if (
            args.visual_validation_profile == "strict"
            and not visual_surface_validation.get("valid", False)
            and not args.diagnostic_preview
            and not manual_adjustment_required
        ):
            print(
                "visual_validation_failures="
                + json.dumps(visual_surface_validation.get("errors", []), ensure_ascii=False),
                file=sys.stderr,
            )
            raise SystemExit(3)

        invalid_part_records = [
            {
                "part_id": part.get("part_id"),
                "open_edges": part["mesh_validation"]["open_edges"],
                "over_shared_edges": part["mesh_validation"]["over_shared_edges"],
                "winding_consistent": part["mesh_validation"]["winding_consistent"],
                "inconsistent_shared_edges": part["mesh_validation"]["inconsistent_shared_edges"],
                "unique_edges": part["mesh_validation"]["unique_edges"],
                "defect_edges": part["mesh_validation"]["defect_edges"],
                "topology_defect_ratio": part["mesh_validation"]["topology_defect_ratio"],
                "open_edge_ratio": part["mesh_validation"]["open_edge_ratio"],
                "over_shared_edge_ratio": part["mesh_validation"]["over_shared_edge_ratio"],
                "inconsistent_orientation_ratio": part["mesh_validation"]["inconsistent_orientation_ratio"],
                "inward_closed_components": part["mesh_validation"].get(
                    "inward_closed_components"
                ),
                "all_closed_components_outward": part["mesh_validation"].get(
                    "all_closed_components_outward"
                ),
                "open_edge_metrics": part["mesh_validation"]["open_edge_metrics"],
                "over_shared_edge_metrics": part["mesh_validation"]["over_shared_edge_metrics"],
                "inconsistent_edge_metrics": part["mesh_validation"]["inconsistent_edge_metrics"],
                "bbox_min": part.get("bbox_min"),
                "bbox_max": part.get("bbox_max"),
                "source_mesh_had_defects": part["mesh_validation"].get("source_mesh_had_defects", False),
                "attribution": part["mesh_validation"].get("attribution"),
            }
            for part in report["parts"]
            if not part["mesh_validation"]["watertight"]
            or not part["mesh_validation"]["winding_consistent"]
            or part["mesh_validation"]["open_edges"]
            or part["mesh_validation"]["over_shared_edges"]
            or part["mesh_validation"]["inconsistent_shared_edges"]
            or part["mesh_validation"].get("inward_closed_components")
        ]
        invalid_parts = [record["part_id"] for record in invalid_part_records]
        max_topology_defect_ratio = max(float(args.max_topology_defect_ratio), 0.0)
        ratio_accepted_part_records: list[dict] = []
        blocking_part_records = list(invalid_part_records)

        if invalid_part_records and args.validation_profile == "ratio":
            for record in invalid_part_records:
                standard_ratio_accepted = bool(
                    not record["inconsistent_shared_edges"]
                    and not record.get("inward_closed_components")
                    and record["winding_consistent"]
                    and float(record["topology_defect_ratio"])
                    <= max_topology_defect_ratio
                )
                localized_accepted, localized_limits = localized_short_open_edge_acceptance(
                    record,
                    max_topology_defect_ratio,
                )
                if standard_ratio_accepted or localized_accepted:
                    accepted_record = dict(record)
                    accepted_record["ratio_acceptance_reason"] = (
                        "standard_defect_ratio"
                        if standard_ratio_accepted
                        else "localized_short_open_edges"
                    )
                    accepted_record["localized_open_edge_limits"] = localized_limits
                    ratio_accepted_part_records.append(accepted_record)
            ratio_accepted_ids = {record["part_id"] for record in ratio_accepted_part_records}
            blocking_part_records = [record for record in invalid_part_records if record["part_id"] not in ratio_accepted_ids]
        if invalid_parts:
            machine_record(
                "topology_findings",
                {"parts": invalid_part_records},
            )
            if ratio_accepted_part_records:
                machine_record(
                    "ratio_accepted_topology_findings",
                    {
                        "parts": ratio_accepted_part_records,
                        "max_topology_defect_ratio": max_topology_defect_ratio,
                        "blocking": False,
                    },
                )
            if blocking_part_records:
                machine_record(
                    "validation_failures",
                    {"parts": blocking_part_records},
                    blocking=True,
                )
            if blocking_part_records and not args.diagnostic_preview:
                raise SystemExit(3)
            if ratio_accepted_part_records and not blocking_part_records:
                progress(
                    "验证",
                    "存在少量局部拓扑边；异常边比例符合打印参考，继续导出",
                    parts=[record["part_id"] for record in ratio_accepted_part_records],
                    max_topology_defect_ratio=max_topology_defect_ratio,
                )
                ratio_accepted_ids = {record["part_id"] for record in ratio_accepted_part_records}
                for part in colored_3mf_parts:
                    if part["part_id"] in ratio_accepted_ids:
                        part["annotation"]["validation_level"] = "ratio_accepted"
                        part["annotation"]["topology_ratio_validation"] = {
                            "max_topology_defect_ratio": max_topology_defect_ratio,
                            "topology_defect_ratio": next(
                                float(record["topology_defect_ratio"])
                                for record in ratio_accepted_part_records
                                if record["part_id"] == part["part_id"]
                            ),
                        }
            else:
                progress("验证", "严格水密验证未通过，按请求输出诊断预览", parts=invalid_parts)
        else:
            progress("验证", "全部部件通过严格水密验证", parts=len(report["parts"]))

        temporary_output = output_path.with_name(output_path.name + ".tmp")
        ratio_acceptance_active = bool(ratio_accepted_part_records and not blocking_part_records)
        runtime_log(
            "导出",
            "final_package_write_start",
            "开始写入临时分组多部件 3MF",
            temporary_output=str(temporary_output),
            parts=int(len(colored_3mf_parts)),
            layout=args.output_layout,
        )
        export_summary = self.writer.write(
            temporary_output,
            colored_3mf_parts,
            title=(
                f"DIAGNOSTIC INVALID PREVIEW - {input_path.stem}"
                if blocking_part_records
                else (
                    f"ASSEMBLY REVIEW - {input_path.stem} split parts"
                    if manual_adjustment_required
                    else (f"RATIO ACCEPTED - {input_path.stem} split printable parts"
                          if ratio_acceptance_active else f"{input_path.stem} split printable parts")
                )
            ),
            source_application=project_settings.get("_source_application"),
            source_filament_colors=source_filament_colors,
            source_project_settings=project_settings,
            output_layout=args.output_layout,
        )
        runtime_log(
            "导出",
            "final_package_write_done",
            "临时分组多部件 3MF 写入完成",
            temporary_output=str(temporary_output),
            objects=int(export_summary["part_count"]),
            build_items=int(export_summary["build_item_count"]),
        )
        runtime_log(
            "验证",
            "final_package_reload_start",
            "开始从磁盘回读临时 3MF 并复检结构、颜色和耗材槽",
            temporary_output=str(temporary_output),
        )
        three_mf_validation = self.validator.validate_package(
            temporary_output,
            colored_3mf_parts,
            source_filament_colors=source_filament_colors,
            source_application=project_settings.get("_source_application"),
            source_project_settings=project_settings,
            output_layout=args.output_layout,
        )
        runtime_log(
            "验证",
            "final_package_reload_done",
            "临时 3MF 磁盘回读复检完成",
            valid=bool(three_mf_validation.get("valid", False)),
            errors=int(len(three_mf_validation.get("errors", []))),
            objects=int(len(three_mf_validation.get("objects", []))),
        )
        if not three_mf_validation["valid"]:
            known_topology_errors = [
                error
                for error in three_mf_validation["errors"]
                if "reloaded mesh is not watertight" in error
                or "reloaded mesh has " in error and ("open edges" in error or "over-shared edges" in error)
            ]
            package_errors = [error for error in three_mf_validation["errors"] if error not in known_topology_errors]
            allow_known_topology = bool(ratio_acceptance_active or args.diagnostic_preview)
            if not allow_known_topology or package_errors:
                temporary_output.unlink(missing_ok=True)
                print(
                    "3mf_validation_failures=" + json.dumps(three_mf_validation["errors"], ensure_ascii=False),
                    file=sys.stderr,
                )
                raise SystemExit(3)
            three_mf_validation["strict_valid"] = False
            three_mf_validation["package_valid"] = True
            three_mf_validation["known_topology_errors"] = known_topology_errors
            three_mf_validation["validation_profile"] = (
                "ratio" if ratio_acceptance_active else "diagnostic-preview"
            )
            three_mf_validation["errors"] = []
            three_mf_validation["valid"] = True
            progress("验证", "3MF 包结构与颜色复检通过；仅保留已标注的网格缺陷", topology_errors=len(known_topology_errors))
        temporary_output.replace(output_path)
        stage_artifacts.write_json(
            "08_validated_publish",
            {"output_3mf": str(output_path.resolve()), "validation": three_mf_validation,
             "part_count": len(colored_3mf_parts),
             "recognized_boundary_fingerprint": recognized_boundaries.fingerprint},
        )
        runtime_log(
            "发布",
            "atomic_publish_done",
            "验证通过的临时 3MF 已原子发布",
            output=str(output_path),
            parts=int(len(colored_3mf_parts)),
        )

        final_summary = {
            "output_3mf": str(output_path),
            "part_count": len(colored_3mf_parts),
            "body_part_id": report["body_part_id"],
            "body_part_ids": report.get("body_part_ids", []),
            "assembly_mode": args.assembly_mode,
            "assembly_tree_strategy": args.assembly_tree_strategy,
            "output_layout": args.output_layout,
            "semantic_direction_overrides": semantic_direction_overrides,
            "clearance_profile": args.clearance_profile,
            "clearance_records": clearance_records,
            "planar_travel_policy": planar_travel_policy,
            "part_processing_mode": args.part_processing_mode,
            "processing_classifications": [
                processing_classifications[index] for index in sorted(processing_classifications)
            ],
            "inward_depth_audit": inward_depth_audit,
            "recognition_surface_profile": args.recognition_surface_profile,
            "exterior_surface_recognition": exterior_visibility,
            "recursive_minimal_layers": recursive_minimal_layers,
            "strict_recursive_execution": report["strict_recursive_execution"],
            "visual_surface_validation": visual_surface_validation,
            "colors": export_summary["colors"],
            "source_filament_colors": export_summary.get("source_filament_colors", []),
            "source_filament_palette_preserved": export_summary.get("source_filament_palette_preserved", False),
            "objects": export_summary["objects"],
            "source_instance_transform": project_settings.get("_source_instance_transform"),
            "output_application": export_summary.get("application"),
            "bambu_project_compatible": export_summary.get("bambu_project_compatible", False),
            "validation": three_mf_validation,
            "validation_level": (
                "diagnostic_invalid"
                if blocking_part_records
                else ('manual_adjustment_required' if manual_adjustment_required
                      else ("ratio_accepted" if ratio_acceptance_active else "strict_validated"))
            ),
            "validation_profile": args.validation_profile,
            "max_topology_defect_ratio": max_topology_defect_ratio,
            "invalid_parts": invalid_parts,
            "ratio_accepted_parts": [record["part_id"] for record in ratio_accepted_part_records],
            "debug_recursive_3mf_directory": str(debug_layers_dir) if debug_layers_dir else None,
            "debug_recursive_3mf_part_count": len(debug_stage_records),
        }
        progress("导出", "已写入彩色多部件 3MF", output=str(output_path), validation_level=final_summary["validation_level"])
        if args.recovery_dir:
            (Path(args.recovery_dir) / 'final_summary.json').write_text(
                json.dumps(final_summary, ensure_ascii=False, indent=2), encoding='utf-8')
