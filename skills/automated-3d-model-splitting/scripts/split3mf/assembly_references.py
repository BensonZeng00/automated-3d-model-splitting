from __future__ import annotations

from scipy.ndimage import convolve1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from .cap_template import fit_affine_cap_inside_parent, progressive_boundary_deformation, refined_harmonic_heightfield_cap
import copy
import os
import time
from .common import *
from .print_tolerance import current as current_print_tolerance
from .project import *
from .recognition import *
from .mesh import *
from .package_io import *
from .selection import *
from .assembly import *
from .validation import *
from .interface_retopology import InterfaceRetopologyService, generated_geometry_boundary_vertices
from .domain import PlanarArcRetopologyContext, CapDecision
from .hidden_interface import HIDDEN_INTERFACE_MINIMUM_LOAD_BEARING_DEPTH_MM, MAX_BOUNDARY_THICKNESS_PROBES, HiddenInterfacePlanner, boundary_screening_indices
from .guided_internal_cut import GuidedInternalCutPlanner, GuidedInternalCutSpec, adaptive_guided_entry_ring
from .local_connectors import LocalConnectorSpec, build_socket_cutter_from_plan, plan_local_connector, subtract_socket_cutters
from .connector_planning import local_connector_safe_depth_at_boundary, local_connector_safe_depth_from_field, local_connector_spec_for_interface
from .connector_geometry import backing_taper_angle_audit as _backing_taper_angle_audit, line_preserving_inset_displacements as _line_preserving_inset_displacements, printable_backing_profile as _printable_backing_profile, printable_backing_rings as _printable_backing_rings, project_connector_points as _project_connector_points, user_reviewed_shallow_minimal_needle_advisory_is_eligible
from .connector_topology import orient_face_patch_consistently, triangulate_bounded_ring_strip, triangulate_connector_annulus
from .connector_surface import refine_connector_annulus_heightfield
from .mesh_finalization import finalize_source_preserving_mesh
from .reporting import runtime_log

def build_component_cut_references(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    model_center: np.ndarray,
    inward_overrides: dict[int, np.ndarray] | None = None,
    fit_clearance_by_part: dict[int, float] | None = None,
    clearance_mode: str = "insert-shrink",
    recognized_boundaries=None,
) -> list[dict]:
    inward_overrides = inward_overrides or {}
    fit_clearance_by_part = fit_clearance_by_part or {}
    refs = []
    for index, component in enumerate(components, start=1):
        local_vertices, local_faces, global_to_local, global_vertex_ids = build_local_mesh(vertices, faces, component)
        if recognized_boundaries is None:
            loops = boundary_loops(local_faces)
        else:
            loops = [
                global_to_local[np.asarray(loop, dtype=np.int64)]
                for loop in recognized_boundaries.loops_for_component(index)
            ]
            if any(np.any(loop < 0) for loop in loops):
                raise ValueError(f"recognized boundary for P{index:02d} references an absent source vertex")
        component_center = local_vertices[local_faces.reshape(-1)].mean(axis=0)
        inward = inward_overrides.get(index)
        if inward is None:
            inward = -average_outward_normal(local_vertices, local_faces, component_center, model_center)
        inward = inward / max(float(np.linalg.norm(inward)), 1e-12)
        vertex_inward_normals = mesh_vertex_inward_normals(local_vertices, local_faces)
        vertex_conormal_evidence = mesh_vertex_conormal_evidence(
            local_vertices, local_faces
        )
        color_info = COLOR_INFO.get(component.color_code, {"name": component.color_code})
        for loop_index, loop in enumerate(loops):
            loop_global = [int(global_vertex_ids[i]) for i in loop]
            loop_conormals = boundary_loop_interior_conormals(
                local_vertices,
                local_faces,
                loop,
                reference_axis=inward,
                vertex_evidence=vertex_conormal_evidence,
            )
            loop_directions, direction_record = safe_boundary_inward_directions(
                vertex_inward_normals,
                loop,
                inward,
                loop_points=local_vertices[np.asarray(loop, dtype=np.int64)],
            )
            refs.append(
                {
                    "component_index": index,
                    "color_code": component.color_code,
                    "color_name": color_info["name"],
                    "loop_index": loop_index,
                    "global_vertices": set(loop_global),
                    "global_loop": loop_global,
                    "global_edges": {
                        tuple(sorted((loop_global[position], loop_global[(position + 1) % len(loop_global)])))
                        for position in range(len(loop_global))
                    },
                    "inward": inward,
                    "inward_by_global": {
                        int(global_id): direction.copy()
                        for global_id, direction in zip(loop_global, loop_directions)
                    },
                    "interior_conormal_by_global": {
                        int(global_id): conormal.copy()
                        for global_id, conormal in zip(loop_global, loop_conormals)
                    },
                    "local_inward_direction_record": direction_record,
                    "fit_clearance_mm": float(fit_clearance_by_part.get(index, 0.0)),
                    "socket_overcut_mm": float(
                        clearance_offsets(clearance_mode, fit_clearance_by_part.get(index, 0.0))[1]
                    ),
                }
            )
    return refs

def subtree_component_indices(root_index: int, children: dict[int, list[int]]) -> list[int]:
    indices = [int(root_index)]
    for child_index in children.get(int(root_index), []):
        indices.extend(subtree_component_indices(int(child_index), children))
    return sorted(indices)

def build_subassembly_component(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    indices: list[int],
    color_code: str,
) -> Component:
    global_faces = np.concatenate([components[int(index) - 1].global_faces for index in indices])
    group_faces = faces[global_faces]
    points = vertices[group_faces.reshape(-1)]
    # Computing every source-triangle area for each recursive subtree made
    # preparation quadratic in the number of children.  Only selected faces
    # contribute to this synthetic component.
    selected_areas = triangle_areas(vertices, group_faces)
    return Component(
        color_code=color_code,
        global_faces=global_faces,
        face_count=int(len(global_faces)),
        area=float(selected_areas.sum()),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        center=points.mean(axis=0),
    )

def boundary_loop_parent_contact(
    loop: list[int],
    global_vertex_ids: np.ndarray,
    boundary_neighbor_lookup: dict[tuple[int, int], set[int]],
    parent_index: int,
    subtree_indices: set[int],
) -> dict:
    neighbor_counts: collections.Counter[int] = collections.Counter()
    parent_edges = 0
    external_edges = 0
    for position, local_a in enumerate(loop):
        local_b = int(loop[(position + 1) % len(loop)])
        edge = tuple(sorted((int(global_vertex_ids[int(local_a)]), int(global_vertex_ids[local_b]))))
        neighbors = {int(value) for value in boundary_neighbor_lookup.get(edge, set())}
        for neighbor_index in neighbors:
            if neighbor_index not in subtree_indices:
                neighbor_counts[neighbor_index] += 1
        if int(parent_index) in neighbors:
            parent_edges += 1
        if any(neighbor_index not in subtree_indices for neighbor_index in neighbors):
            external_edges += 1
    return {
        "parent_edges": int(parent_edges),
        "external_edges": int(external_edges),
        "neighbor_counts": {str(key): int(value) for key, value in sorted(neighbor_counts.items())},
    }

def build_layer_child_cut_references(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    parent_index: int,
    direct_child_indices: list[int],
    assembly_children: dict[int, list[int]],
    model_center: np.ndarray,
    boundary_neighbor_lookup: dict[tuple[int, int], set[int]],
    inward_overrides: dict[int, np.ndarray],
    effective_cap_mode,
    guided_internal_cuts_by_part: dict[int, GuidedInternalCutSpec] | None = None,
    effective_planar_extra_limit=None,
    fit_clearance_by_part: dict[int, float] | None = None,
    boundary_reconciliation_tolerance_mm: float | None = None,
    clearance_mode: str = "insert-shrink",
    interface_retopology: PlanarArcRetopologyContext | None = None,
    max_extension_mm: float | None = None,
    flat_clearance_mm: float | None = None,
    bottom_clearance_mm: float = 0.0,
    lead_in_mm: float = 0.0,
    interface_geometry: str = "boundary-extrusion",
    recognized_boundaries=None,
) -> tuple[list[dict], dict[int, Component], dict[int, list[int]]]:
    context_started_at = time.perf_counter()
    from .layer_seam_planning import prepare_layer_seams, layer_child_boundary_topology
    vertices, faces, interface_retopology = prepare_layer_seams(
        vertices, faces, components, parent_index, direct_child_indices,
        assembly_children, boundary_neighbor_lookup, interface_retopology,
        recognized_boundaries=recognized_boundaries)
    if interface_retopology is not None and interface_retopology.active_layer_seam is not None:
        model_center = np.asarray(vertices).mean(axis=0)
    fit_clearance_by_part = fit_clearance_by_part or {}
    guided_internal_cuts_by_part = guided_internal_cuts_by_part or {}
    cap_planning_enabled = (
        interface_retopology is not None
        and max_extension_mm is not None
        and flat_clearance_mm is not None
    )
    face_owner_indices = np.zeros(len(faces), dtype=np.int32)
    for component_index, component in enumerate(components, start=1):
        component_faces = np.asarray(component.global_faces, dtype=np.int64)
        if len(component_faces):
            face_owner_indices[component_faces] = int(component_index)
    parent_thickness_probe = (
        ParentThicknessProbe(
            vertices,
            faces,
            triangle_owner_indices=face_owner_indices,
        )
        if cap_planning_enabled
        else None
    )
    runtime_log(
        "递归预计算",
        "layer_child_context_start",
        "开始预计算直属子件帽底和母槽上下文",
        parent_index=int(parent_index),
        direct_child_indices=sorted(int(index) for index in direct_child_indices),
        source_face_count=int(len(faces)),
        cap_planning_enabled=bool(cap_planning_enabled),
    )
    refs: list[dict] = []
    union_by_child: dict[int, Component] = {}
    subtree_by_child: dict[int, list[int]] = {}
    for child_index in sorted(int(index) for index in direct_child_indices):
        child_started_at = time.perf_counter()
        subtree = subtree_component_indices(child_index, assembly_children)
        runtime_log(
            "递归预计算",
            "layer_child_start",
            "开始预计算直属子件",
            parent_index=int(parent_index),
            child_index=int(child_index),
            subtree_indices=[int(index) for index in subtree],
        )
        subtree_by_child[child_index] = subtree
        union_component = build_subassembly_component(
            vertices,
            faces,
            components,
            subtree,
            components[child_index - 1].color_code,
        )
        union_by_child[child_index] = union_component
        child_parent_thickness_probe = parent_thickness_probe
        if parent_thickness_probe is not None:
            excluded_subtree_faces = np.unique(
                np.asarray(union_component.global_faces, dtype=np.int64)
            )
            child_parent_thickness_probe = (
                parent_thickness_probe.excluding_triangles(
                    excluded_subtree_faces,
                    filter_context={
                        "parent_part_index": int(parent_index),
                        "child_part_index": int(child_index),
                        "excluded_subtree_part_indices": [
                            int(index) for index in subtree
                        ],
                    },
                )
            )
            runtime_log(
                "递归预计算",
                "parent_thickness_probe_subtree_excluded",
                "父体厚度探针已排除当前子件整棵子树",
                parent_index=int(parent_index),
                child_index=int(child_index),
                subtree_indices=[int(index) for index in subtree],
                total_face_count=int(len(faces)),
                excluded_subtree_face_count=int(len(excluded_subtree_faces)),
                included_probe_face_count=int(
                    child_parent_thickness_probe.active_triangle_count
                ),
            )
        root_component = components[child_index - 1]
        guided_internal_cut = guided_internal_cuts_by_part.get(child_index)
        # A direct child may represent a recursive subassembly.  Its parent
        # socket must follow the *outer boundary of the complete subtree*, not
        # the boundary of the root color patch alone.  At a three-color
        # junction the root patch boundary contains an internal sibling arc;
        # using that arc as part of the parent socket selects different closed
        # loops on the two sides of the interface.
        local_faces, global_vertex_ids, loops = layer_child_boundary_topology(
            faces, components, subtree, interface_retopology,
            boundary_neighbor_lookup, recognized_boundaries,
        )
        local_vertices = np.asarray(vertices)[global_vertex_ids]
        runtime_log(
            "递归预计算",
            "layer_child_boundary_done",
            "直属子件局部网格和边界环已建立",
            parent_index=int(parent_index),
            child_index=int(child_index),
            local_face_count=int(len(local_faces)),
            boundary_loop_count=int(len(loops)),
            stage_elapsed_seconds=round(
                float(time.perf_counter() - child_started_at),
                3,
            ),
        )
        inward = inward_overrides.get(child_index)
        if inward is None:
            inward = component_inward_direction(vertices, faces, root_component, model_center)
        inward = inward / max(float(np.linalg.norm(inward)), 1e-12)
        vertex_inward_normals = mesh_vertex_inward_normals(local_vertices, local_faces)
        color_info = COLOR_INFO.get(union_component.color_code, {"name": union_component.color_code})
        current_layer_component_set = {int(index) for index in subtree}
        selected_loop_records: list[dict] = []
        for loop_index, loop in enumerate(loops):
            contact = boundary_loop_parent_contact(loop, global_vertex_ids, boundary_neighbor_lookup, parent_index, current_layer_component_set)
            if int(contact["parent_edges"]) < 3:
                continue
            selected_loop_records.append(
                {
                    "loop_index": int(loop_index),
                    "loop": [int(value) for value in loop],
                    "contact": contact,
                }
            )

        from .micro_interfaces import filter_micro_interface_loops
        selected_loop_records, _micro_loops = filter_micro_interface_loops(
            local_vertices, selected_loop_records, part_index=child_index)
        for record in selected_loop_records:
            loop = record["loop"]
            loop_global = [int(global_vertex_ids[i]) for i in loop]
            # A consolidated paint component can cover surfaces with opposite
            # orientations.  Resolve the sign independently at every source
            # rim before local normals, thickness probes, or connector geometry
            # are planned; the late audit remains a hard safety backstop.
            from .surface_direction import resolve_inward_axis
            interface_inward, surface_direction_record = resolve_inward_axis(
                local_vertices,
                local_faces,
                loop,
                inward,
            )
            loop_directions, direction_record = safe_boundary_inward_directions(
                vertex_inward_normals,
                loop,
                interface_inward,
                loop_points=local_vertices[np.asarray(loop, dtype=np.int64)],
            )
            record.update(
                {
                    "global_loop": loop_global,
                    "interface_inward": interface_inward,
                    "loop_directions": loop_directions,
                    "loop_local_inward_normals": np.asarray(
                        vertex_inward_normals[np.asarray(loop, dtype=np.int64)],
                        dtype=np.float64,
                    ).copy(),
                    # The taper axis is the already safety-corrected insertion
                    # field.  A raw surface normal can be radial around a
                    # rounded feature and collapse the axial chamfer into an
                    # almost straight wall.
                    "lead_in_directions": loop_directions.copy(),
                    "direction_record": {
                        **direction_record,
                        "surface_orientation": surface_direction_record,
                    },
                }
            )
        planned_vertices = local_vertices
        _retopology_records = []
        if cap_planning_enabled and selected_loop_records:
            retopology_started_at = time.perf_counter()
            visible_planned_vertices, _retopology_records = InterfaceRetopologyService.retopologize_local_loops(
                local_vertices,
                [record["loop"] for record in selected_loop_records],
                global_vertex_ids,
                interface_retopology,
                local_faces=local_faces,
            )
            # The source rim may intentionally stay immutable while a farther
            # fitted ring is realized by generated walls/caps.  Do not lose
            # that manufacturing target when preparing the hidden geometry.
            planned_vertices = generated_geometry_boundary_vertices(
                visible_planned_vertices,
                [record["loop"] for record in selected_loop_records],
                _retopology_records,
            )
            runtime_log(
                "递归预计算",
                "layer_child_retopology_done",
                "直属子件边界平顺预计算完成",
                parent_index=int(parent_index),
                child_index=int(child_index),
                selected_loop_count=int(len(selected_loop_records)),
                stage_elapsed_seconds=round(
                    float(time.perf_counter() - retopology_started_at),
                    3,
                ),
            )

        requested_cap_mode = str(effective_cap_mode(child_index))
        planar_extra_limit_mm = (
            None
            if effective_planar_extra_limit is None
            else float(effective_planar_extra_limit(child_index))
        )
        fit_clearance_mm = float(fit_clearance_by_part.get(child_index, 0.0))
        interface_reconciliation_tolerance_mm = max(
            float(boundary_reconciliation_tolerance_mm or 0.0),
            float(interface_retopology.config.maximum_safe_target_offset_mm)
            if interface_retopology is not None
            else 0.0,
            # A large absolute fit can still be a small, already validated
            # relative change on a large loop (P09 is representative).  The
            # visible source rim remains immutable; this allowance applies
            # only to correspondence with the generated hidden ring.
            max((float(record.get("requested_maximum_target_displacement_mm", 0.0))
                 for record in _retopology_records), default=0.0)
            if cap_planning_enabled and selected_loop_records else 0.0,
        )
        insert_shrink_mm, socket_overcut_mm = clearance_offsets(
            clearance_mode,
            fit_clearance_mm,
        )
        for selected in selected_loop_records:
            loop_started_at = time.perf_counter()
            loop = selected["loop"]
            loop_global = selected["global_loop"]
            loop_directions = selected["loop_directions"]
            lead_in_directions = selected["lead_in_directions"]
            interface_inward = selected["interface_inward"]
            active_inward = interface_inward.copy()
            loop_conormals = boundary_loop_interior_conormals(
                planned_vertices, local_faces, loop,
                reference_axis=interface_inward,
            )
            contact = selected["contact"]
            ref = {
                "component_index": int(child_index),
                "color_code": union_component.color_code,
                "color_name": color_info["name"],
                "loop_index": int(selected["loop_index"]),
                "global_vertices": set(loop_global),
                "global_loop": loop_global,
                "global_edges": {
                    tuple(
                        sorted(
                            (
                                loop_global[position],
                                loop_global[(position + 1) % len(loop_global)],
                            )
                        )
                    )
                    for position in range(len(loop_global))
                },
                "inward": interface_inward,
                "inward_by_global": {
                    int(global_id): direction.copy()
                    for global_id, direction in zip(loop_global, loop_directions)
                },
                "interior_conormal_by_global": {
                    int(global_id): conormal.copy()
                    for global_id, conormal in zip(loop_global, loop_conormals)
                },
                "local_inward_direction_record": selected["direction_record"],
                "requested_cap_mode": requested_cap_mode,
                "cap_mode": requested_cap_mode,
                "planar_extra_limit_mm": planar_extra_limit_mm,
                "stage_subtree_indices": sorted(int(index) for index in subtree),
                "contact_source_component_index": int(child_index),
                "parent_contact_edges": int(contact["parent_edges"]),
                "parent_contact_neighbor_counts": contact["neighbor_counts"],
                "fit_clearance_mm": fit_clearance_mm,
                "boundary_reconciliation_tolerance_mm": float(
                    interface_reconciliation_tolerance_mm
                ),
                "socket_overcut_mm": float(socket_overcut_mm),
            }
            if cap_planning_enabled:
                runtime_log(
                    "递归预计算",
                    "layer_child_cap_start",
                    "开始测量直属子件帽底安全深度",
                    parent_index=int(parent_index),
                    child_index=int(child_index),
                    loop_index=int(selected["loop_index"]),
                    boundary_vertex_count=int(len(loop)),
                )
                loop_array = np.asarray(loop, dtype=np.int64)
                source_points = planned_vertices[loop_array]

                def build_reserved_cap_decision(
                    candidate_points: np.ndarray,
                    candidate_conormals: np.ndarray,
                    candidate_fallback_inward: np.ndarray,
                    candidate_inward_directions: np.ndarray,
                ) -> CapDecision:
                    candidate_socket_points = offset_points_along_conormals(
                        candidate_points,
                        candidate_conormals,
                        -max(float(socket_overcut_mm), 0.0),
                    )
                    base_insert_shrink_mm, taper_profile_record = tapered_profile_inset_limit(
                        candidate_points,
                        fit_clearance_mm=max(float(insert_shrink_mm), 0.0),
                        maximum_taper_depth_mm=max(float(lead_in_mm), 0.0),
                        maximum_total_depth_mm=tapered_profile_total_depth_budget(
                            max_extension_mm,
                            planar_extra_limit_mm,
                        ),
                    )
                    requested_fit_clearance_mm = max(
                        float(insert_shrink_mm), 0.0
                    )
                    inset_candidates = tapered_profile_inset_candidates(
                        requested_fit_clearance_mm,
                        base_insert_shrink_mm,
                    )
                    rejected_insets: list[dict] = []
                    valid_candidates: list[dict] = []
                    last_thickness_error: ValueError | None = None
                    preferred_minimum_depth_mm = min(
                        max(float(max_extension_mm), 0.0),
                        DEFAULT_EFFECTIVE_MINIMUM_INWARD_DEPTH_MM,
                    )
                    cap_planar_extra_limit_mm = max(
                        float(planar_extra_limit_mm or 0.0)
                        - max(float(bottom_clearance_mm), 0.0),
                        0.0,
                    )
                    early_stop_applied = False
                    for inset_candidate in inset_candidates:
                        effective_insert_shrink_mm = float(
                            inset_candidate["effective_insert_shrink_mm"]
                        )
                        candidate_fit_points = offset_points_along_conormals(
                            candidate_points,
                            candidate_conormals,
                            effective_insert_shrink_mm,
                        )
                        try:
                            candidate_decision = plan_cap_decision(
                                points=candidate_fit_points,
                                source_vertex_ids=loop_global,
                                fallback_inward=candidate_fallback_inward,
                                inward_directions=candidate_inward_directions,
                                fixed_depth_mm=float(max_extension_mm),
                                flat_clearance_mm=float(flat_clearance_mm),
                                cap_mode=requested_cap_mode,
                                planar_extra_limit_mm=cap_planar_extra_limit_mm,
                                parent_thickness_probe=child_parent_thickness_probe,
                                source_points=source_points,
                                # Keep the full ray horizon even for sparse
                                # boundary screening so entry/exit classification
                                # remains consistent with the final audit.
                            )
                        except ValueError as exc:
                            if "No positive inward depth remains" not in str(exc):
                                raise
                            last_thickness_error = exc
                            rejected_insets.append(
                                {
                                    **inset_candidate,
                                    "effective_insert_shrink_mm": float(
                                        effective_insert_shrink_mm
                                    ),
                                    "reason": "no_positive_parent_thickness_depth",
                                }
                            )
                            continue
                        reserved_distances, reserved_record = reserve_flat_socket_travel_budget(
                            child_fit_points=candidate_fit_points,
                            child_distances=candidate_decision.distances,
                            child_directions=candidate_decision.directions,
                            socket_top_points=candidate_socket_points,
                            bottom_clearance_mm=bottom_clearance_mm,
                            maximum_socket_travel_mm=float(max_extension_mm)
                            + max(float(planar_extra_limit_mm or 0.0), 0.0),
                            plane_record=candidate_decision.record,
                        )
                        valid_candidates.append(
                            {
                                **inset_candidate,
                                "effective_insert_shrink_mm": float(
                                    effective_insert_shrink_mm
                                ),
                                "minimum_depth_mm": float(
                                    np.min(reserved_distances)
                                ),
                                "decision": candidate_decision,
                                "reserved_distances": np.asarray(
                                    reserved_distances, dtype=np.float64
                                ).copy(),
                                "reserved_record": dict(reserved_record),
                            }
                        )
                        # Candidates are generated in descending lateral-inset
                        # order.  The selection contract asks for the largest
                        # inset which preserves the preferred minimum depth, so
                        # the first candidate meeting that gate is already the
                        # mathematically final answer.  Continuing would repeat
                        # the same two thickness fields for every smaller inset;
                        # on dense vendor loops that turns seconds into an hour.
                        if (
                            float(np.min(reserved_distances))
                            >= preferred_minimum_depth_mm - 1e-9
                        ):
                            early_stop_applied = True
                            break
                    if valid_candidates:
                        selected, selection_policy = select_taper_profile_candidate(
                            valid_candidates,
                            preferred_minimum_depth_mm,
                        )
                        effective_insert_shrink_mm = float(
                            selected["effective_insert_shrink_mm"]
                        )
                        # Candidate planning retains the complete cap geometry;
                        # ParentThicknessProbe limits every ray query to
                        # equal-arc boundary samples. Reuse this sampled result
                        # instead of repeating the same ray query.
                        candidate_decision = selected["decision"]
                        sampled_reserved_distances = np.asarray(
                            selected["reserved_distances"], dtype=np.float64
                        ).copy()
                        reserved_record = dict(selected["reserved_record"])
                        screening_minimum_depth_mm = float(
                            selected["minimum_depth_mm"]
                        )
                        sampled_minimum_depth_mm = float(
                            np.min(sampled_reserved_distances)
                        )
                        candidate_evaluations = [
                            {
                                "effective_insert_shrink_mm": float(
                                    item["effective_insert_shrink_mm"]
                                ),
                                "profile_factor": item.get("profile_factor"),
                                "candidate_source": item.get("candidate_source"),
                                "minimum_depth_mm": float(item["minimum_depth_mm"]),
                            }
                            for item in valid_candidates
                        ]
                        reserved_record.update(taper_profile_record)
                        reserved_record.update(
                            {
                                "thin_boundary_adaptive_inset_applied": bool(
                                    effective_insert_shrink_mm
                                    > requested_fit_clearance_mm + 1e-9
                                ),
                                "thin_boundary_base_insert_shrink_mm": float(
                                    base_insert_shrink_mm
                                ),
                                "thin_boundary_additional_inset_mm": float(
                                    max(
                                        effective_insert_shrink_mm
                                        - requested_fit_clearance_mm,
                                        0.0,
                                    )
                                ),
                                "thin_boundary_effective_insert_shrink_mm": float(
                                    effective_insert_shrink_mm
                                ),
                                "thin_boundary_rejected_inset_attempts": rejected_insets,
                                "thin_boundary_inset_policy": (
                                    "smooth_45deg_shoulder_depth_first_profile_backoff"
                                ),
                                "requested_fit_clearance_shrink_mm": float(
                                    requested_fit_clearance_mm
                                ),
                                "selected_lateral_inset_mm": float(
                                    effective_insert_shrink_mm
                                ),
                                "taper_profile_selected_factor": selected.get(
                                    "profile_factor"
                                ),
                                "taper_profile_candidate_source": selected.get(
                                    "candidate_source"
                                ),
                                "taper_profile_selection_policy": selection_policy,
                                "taper_profile_preferred_minimum_depth_mm": float(
                                    preferred_minimum_depth_mm
                                ),
                                "taper_profile_candidate_evaluations": candidate_evaluations,
                                "taper_profile_early_stop_applied": bool(
                                    early_stop_applied
                                ),
                                "taper_profile_candidates_evaluated": int(
                                    len(valid_candidates) + len(rejected_insets)
                                ),
                                "taper_profile_candidates_available": int(
                                    len(inset_candidates)
                                ),
                                "taper_profile_screening_ceiling_mm": float(
                                    preferred_minimum_depth_mm
                                ),
                                "taper_profile_screening_minimum_depth_mm": (
                                    screening_minimum_depth_mm
                                ),
                                "taper_profile_sampled_minimum_depth_mm": (
                                    sampled_minimum_depth_mm
                                ),
                            }
                        )
                        return CapDecision(
                            mode=str(reserved_record["cap_mode"]),
                            source_vertex_ids=candidate_decision.source_vertex_ids,
                            fit_points=np.asarray(
                                candidate_decision.fit_points,
                                dtype=np.float64,
                            ).copy(),
                            directions=np.asarray(
                                candidate_decision.directions,
                                dtype=np.float64,
                            ).copy(),
                            distances=np.asarray(
                                sampled_reserved_distances,
                                dtype=np.float64,
                            ).copy(),
                            record=reserved_record,
                            source_points=np.asarray(
                                candidate_decision.source_points,
                                dtype=np.float64,
                            ).copy(),
                        )
                    if last_thickness_error is not None:
                        raise ValueError(
                            "No positive inward depth remains after adaptive internal-ring "
                            "taper-profile backoff attempts: "
                            + str(last_thickness_error)
                        ) from last_thickness_error
                    raise RuntimeError("adaptive cap inset search produced no decision")

                if guided_internal_cut is not None:
                    # A section guide is already the authoritative cap plan.
                    # Do not spend (or distort) it through the ordinary adaptive
                    # plane search first.  Recess only the locally thin entry
                    # arc; a uniform large offset can collapse a concave loop.
                    if child_parent_thickness_probe is None:
                        raise ValueError(
                            "guided internal cut requires parent-thickness planning"
                        )
                    (
                        guided_fit_points,
                        guided_entry_insets,
                        guided_entry_record,
                    ) = adaptive_guided_entry_ring(
                        boundary_points=planned_vertices[loop_array],
                        interior_conormals=loop_conormals,
                        parent_thickness_probe=child_parent_thickness_probe,
                        spec=guided_internal_cut,
                        base_inset_mm=float(insert_shrink_mm),
                    )
                    cap_decision = CapDecision(
                        mode="flat",
                        source_vertex_ids=tuple(int(value) for value in loop_global),
                        fit_points=guided_fit_points,
                        directions=np.asarray(loop_directions, dtype=np.float64).copy(),
                        distances=np.full(
                            len(loop_global),
                            guided_internal_cut.minimum_depth_mm,
                            dtype=np.float64,
                        ),
                        record={
                            "cap_mode": "flat",
                            "requested_cap_mode": requested_cap_mode,
                            "thin_boundary_effective_insert_shrink_mm": float(
                                guided_entry_insets.max(
                                    initial=max(float(insert_shrink_mm), 0.0)
                                )
                            ),
                            "guided_internal_cut_entry_inset_limit_mm": max(
                                float(insert_shrink_mm),
                                float(guided_internal_cut.entry_inset_mm),
                                0.0,
                            ),
                            "guided_internal_cut_direct_planning": True,
                            **guided_entry_record,
                        },
                        source_points=np.asarray(source_points, dtype=np.float64).copy(),
                    )
                else:
                    cap_decision = build_reserved_cap_decision(
                        planned_vertices[loop_array],
                        loop_conormals,
                        active_inward,
                        loop_directions,
                    )

                if guided_internal_cut is not None:
                    guided_plan = GuidedInternalCutPlanner.plan(
                        points=np.asarray(cap_decision.fit_points, dtype=np.float64),
                        local_inward_normals=selected[
                            "loop_local_inward_normals"
                        ],
                        parent_thickness_probe=child_parent_thickness_probe,
                        spec=guided_internal_cut,
                        visible_top_points=planned_vertices[loop_array],
                    )
                    guided_record = dict(cap_decision.record)
                    guided_record.update(guided_plan.record)
                    guided_record.update(
                        {
                            "cap_mode": "flat",
                            "requested_cap_mode": requested_cap_mode,
                            "flat_orientation": "visual_section_guide",
                            "cap_depth_reference": (
                                "user_marked_section_plane_with_3d_lateral_search"
                            ),
                            "inward_depth_policy": (
                                "exact_guided_plane_with_per_vertex_parent_clearance"
                            ),
                            "effective_minimum_inward_depth_mm": float(
                                np.min(guided_plan.distances)
                            ),
                            "safe_maximum_inward_depth_mm": float(
                                np.max(guided_plan.distances)
                                + guided_plan.record[
                                    "minimum_thickness_margin_mm"
                                ]
                            ),
                            "minimum_generated_inward_travel_mm": float(
                                np.min(guided_plan.distances)
                            ),
                            "maximum_generated_inward_travel_mm": float(
                                np.max(guided_plan.distances)
                            ),
                            "flat_bottom_plane_s": float(
                                np.dot(
                                    guided_plan.effective_plane_point_mm,
                                    guided_internal_cut.target_plane_normal,
                                )
                            ),
                            "flat_bottom_required_max_extension_mm": float(
                                np.max(guided_plan.distances)
                            ),
                            "flat_plane_extension_max_mm": float(
                                np.max(guided_plan.distances)
                            ),
                            "flat_priority_applied": True,
                            "flat_feasibility_basis": (
                                "guided_plane_with_per_vertex_parent_thickness"
                            ),
                            "flat_travel_within_limit": True,
                            "guided_internal_cut_applied": True,
                            "guided_internal_cut_visible_boundary_locked": True,
                        }
                    )
                    cap_decision = CapDecision(
                        mode="flat",
                        source_vertex_ids=cap_decision.source_vertex_ids,
                        fit_points=np.asarray(
                            cap_decision.fit_points, dtype=np.float64
                        ).copy(),
                        directions=guided_plan.directions.copy(),
                        distances=guided_plan.distances.copy(),
                        record=guided_record,
                        source_points=np.asarray(
                            cap_decision.source_points, dtype=np.float64
                        ).copy(),
                    )
                    active_inward = guided_internal_cut.entry_direction.copy()
                    loop_directions = guided_plan.directions.copy()
                    lead_in_directions = guided_plan.directions.copy()
                    ref["inward"] = active_inward.copy()
                    ref["inward_by_global"] = {
                        int(global_id): direction.copy()
                        for global_id, direction in zip(
                            loop_global, guided_plan.directions
                        )
                    }
                    ref["local_inward_direction_record"] = {
                        **dict(ref["local_inward_direction_record"]),
                        "direction_mode": "guided_internal_section_field",
                        "guided_internal_cut_selected_candidate": (
                            guided_plan.record[
                                "guided_internal_cut_selected_candidate"
                            ]
                        ),
                    }
                    runtime_log(
                        "引导切面",
                        "guided_internal_cut_selected",
                        "已按标注折线生成内部共享切面方向场",
                        parent_index=int(parent_index),
                        child_index=int(child_index),
                        loop_index=int(selected["loop_index"]),
                        selected_candidate=guided_plan.record[
                            "guided_internal_cut_selected_candidate"
                        ],
                        minimum_depth_mm=float(np.min(guided_plan.distances)),
                        maximum_depth_mm=float(np.max(guided_plan.distances)),
                    )

                baseline_safe_depth_mm = float(
                    cap_decision.record.get(
                        "safe_maximum_inward_depth_mm",
                        np.min(cap_decision.distances),
                    )
                )
                if (
                    child_parent_thickness_probe is not None
                    and guided_internal_cut is None
                    and baseline_safe_depth_mm
                    < HIDDEN_INTERFACE_MINIMUM_LOAD_BEARING_DEPTH_MM - 1e-9
                ):
                    hidden_plan = HiddenInterfacePlanner.plan(
                        points=planned_vertices[loop_array],
                        initial_directions=loop_directions,
                        fallback_axis=active_inward,
                        local_inward_normals=selected[
                            "loop_local_inward_normals"
                        ],
                        parent_thickness_probe=child_parent_thickness_probe,
                        baseline_safe_depth_mm=baseline_safe_depth_mm,
                        preferred_depth_mm=float(max_extension_mm),
                    )
                    hidden_attempts: list[dict] = []
                    selected_hidden = None
                    selected_hidden_decision = cap_decision
                    selected_hidden_minimum = float(
                        np.min(cap_decision.distances)
                    )
                    for hidden_candidate in hidden_plan.candidates:
                        try:
                            hidden_decision = build_reserved_cap_decision(
                                planned_vertices[loop_array],
                                loop_conormals,
                                hidden_candidate.axis,
                                hidden_candidate.directions,
                            )
                        except (ValueError, RuntimeError) as exc:
                            hidden_attempts.append(
                                {
                                    "name": hidden_candidate.name,
                                    "screening_depth_mm": (
                                        hidden_candidate.screening_depth_mm
                                    ),
                                    "authoritative_safe_depth_mm": None,
                                    "accepted": False,
                                    "reason": str(exc),
                                }
                            )
                            continue
                        hidden_minimum = float(
                            np.min(hidden_decision.distances)
                        )
                        improved = bool(
                            hidden_minimum
                            > selected_hidden_minimum + 1e-6
                        )
                        hidden_attempts.append(
                            {
                                "name": hidden_candidate.name,
                                "screening_depth_mm": (
                                    hidden_candidate.screening_depth_mm
                                ),
                                "authoritative_safe_depth_mm": float(
                                    hidden_decision.record.get(
                                        "safe_maximum_inward_depth_mm",
                                        hidden_minimum,
                                    )
                                ),
                                "generated_minimum_depth_mm": hidden_minimum,
                                "accepted": improved,
                            }
                        )
                        if not improved:
                            continue
                        selected_hidden = hidden_candidate
                        selected_hidden_decision = hidden_decision
                        selected_hidden_minimum = hidden_minimum
                        if (
                            hidden_minimum
                            >= HIDDEN_INTERFACE_MINIMUM_LOAD_BEARING_DEPTH_MM
                            - 1e-9
                        ):
                            break

                    hidden_record = dict(selected_hidden_decision.record)
                    hidden_record.update(hidden_plan.record)
                    hidden_record["hidden_interface_attempts"] = hidden_attempts
                    hidden_record["hidden_interface_applied"] = bool(
                        selected_hidden is not None
                    )
                    if selected_hidden is not None:
                        hidden_record.update(
                            {
                                "hidden_interface_selected_candidate": (
                                    selected_hidden.name
                                ),
                                "hidden_interface_selected_axis": (
                                    selected_hidden.axis.round(6).tolist()
                                ),
                                "hidden_interface_selected_safe_depth_mm": (
                                    selected_hidden_decision.record.get(
                                        "safe_maximum_inward_depth_mm",
                                        selected_hidden_minimum,
                                    )
                                ),
                                "hidden_interface_selected_screening_depth_mm": (
                                    selected_hidden.screening_depth_mm
                                ),
                                "hidden_interface_generated_minimum_depth_mm": (
                                    selected_hidden_minimum
                                ),
                                "hidden_interface_visible_boundary_locked": True,
                                "hidden_interface_load_bearing_zone": (
                                    "complete_hidden_cap_after_interior_direction_search"
                                ),
                            }
                        )
                        active_inward = np.asarray(
                            selected_hidden.axis,
                            dtype=np.float64,
                        ).copy()
                        active_inward /= max(
                            float(np.linalg.norm(active_inward)), 1e-12
                        )
                        loop_directions = np.asarray(
                            selected_hidden.directions,
                            dtype=np.float64,
                        ).copy()
                        lead_in_directions = loop_directions.copy()
                        ref["inward"] = active_inward.copy()
                        ref["inward_by_global"] = {
                            int(global_id): direction.copy()
                            for global_id, direction in zip(
                                loop_global,
                                loop_directions,
                            )
                        }
                        ref["local_inward_direction_record"] = {
                            **dict(ref["local_inward_direction_record"]),
                            "direction_mode": (
                                "hidden_interface_parent_interior_search"
                            ),
                            "hidden_interface_selected_candidate": (
                                selected_hidden.name
                            ),
                        }
                        runtime_log(
                            "隐藏切面",
                            "hidden_interface_direction_selected",
                            "薄边限制已由内部承力方向搜索解除",
                            parent_index=int(parent_index),
                            child_index=int(child_index),
                            loop_index=int(selected["loop_index"]),
                            baseline_safe_depth_mm=baseline_safe_depth_mm,
                            selected_candidate=selected_hidden.name,
                            selected_safe_depth_mm=float(
                                selected_hidden_decision.record.get(
                                    "safe_maximum_inward_depth_mm",
                                    selected_hidden_minimum,
                                )
                            ),
                            selected_screening_depth_mm=float(
                                selected_hidden.screening_depth_mm
                            ),
                            generated_minimum_depth_mm=float(
                                selected_hidden_minimum
                            ),
                        )
                    cap_decision = CapDecision(
                        mode=selected_hidden_decision.mode,
                        source_vertex_ids=(
                            selected_hidden_decision.source_vertex_ids
                        ),
                        fit_points=selected_hidden_decision.fit_points,
                        directions=selected_hidden_decision.directions,
                        distances=selected_hidden_decision.distances,
                        record=hidden_record,
                        source_points=selected_hidden_decision.source_points,
                    )

                def attach_harmonic_cap_template(
                    decision: CapDecision,
                ) -> CapDecision:
                    target_bottom = (
                        np.asarray(decision.fit_points, dtype=np.float64)
                        + np.asarray(decision.directions, dtype=np.float64)
                        * np.asarray(decision.distances, dtype=np.float64)[:, None]
                    )
                    boundary_only_template = bool(
                        len(loops) != 1 or len(selected_loop_records) != 1
                    )
                    if str(decision.mode) == "flat":
                        try:
                            flat_triangles, flat_normal, _flat_record = (
                                triangulate_ordered_loop_3d(
                                    target_bottom,
                                    np.asarray(active_inward, dtype=np.float64),
                                )
                            )
                            flat_quality = inward_cap_surface_quality(
                                target_bottom,
                                flat_triangles,
                                flat_normal,
                            )
                        except ValueError:
                            flat_quality = {"valid": False}
                        if bool(flat_quality["valid"]):
                            return decision
                        boundary_only_template = True
                    if boundary_only_template:
                        try:
                            direct_triangles, direct_normal, triangulation = (
                                triangulate_ordered_loop_3d(
                                    target_bottom,
                                    np.asarray(active_inward, dtype=np.float64),
                                )
                            )
                            (
                                heightfield_points,
                                heightfield_faces,
                                heightfield_quality,
                            ) = refined_harmonic_heightfield_cap(
                                boundary_points=target_bottom,
                                initial_faces=direct_triangles,
                                reference_normal=direct_normal,
                            )
                        except ValueError as error:
                            runtime_log(
                                "cap-template",
                                "multi_opening_heightfield_cap_failed",
                                "Multi-opening local heightfield cap failed",
                                error=str(error),
                            )
                            return decision
                        runtime_log(
                            "cap-template",
                            "multi_opening_heightfield_cap_audited",
                            "Multi-opening local heightfield cap audit completed",
                            **heightfield_quality,
                        )
                        if not bool(heightfield_quality["valid"]):
                            return decision
                        heightfield_record = dict(decision.record)
                        heightfield_record.update(
                            {
                                "cap_mode": "source-patch-template",
                                "cap_surface_strategy": (
                                    "refined_harmonic_heightfield"
                                ),
                                "heightfield_cap_quality": heightfield_quality,
                                "heightfield_initial_triangulation": (
                                    triangulation
                                ),
                                "multi_opening_local_template": True,
                            }
                        )
                        return CapDecision(
                            mode="source-patch-template",
                            source_vertex_ids=decision.source_vertex_ids,
                            fit_points=decision.fit_points,
                            directions=decision.directions,
                            distances=decision.distances,
                            record=heightfield_record,
                            source_points=decision.source_points,
                            cap_template_points=heightfield_points,
                            cap_template_faces=heightfield_faces,
                            cap_template_boundary_ids=tuple(
                                range(len(target_bottom))
                            ),
                        )
                    template_points, template_quality = harmonic_patch_deformation(
                        local_vertices,
                        local_faces,
                        loop,
                        target_bottom,
                    )
                    if not bool(template_quality["valid"]):
                        rejected_harmonic_quality = template_quality
                        direct_cap_record: dict = {
                            "valid": False,
                            "strategy": "exact_boundary_triangulation",
                        }
                        direct_triangles = None
                        direct_normal = None
                        try:
                            direct_triangles, direct_normal, triangulation = (
                                triangulate_ordered_loop_3d(
                                    target_bottom,
                                    np.asarray(active_inward, dtype=np.float64),
                                )
                            )
                            direct_surface_quality = inward_cap_surface_quality(
                                target_bottom,
                                direct_triangles,
                                direct_normal,
                            )
                            direct_cap_record = {
                                "valid": bool(
                                    len(direct_triangles)
                                    == max(len(target_bottom) - 2, 0)
                                    and direct_surface_quality["valid"]
                                ),
                                "strategy": "exact_boundary_triangulation",
                                "face_count": int(len(direct_triangles)),
                                "expected_face_count": int(
                                    max(len(target_bottom) - 2, 0)
                                ),
                                "triangulation": triangulation,
                                "surface_quality": direct_surface_quality,
                            }
                        except ValueError as error:
                            direct_cap_record["error"] = str(error)
                        runtime_log(
                            "cap-template",
                            "exact_boundary_cap_fallback_audited",
                            "Exact-boundary cap fallback audit completed",
                            valid=bool(direct_cap_record["valid"]),
                            face_count=direct_cap_record.get("face_count"),
                            expected_face_count=direct_cap_record.get(
                                "expected_face_count"
                            ),
                            error=direct_cap_record.get("error"),
                            surface_quality=direct_cap_record.get(
                                "surface_quality"
                            ),
                        )
                        if bool(direct_cap_record["valid"]):
                            direct_record = dict(decision.record)
                            direct_record.update(
                                {
                                    "cap_surface_strategy": (
                                        "exact_boundary_triangulation_fallback"
                                    ),
                                    "exact_boundary_cap_quality": direct_cap_record,
                                    "rejected_harmonic_quality": (
                                        rejected_harmonic_quality
                                    ),
                                }
                            )
                            return CapDecision(
                                mode=decision.mode,
                                source_vertex_ids=decision.source_vertex_ids,
                                fit_points=decision.fit_points,
                                directions=decision.directions,
                                distances=decision.distances,
                                record=direct_record,
                                source_points=decision.source_points,
                            )
                        if (
                            direct_triangles is not None
                            and direct_normal is not None
                        ):
                            (
                                heightfield_points,
                                heightfield_faces,
                                heightfield_quality,
                            ) = refined_harmonic_heightfield_cap(
                                boundary_points=target_bottom,
                                initial_faces=direct_triangles,
                                reference_normal=direct_normal,
                            )
                            runtime_log(
                                "cap-template",
                                "refined_heightfield_cap_audited",
                                "Refined harmonic heightfield cap audit completed",
                                **heightfield_quality,
                            )
                            if bool(heightfield_quality["valid"]):
                                heightfield_record = dict(decision.record)
                                heightfield_record.update(
                                    {
                                        "cap_mode": "source-patch-template",
                                        "cap_surface_strategy": (
                                            "refined_harmonic_heightfield"
                                        ),
                                        "heightfield_cap_quality": (
                                            heightfield_quality
                                        ),
                                        "rejected_harmonic_quality": (
                                            rejected_harmonic_quality
                                        ),
                                        "rejected_exact_boundary_cap_quality": (
                                            direct_cap_record
                                        ),
                                    }
                                )
                                return CapDecision(
                                    mode="source-patch-template",
                                    source_vertex_ids=(
                                        decision.source_vertex_ids
                                    ),
                                    fit_points=decision.fit_points,
                                    directions=decision.directions,
                                    distances=decision.distances,
                                    record=heightfield_record,
                                    source_points=decision.source_points,
                                    cap_template_points=heightfield_points,
                                    cap_template_faces=heightfield_faces,
                                    cap_template_boundary_ids=tuple(
                                        range(len(target_bottom))
                                    ),
                                )
                        template_points, template_quality = (
                            progressive_boundary_deformation(
                                source_points=local_vertices,
                                source_faces=local_faces,
                                boundary_indices=loop_array,
                                target_boundary_points=target_bottom,
                                deform=harmonic_patch_deformation,
                            )
                        )
                        if bool(template_quality["valid"]):
                            template_quality["rejected_one_step_quality"] = (
                                rejected_harmonic_quality
                            )
                    if not bool(template_quality["valid"]):
                        rejected_progressive_quality = template_quality
                        affine_matrix, affine_fit = fit_orientation_preserving_affine(
                            local_vertices[loop_array],
                            target_bottom,
                        )
                        template_points = np.column_stack(
                            (
                                local_vertices,
                                np.ones(len(local_vertices), dtype=np.float64),
                            )
                        ) @ affine_matrix
                        determinant = float(np.linalg.det(affine_matrix[:3, :]))
                        if determinant <= 1e-9:
                            raise ValueError(
                                "source-patch affine cap transform is reflected or singular: "
                                f"determinant={determinant:.9f}"
                            )
                        affine_boundary = template_points[loop_array]
                        target_deviation = np.linalg.norm(
                            affine_boundary - target_bottom,
                            axis=1,
                        )
                        fit_points = np.asarray(
                            decision.fit_points,
                            dtype=np.float64,
                        )
                        axis = np.asarray(decision.directions, dtype=np.float64).mean(
                            axis=0
                        )
                        axis /= max(float(np.linalg.norm(axis)), 1e-12)
                        try:
                            affine_backoff = fit_affine_cap_inside_parent(
                                template_points=template_points,
                                boundary_indices=loop_array,
                                fit_points=fit_points,
                                inward_axis=axis,
                                safety_limit=(
                                    child_parent_thickness_probe.safety_limit
                                ),
                                maximum_depth_mm=MAXIMUM_SAFE_INWARD_DEPTH_MM,
                            )
                        except ValueError as error:
                            harmonic_summary = {
                                key: rejected_harmonic_quality.get(key)
                                for key in (
                                    "degenerate_face_count",
                                    "reversed_face_count",
                                    "minimum_source_normal_cosine",
                                    "maximum_edge_stretch_ratio",
                                    "maximum_affine_baseline_edge_mm",
                                    "maximum_triangle_edge_mm",
                                    "long_edge_valid",
                                )
                            }
                            last_progressive_quality = (
                                rejected_progressive_quality.get(
                                    "last_rejected_quality", {}
                                )
                            )
                            progressive_summary = {
                                "progress_fraction": (
                                    rejected_progressive_quality.get(
                                        "progress_fraction"
                                    )
                                ),
                                "accepted_step_count": (
                                    rejected_progressive_quality.get(
                                        "accepted_step_count"
                                    )
                                ),
                                "rejected_step_count": (
                                    rejected_progressive_quality.get(
                                        "rejected_step_count"
                                    )
                                ),
                                "failure_reason": (
                                    rejected_progressive_quality.get(
                                        "failure_reason"
                                    )
                                ),
                                "last_reversed_face_count": (
                                    last_progressive_quality.get(
                                        "reversed_face_count"
                                    )
                                ),
                                "last_minimum_normal_cosine": (
                                    last_progressive_quality.get(
                                        "minimum_source_normal_cosine"
                                    )
                                ),
                            }
                            direct_surface_summary = direct_cap_record.get(
                                "surface_quality", {}
                            )
                            direct_summary = {
                                "face_count": direct_cap_record.get("face_count"),
                                "expected_face_count": direct_cap_record.get(
                                    "expected_face_count"
                                ),
                                "error": direct_cap_record.get("error"),
                                "maximum_planarity_error_mm": (
                                    direct_surface_summary.get(
                                        "maximum_planarity_error_mm"
                                    )
                                ),
                                "minimum_normal_cosine": (
                                    direct_surface_summary.get(
                                        "minimum_normal_cosine"
                                    )
                                ),
                                "inconsistent_normal_faces": (
                                    direct_surface_summary.get(
                                        "inconsistent_normal_faces"
                                    )
                                ),
                                "long_folded_faces": direct_surface_summary.get(
                                    "long_folded_faces"
                                ),
                            }
                            raise ValueError(
                                f"{error}; rejected_harmonic_quality="
                                f"{harmonic_summary}; "
                                f"rejected_direct_cap_quality={direct_summary}; "
                                f"rejected_progressive_quality="
                                f"{progressive_summary}"
                            ) from error
                        template_points = affine_backoff.template_points
                        affine_boundary = affine_backoff.boundary_points
                        affine_distances = affine_backoff.distances
                        affine_directions = affine_backoff.directions
                        thickness_record = affine_backoff.thickness_record
                        template_quality = {
                            "valid": True,
                            "strategy": "source_patch_affine_fallback",
                            "face_count": int(len(local_faces)),
                            "vertex_count": int(len(local_vertices)),
                            "boundary_vertex_count": int(len(loop)),
                            "affine_determinant": determinant,
                            "affine_fit": affine_fit,
                            "target_boundary_deviation_max_mm": float(
                                target_deviation.max()
                            ),
                            "target_boundary_deviation_mean_mm": float(
                                target_deviation.mean()
                            ),
                            "minimum_depth_mm": float(affine_distances.min()),
                            "maximum_depth_mm": float(affine_distances.max()),
                            "affine_backoff_attempts": int(
                                affine_backoff.attempts
                            ),
                            "affine_total_backoff_mm": float(
                                affine_backoff.total_backoff_mm
                            ),
                            "affine_backoff_trace": list(affine_backoff.trace),
                            **thickness_record,
                            "rejected_harmonic_quality": (
                                rejected_harmonic_quality
                            ),
                            "rejected_progressive_quality": (
                                rejected_progressive_quality
                            ),
                            "rejected_exact_boundary_cap_quality": (
                                direct_cap_record
                            ),
                        }
                        decision = CapDecision(
                            mode=decision.mode,
                            source_vertex_ids=decision.source_vertex_ids,
                            fit_points=decision.fit_points,
                            directions=affine_directions,
                            distances=affine_distances,
                            record=dict(decision.record),
                            source_points=decision.source_points,
                        )
                    template_record = dict(decision.record)
                    template_record.update(
                        {
                            "cap_mode": "source-patch-template",
                            "cap_surface_strategy": template_quality["strategy"],
                            "harmonic_cap_quality": template_quality,
                        }
                    )
                    return CapDecision(
                        mode="source-patch-template",
                        source_vertex_ids=decision.source_vertex_ids,
                        fit_points=decision.fit_points,
                        directions=decision.directions,
                        distances=decision.distances,
                        record=template_record,
                        source_points=decision.source_points,
                        cap_template_points=template_points,
                        cap_template_faces=np.asarray(
                            local_faces,
                            dtype=np.int64,
                        ).copy(),
                        cap_template_boundary_ids=tuple(
                            int(value) for value in loop
                        ),
                    )

                local_connector_safety: dict | None = None
                if interface_geometry == "local-connector":
                    # The compact connector has its own geometry-aware bridge from
                    # the immutable source rim to a simplified backing ring.  The
                    # equal-count boundary-extrusion proxy is not part of that
                    # output and can falsely report folded quads on concave rims.
                    # Preflight the exact production geometry instead.
                    local_connector_safety = local_connector_safe_depth_at_boundary(
                        boundary_points=np.asarray(
                            cap_decision.source_points,
                            dtype=np.float64,
                        ),
                        inward=active_inward,
                        fit_clearance_mm=fit_clearance_mm,
                        bottom_clearance_mm=bottom_clearance_mm,
                        lead_in_mm=lead_in_mm,
                        boundary_distances=np.asarray(
                            cap_decision.distances,
                            dtype=np.float64,
                        ),
                        parent_thickness_probe=child_parent_thickness_probe,
                    )
                    connector_spec = local_connector_spec_for_interface(
                        fit_clearance_mm=fit_clearance_mm,
                        bottom_clearance_mm=bottom_clearance_mm,
                        lead_in_mm=lead_in_mm,
                        safe_engagement_depth_mm=float(
                            local_connector_safety[
                                "local_connector_safety_budget_mm"
                            ]
                        ),
                        safe_backing_depth_mm=float(
                            local_connector_safety.get(
                                "local_connector_backing_safety_limit_mm",
                                local_connector_safety[
                                    "boundary_safety_minimum_mm"
                                ],
                            )
                        ),
                        compact_peg_supported=bool(
                            local_connector_safety.get(
                                "local_connector_compact_peg_supported",
                                True,
                            )
                        ),
                        slope_validation_mode=str(
                            interface_retopology.config.connector_slope_validation
                        ),
                        surface_validation_mode=str(
                            interface_retopology.config.connector_surface_validation
                        ),
                    )
                    initial_cap_quality = preflight_local_connector_patch(
                        source_vertices=planned_vertices,
                        source_faces=local_faces,
                        boundary_ids=[int(value) for value in loop],
                        boundary_points=np.asarray(
                            cap_decision.source_points,
                            dtype=np.float64,
                        ),
                        inward=active_inward,
                        inward_directions=np.asarray(
                            cap_decision.directions,
                            dtype=np.float64,
                        ),
                        spec=connector_spec,
                    )
                    large_cap_loop = False
                else:
                    cap_decision = attach_harmonic_cap_template(cap_decision)
                    lead_in_applied = bool(
                        float(lead_in_mm) > 1e-9
                        and float(
                            cap_decision.record.get(
                                "thin_boundary_effective_insert_shrink_mm", 0.0
                            )
                        )
                        > 1e-9
                    )
                    large_cap_loop = len(loop) > 512
                    initial_cap_quality = cap_decision_patch_quality_preflight(
                        cap_decision,
                        active_inward,
                        source_points,
                        lead_in_directions,
                        lead_in_mm,
                        lead_in_applied,
                        source_mesh_vertices=planned_vertices,
                        source_mesh_faces=local_faces,
                        defer_cap_triangulation=large_cap_loop,
                    )
                best_candidate = {
                    "factor": 1.0,
                    "decision": cap_decision,
                    "quality": initial_cap_quality,
                    "conormals": loop_conormals,
                }
                runtime_log(
                    "planar-arc-retopology",
                    "interface_retopology_patch_quality",
                    "Planar-arc interface patch quality evaluated",
                    parent_index=int(parent_index),
                    child_index=int(child_index),
                    loop_index=int(selected["loop_index"]),
                    large_loop_deferred=bool(large_cap_loop),
                    invalid_faces=int(initial_cap_quality["invalid_faces"]),
                    degenerate_faces=int(initial_cap_quality["degenerate_faces"]),
                    degenerate_source_faces=int(
                        initial_cap_quality["degenerate_source_faces"]
                    ),
                    degenerate_side_faces=int(
                        initial_cap_quality["degenerate_side_faces"]
                    ),
                    degenerate_cap_faces=int(
                        initial_cap_quality["degenerate_cap_faces"]
                    ),
                    degenerate_cap_samples=initial_cap_quality[
                        "degenerate_cap_samples"
                    ],
                    duplicate_faces=int(initial_cap_quality["duplicate_faces"]),
                )
                selected_cap_surface_quality = best_candidate["quality"].get(
                    "cap_surface_quality",
                    {},
                )
                if not bool(selected_cap_surface_quality.get("valid", False)):
                    selected_decision = best_candidate["decision"]
                    decision_context = {
                        "decision_mode": str(selected_decision.mode),
                        "boundary_loop_count": int(len(loops)),
                        "selected_parent_contact_loop_count": int(
                            len(selected_loop_records)
                        ),
                        "has_cap_template": bool(
                            selected_decision.cap_template_points is not None
                        ),
                        "cap_surface_strategy": selected_decision.record.get(
                            "cap_surface_strategy"
                        ),
                    }
                    raise ValueError(
                        "No printable coherent planar inward cap remains after "
                        "boundary planning; refusing a folded local-offset cap: "
                        + json.dumps(
                            selected_cap_surface_quality,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "; decision_context="
                        + json.dumps(
                            decision_context,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                if not bool(best_candidate["quality"].get("valid", False)):
                    selected_decision = best_candidate["decision"]
                    raise ValueError(
                        "No printable guided/shared interface patch remains after "
                        "boundary planning; refusing a candidate with degenerate "
                        "or stretched side-wall triangles: "
                        + json.dumps(
                            {
                                "part_index": int(child_index),
                                "loop_index": int(selected["loop_index"]),
                                "factor": float(best_candidate["factor"]),
                                "guided_internal_cut": bool(
                                    guided_internal_cut is not None
                                ),
                                "invalid_faces": int(
                                    best_candidate["quality"]["invalid_faces"]
                                ),
                                "degenerate_faces": int(
                                    best_candidate["quality"]["degenerate_faces"]
                                ),
                                "duplicate_faces": int(
                                    best_candidate["quality"]["duplicate_faces"]
                                ),
                                "side_wall_quality": best_candidate["quality"].get(
                                    "side_wall_quality", {}
                                ),
                                "cap_surface_quality": best_candidate["quality"].get(
                                    "cap_surface_quality", {}
                                ),
                                "cap_surface_strategy": selected_decision.record.get(
                                    "cap_surface_strategy"
                                ),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                cap_record = dict(cap_decision.record)
                cap_record["interface_retopology_quality"] = initial_cap_quality
                cap_record["interface_retopology_target_preserved"] = True
                cap_decision = CapDecision(
                    mode=cap_decision.mode,
                    source_vertex_ids=cap_decision.source_vertex_ids,
                    fit_points=cap_decision.fit_points,
                    directions=cap_decision.directions,
                    distances=cap_decision.distances,
                    record=cap_record,
                    source_points=cap_decision.source_points,
                    cap_template_points=cap_decision.cap_template_points,
                    cap_template_faces=cap_decision.cap_template_faces,
                    cap_template_boundary_ids=cap_decision.cap_template_boundary_ids,
                )
                if interface_geometry == "local-connector":
                    if local_connector_safety is None:
                        raise RuntimeError(
                            "local connector preflight did not publish its safety record"
                        )
                    connector_safety = dict(local_connector_safety)
                    connector_record = dict(cap_decision.record)
                    connector_record["local_connector_safety"] = dict(
                        connector_safety
                    )
                    cap_decision = CapDecision(
                        mode=cap_decision.mode,
                        source_vertex_ids=cap_decision.source_vertex_ids,
                        fit_points=cap_decision.fit_points,
                        directions=cap_decision.directions,
                        distances=cap_decision.distances,
                        record=connector_record,
                        source_points=cap_decision.source_points,
                        cap_template_points=cap_decision.cap_template_points,
                        cap_template_faces=cap_decision.cap_template_faces,
                        cap_template_boundary_ids=(
                            cap_decision.cap_template_boundary_ids
                        ),
                    )
                    runtime_log(
                        "厚度测量",
                        "local_connector_safety_precomputed",
                        "局部连接器已使用排除子树后的接口边界等弧采样预计算安全深度",
                        parent_index=int(parent_index),
                        child_index=int(child_index),
                        loop_index=int(selected["loop_index"]),
                        total_safety_mm=float(
                            connector_safety[
                                "local_connector_safety_budget_mm"
                            ]
                        ),
                        backing_safety_mm=float(
                            connector_safety.get(
                                "local_connector_backing_safety_limit_mm",
                                connector_safety[
                                    "boundary_safety_minimum_mm"
                                ],
                            )
                        ),
                    )
                ref["cap_decision"] = cap_decision
                ref["cap_mode"] = cap_decision.mode
                runtime_log(
                    "递归预计算",
                    "layer_child_cap_done",
                    "直属子件帽底安全深度测量完成",
                    parent_index=int(parent_index),
                    child_index=int(child_index),
                    loop_index=int(selected["loop_index"]),
                    boundary_vertex_count=int(len(loop)),
                    selected_cap_mode=str(cap_decision.mode),
                    stage_elapsed_seconds=round(
                        float(time.perf_counter() - loop_started_at),
                        3,
                    ),
                )
            refs.append(ref)
        runtime_log(
            "递归预计算",
            "layer_child_done",
            "直属子件帽底和母槽上下文预计算完成",
            parent_index=int(parent_index),
            child_index=int(child_index),
            selected_loop_count=int(len(selected_loop_records)),
            stage_elapsed_seconds=round(
                float(time.perf_counter() - child_started_at),
                3,
            ),
        )
    runtime_log(
        "递归预计算",
        "layer_child_context_done",
        "全部直属子件帽底和母槽上下文预计算完成",
        parent_index=int(parent_index),
        child_count=int(len(direct_child_indices)),
        cut_reference_count=int(len(refs)),
        stage_elapsed_seconds=round(
            float(time.perf_counter() - context_started_at),
            3,
        ),
    )
    return refs, union_by_child, subtree_by_child

def best_cut_reference(
    body_loop_global_vertices: set[int],
    body_loop_global_edges: set[tuple[int, int]],
    cut_refs: list[dict],
) -> dict | None:
    best_ref = None
    best_rank = None
    for ref in cut_refs:
        ref_edges = ref.get("global_edges", set())
        edge_overlap = len(body_loop_global_edges.intersection(ref_edges))
        vertex_overlap = len(body_loop_global_vertices.intersection(ref["global_vertices"]))
        minimum_edge_overlap = max(3, int(math.ceil(0.1 * max(len(body_loop_global_edges), 1))))
        if edge_overlap < minimum_edge_overlap:
            continue
        rank = (
            edge_overlap / max(len(body_loop_global_edges), 1),
            edge_overlap,
            vertex_overlap,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_ref = ref
    return best_ref

def classify_body_cut_loop_references(
    loops: list[list[int]],
    global_vertex_ids: np.ndarray,
    cut_refs: list[dict],
    *,
    preserve_unmatched_source_geometry: bool,
) -> tuple[list[dict | None], list[dict]]:
    """Match body boundary loops and identify inherited loops that must stay immutable.

    A reloaded recursive body already owns the parent-contact shell emitted by
    its parent step. Only loops matching a current direct-child reference may
    be changed; every unmatched loop belongs to that validated input state.
    """
    loop_refs: list[dict | None] = []
    immutable_records: list[dict] = []
    for loop_index, loop in enumerate(loops):
        ordered_global_ids = [
            int(global_vertex_ids[int(local_index)]) for local_index in loop
        ]
        global_edges = {
            tuple(
                sorted(
                    (
                        ordered_global_ids[position],
                        ordered_global_ids[(position + 1) % len(ordered_global_ids)],
                    )
                )
            )
            for position in range(len(ordered_global_ids))
        }
        ref = best_cut_reference(
            set(ordered_global_ids),
            global_edges,
            cut_refs,
        )
        loop_refs.append(ref)
        if preserve_unmatched_source_geometry and ref is None:
            immutable_records.append(
                {
                    "loop_index": int(loop_index),
                    "vertices": int(len(loop)),
                    "reason": "inherited_parent_contact_shell_or_validated_source_boundary",
                }
            )
    return loop_refs, immutable_records
