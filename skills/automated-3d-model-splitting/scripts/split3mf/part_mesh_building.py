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

def make_part_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    component: Component,
    component_index: int,
    assembly_parent_index: int | None,
    part_id: str,
    max_extension_mm: float,
    interface_retopology: PlanarArcRetopologyContext,
    flat_clearance_mm: float,
    fit_clearance_mm: float,
    insert_shrink_mm: float,
    lead_in_mm: float,
    top_edge_clearance_mm: float,
    clearance_mode: str,
    child_cut_refs: list[dict] | None,
    boundary_neighbor_lookup: dict[tuple[int, int], set[int]],
    component_centers: dict[int, np.ndarray],
    sibling_clearance_mm: float,
    socket_overcut_mm: float,
    bottom_clearance_mm: float,
    inward_override: np.ndarray | None,
    model_center: np.ndarray,
    cap_mode: str,
    planar_extra_limit_mm: float,
    parent_contact_only: bool = False,
) -> tuple[trimesh.Trimesh, dict]:
    parent_thickness_probe = ParentThicknessProbe(vertices, faces)
    local_vertices, local_faces, _, global_vertex_ids = build_local_mesh(vertices, faces, component)
    boundary_match_vertices = np.asarray(local_vertices, dtype=np.float64).copy()
    loops = boundary_loops(local_faces)
    boundary_vertex_count = int(sum(len(loop) for loop in loops))

    component_center = local_vertices[local_faces.reshape(-1)].mean(axis=0)
    outward = average_outward_normal(local_vertices, local_faces, component_center, model_center)
    inward = inward_override if inward_override is not None else -outward
    inward = inward / max(float(np.linalg.norm(inward)), 1e-12)
    vertex_inward_normals = mesh_vertex_inward_normals(local_vertices, local_faces)
    loop_inward_directions: dict[int, np.ndarray] = {}
    local_inward_direction_records: list[dict] = []
    for loop_index, loop in enumerate(loops):
        directions, direction_record = safe_boundary_inward_directions(
            vertex_inward_normals,
            loop,
            inward,
            loop_points=local_vertices[np.asarray(loop, dtype=np.int64)],
        )
        loop_inward_directions[int(loop_index)] = directions
        local_inward_direction_records.append({"loop_index": int(loop_index), **direction_record})

    visible_source_vertices = np.asarray(local_vertices, dtype=np.float64).copy()
    retopology_vertices, interface_retopology_records = InterfaceRetopologyService.retopologize_local_loops(
        visible_source_vertices.copy(),
        loops,
        global_vertex_ids,
        interface_retopology,
        local_faces=local_faces,
    )
    visible_source_vertices = np.asarray(
        retopology_vertices, dtype=np.float64
    ).copy()
    hidden_geometry_vertices = generated_geometry_boundary_vertices(
        visible_source_vertices,
        loops,
        interface_retopology_records,
    )
    boundary_match_vertices = visible_source_vertices.copy()

    component_center = visible_source_vertices[local_faces.reshape(-1)].mean(axis=0)
    u, v = orthonormal_basis(inward)
    insert_shrink_mm = max(float(insert_shrink_mm), 0.0)
    lead_in_mm = max(float(lead_in_mm), 0.0)
    requested_top_edge_clearance_mm = min(
        max(float(top_edge_clearance_mm), 0.0),
        insert_shrink_mm,
    )
    top_edge_clearance_mm = visible_top_edge_clearance(insert_shrink_mm)
    socket_overcut_mm = max(float(socket_overcut_mm), 0.0)
    parent_socket_overcut_mm = clearance_offsets(clearance_mode, fit_clearance_mm)[1]
    bottom_clearance_mm = max(float(bottom_clearance_mm), 0.0)
    insert_planar_extra_limit_mm = max(
        float(planar_extra_limit_mm) - bottom_clearance_mm,
        0.0,
    )
    child_cut_refs = child_cut_refs or []
    child_component_indices = {int(ref["component_index"]) for ref in child_cut_refs}
    socket_flat_clearance_mm = max(float(flat_clearance_mm), 0.0)
    sibling_clearance_mm = max(float(sibling_clearance_mm), 0.0)

    loop_fit_points: list[np.ndarray] = []
    loop_clearance_records = []
    insert_loop_records: list[dict] = []
    sibling_clearance_records: list[dict] = []
    socket_extension_records: list[dict] = []
    skipped_socket_records: list[dict] = []
    skipped_non_parent_loop_records: list[dict] = []
    dropped_non_parent_face_indices: set[int] = set()
    socket_side_faces = 0
    socket_cap_faces = 0

    output_vertices: list[np.ndarray] = [p.copy() for p in visible_source_vertices]
    output_faces: list[list[int]] = local_faces.astype(int).tolist()

    def sibling_cleared_points(loop: list[int], points: np.ndarray) -> tuple[np.ndarray, dict]:
        if sibling_clearance_mm <= 1e-9:
            return points, {"sibling_edges": 0, "sibling_vertices": 0, "sibling_neighbor_indices": []}
        offsets = np.zeros_like(points)
        counts = np.zeros(len(points), dtype=np.int64)
        sibling_neighbors: set[int] = set()
        loop_position = {int(local_index): position for position, local_index in enumerate(loop)}
        for position, local_a in enumerate(loop):
            local_b = int(loop[(position + 1) % len(loop)])
            global_edge = tuple(sorted((int(global_vertex_ids[local_a]), int(global_vertex_ids[local_b]))))
            side_neighbors = [
                int(neighbor)
                for neighbor in boundary_neighbor_lookup.get(global_edge, set())
                if int(neighbor) != component_index
                and int(neighbor) != int(assembly_parent_index or -1)
                and int(neighbor) not in child_component_indices
            ]
            if not side_neighbors:
                continue
            for neighbor_index in side_neighbors:
                neighbor_center = component_centers.get(neighbor_index)
                if neighbor_center is None:
                    continue
                sibling_neighbors.add(neighbor_index)
                for local_index in (int(local_a), local_b):
                    point_position = loop_position[int(local_index)]
                    direction = points[point_position] - neighbor_center
                    direction = direction - inward * float(np.dot(direction, inward))
                    length = float(np.linalg.norm(direction))
                    if length <= 1e-9:
                        continue
                    offsets[point_position] += direction / length * sibling_clearance_mm
                    counts[point_position] += 1
        shifted = points.copy()
        active = counts > 0
        if np.any(active):
            shifted[active] = shifted[active] + offsets[active] / counts[active, None]
        return shifted, {
            "sibling_edges": int(np.sum(counts) // 2),
            "sibling_vertices": int(np.sum(active)),
            "sibling_neighbor_indices": sorted(sibling_neighbors),
        }

    for loop_index, loop in enumerate(loops):
        loop_array = np.array(loop, dtype=np.int64)
        source_boundary_points = visible_source_vertices[loop_array]
        internal_boundary_points = hidden_geometry_vertices[loop_array]
        loop_interior_conormals = boundary_loop_interior_conormals(
            hidden_geometry_vertices, local_faces, loop, reference_axis=inward
        )
        internal_boundary_points, sibling_record = sibling_cleared_points(loop, internal_boundary_points)
        loop_global_vertices = set(int(global_vertex_ids[i]) for i in loop)
        loop_global_ordered = [int(global_vertex_ids[i]) for i in loop]
        loop_global_edges = {
            tuple(sorted((loop_global_ordered[position], loop_global_ordered[(position + 1) % len(loop_global_ordered)])))
            for position in range(len(loop_global_ordered))
        }
        socket_ref = best_cut_reference(loop_global_vertices, loop_global_edges, child_cut_refs)
        loop_parent_edges = 0
        loop_neighbor_counts: collections.Counter[int] = collections.Counter()
        if assembly_parent_index is not None:
            for position, local_a in enumerate(loop):
                local_b = loop[(position + 1) % len(loop)]
                global_edge = tuple(sorted((int(global_vertex_ids[local_a]), int(global_vertex_ids[local_b]))))
                neighbors = boundary_neighbor_lookup.get(global_edge, set())
                for neighbor_index in neighbors:
                    if int(neighbor_index) != component_index:
                        loop_neighbor_counts[int(neighbor_index)] += 1
                if int(assembly_parent_index) in neighbors:
                    loop_parent_edges += 1
        shared_loop_parent_edges = loop_parent_edges
        shared_loop_child_edges = 0
        shared_parent_child_loop = False
        if socket_ref is not None and assembly_parent_index is not None:
            child_component_index = int(socket_ref["component_index"])
            for position, local_a in enumerate(loop):
                local_b = loop[(position + 1) % len(loop)]
                global_edge = tuple(sorted((int(global_vertex_ids[local_a]), int(global_vertex_ids[local_b]))))
                neighbors = boundary_neighbor_lookup.get(global_edge, set())
                if child_component_index in neighbors:
                    shared_loop_child_edges += 1
            shared_parent_child_loop = shared_loop_parent_edges >= 3 and shared_loop_child_edges >= 3
        if socket_ref is not None:
            socket_inward = socket_ref["inward"]
            socket_cap_mode = str(socket_ref.get("cap_mode", cap_mode))
            socket_planar_extra_limit_mm = float(
                socket_ref.get("planar_extra_limit_mm")
                if socket_ref.get("planar_extra_limit_mm") is not None
                else planar_extra_limit_mm
            )
            socket_processing_mode = str(socket_ref.get("processing_mode", "inward"))
            if shared_parent_child_loop:
                skipped_socket_records.append(
                    {
                        "loop_index": loop_index,
                        "matched_component_index": socket_ref["component_index"],
                        "matched_cap_mode": socket_cap_mode,
                        "matched_processing_mode": socket_processing_mode,
                        "matched_color_code": socket_ref["color_code"],
                        "matched_color_name": socket_ref["color_name"],
                        "skip_reason": "shared_parent_child_loop",
                        "shared_loop_parent_edges": shared_loop_parent_edges,
                        "shared_loop_child_edges": shared_loop_child_edges,
                    }
                )
            else:
                socket_u, socket_v = orthonormal_basis(socket_inward)
                socket_points = internal_boundary_points
                effective_socket_overcut_mm = max(
                    float(socket_ref.get("socket_overcut_mm", socket_overcut_mm)), 0.0
                )
                if effective_socket_overcut_mm > 1e-9:
                    socket_points = offset_points_along_conormals(
                        internal_boundary_points,
                        loop_interior_conormals,
                        effective_socket_overcut_mm,
                    )
                socket_directions = reference_loop_inward_directions(
                    socket_ref,
                    loop_global_ordered,
                    socket_inward,
                )
                matched_insert_shrink_mm, _ = clearance_offsets(
                    clearance_mode,
                    float(socket_ref.get("fit_clearance_mm", fit_clearance_mm)),
                )
                socket_bottom_points, matched_plane_record = matched_socket_bottom_geometry(
                    source_boundary_points=source_boundary_points,
                    socket_top_points=socket_points,
                    inward=socket_inward,
                    inward_directions=socket_directions,
                    child_insert_shrink_mm=matched_insert_shrink_mm,
                    max_extension_mm=max_extension_mm,
                    flat_clearance_mm=socket_flat_clearance_mm,
                    cap_mode=socket_cap_mode,
                    planar_extra_limit_mm=socket_planar_extra_limit_mm,
                    bottom_clearance_mm=bottom_clearance_mm,
                    parent_thickness_probe=parent_thickness_probe,
                    cap_decision=socket_ref.get("cap_decision"),
                    source_vertex_ids=loop_global_ordered,
                    boundary_match_points=boundary_match_vertices[loop_array],
                    boundary_reconciliation_tolerance_mm=max(
                        float(interface_retopology.config.maximum_safe_target_offset_mm),
                        float(
                            socket_ref.get(
                                "boundary_reconciliation_tolerance_mm",
                                socket_ref.get("fit_clearance_mm", fit_clearance_mm),
                            )
                        ),
                    ),
                    error_context=(
                        f"{part_id} loop {int(loop_index)} matching child "
                        f"P{int(socket_ref['component_index']):02d}"
                    ),
                    child_interior_conormals=reference_loop_interior_conormals(
                        socket_ref, loop_global_ordered
                    ),
                )
                added_side, added_cap, extension_record = add_inward_lead_extrusion_and_cap(
                    output_vertices=output_vertices,
                    output_faces=output_faces,
                    visible_top_ids=[int(i) for i in loop],
                    visible_top_points=source_boundary_points,
                    internal_fit_points=socket_points,
                    inward=socket_inward,
                    inward_directions=socket_directions,
                    bottom_points=socket_bottom_points,
                    lead_in_mm=lead_in_mm,
                    max_extension_mm=max_extension_mm,
                    flat_clearance_mm=socket_flat_clearance_mm,
                    cap_mode=socket_cap_mode,
                    planar_extra_limit_mm=socket_planar_extra_limit_mm,
                    plane_record=matched_plane_record,
                    parent_thickness_probe=parent_thickness_probe,
                    lead_in_directions=socket_directions,
                    cap_template_decision=socket_ref.get("cap_decision"),
                )
                socket_side_faces += added_side
                socket_cap_faces += added_cap
                extension_record["loop_index"] = loop_index
                extension_record["matched_component_index"] = socket_ref["component_index"]
                extension_record["matched_cap_mode"] = socket_cap_mode
                extension_record["matched_color_code"] = socket_ref["color_code"]
                extension_record["matched_color_name"] = socket_ref["color_name"]
                extension_record["socket_overcut_mm"] = effective_socket_overcut_mm
                extension_record["bottom_clearance_mm"] = bottom_clearance_mm
                extension_record["shared_loop_parent_edges"] = shared_loop_parent_edges
                extension_record["shared_loop_child_edges"] = shared_loop_child_edges
                extension_record["shared_parent_child_loop"] = shared_parent_child_loop
                socket_extension_records.append(extension_record)
                continue

        if parent_contact_only and assembly_parent_index is not None and loop_parent_edges < 3:
            dropped_source_faces: list[int] = []
            if len(loop) <= 8 and not loop_neighbor_counts:
                loop_vertex_set = {int(value) for value in loop}
                loop_edges = {
                    tuple(sorted((int(loop[position]), int(loop[(position + 1) % len(loop)]))))
                    for position in range(len(loop))
                }
                for face_index, face in enumerate(local_faces):
                    face_values = [int(value) for value in face]
                    if not set(face_values).issubset(loop_vertex_set):
                        continue
                    face_edges = {
                        tuple(sorted((face_values[0], face_values[1]))),
                        tuple(sorted((face_values[1], face_values[2]))),
                        tuple(sorted((face_values[2], face_values[0]))),
                    }
                    if face_edges.intersection(loop_edges):
                        dropped_non_parent_face_indices.add(int(face_index))
                        dropped_source_faces.append(int(face_index))
            skipped_non_parent_loop_records.append(
                {
                    "loop_index": loop_index,
                    "vertices": len(loop),
                    "parent_edges": int(loop_parent_edges),
                    "neighbor_counts": {str(key): int(value) for key, value in sorted(loop_neighbor_counts.items())},
                    "dropped_source_faces": dropped_source_faces,
                    "skip_reason": "non_parent_non_child_boundary",
                }
            )
            continue

        top_points = source_boundary_points.copy()
        effective_profile_inset_mm, taper_profile_record = tapered_profile_inset_limit(
            top_points,
            fit_clearance_mm=insert_shrink_mm,
            maximum_taper_depth_mm=lead_in_mm,
            maximum_total_depth_mm=tapered_profile_total_depth_budget(
                max_extension_mm,
                insert_planar_extra_limit_mm,
            ),
        )
        fit_points = offset_points_along_conormals(
            internal_boundary_points,
            loop_interior_conormals,
            effective_profile_inset_mm,
        )
        loop_fit_points.append(fit_points)
        insert_loop_records.append(
            {
                "loop_index": loop_index,
                "loop": [int(i) for i in loop],
                "top_points": top_points,
                "source_boundary_points": source_boundary_points,
                "fit_points": fit_points,
                "inward_directions": loop_inward_directions[int(loop_index)],
                "lead_in_directions": loop_inward_directions[int(loop_index)],
                "interior_conormals": loop_interior_conormals,
                "effective_profile_inset_mm": effective_profile_inset_mm,
                "taper_profile_record": taper_profile_record,
            }
        )
        loop_clearance_records.append(
            {
                "loop_index": loop_index,
                "vertices": len(loop),
                "insert_shrink_mm": insert_shrink_mm,
                "effective_profile_inset_mm": effective_profile_inset_mm,
                **taper_profile_record,
                "top_edge_clearance_mm": top_edge_clearance_mm,
                "lead_in_mm": lead_in_mm,
                "sibling_clearance_mm": sibling_clearance_mm,
                **sibling_record,
            }
        )
        if sibling_record["sibling_edges"] > 0:
            record = dict(sibling_record)
            record["loop_index"] = loop_index
            record["sibling_clearance_mm"] = sibling_clearance_mm
            sibling_clearance_records.append(record)

    if dropped_non_parent_face_indices:
        output_faces = [
            face
            for face_index, face in enumerate(output_faces)
            if face_index not in dropped_non_parent_face_indices
        ]

    loop_points_2d_for_grouping = [project_points(record["top_points"], component_center, u, v) for record in insert_loop_records]
    loop_groups = group_loops(loop_points_2d_for_grouping)

    loops_bottom: list[list[int]] = []
    loops_lead: list[list[int]] = []
    bottom_loop_points_2d: list[np.ndarray] = []
    group_extension_records = []

    for insert_index, record in enumerate(insert_loop_records):
        loop_index = int(record["loop_index"])
        top_ids = record["loop"]
        fit_points = record["fit_points"]
        inward_directions = record["inward_directions"]
        lead_in_directions = record["lead_in_directions"]
        interior_conormals = record["interior_conormals"]
        distances, inward_directions, plane_record = boundary_cap_distances(
            fit_points,
            inward,
            inward_directions,
            max_extension_mm,
            flat_clearance_mm,
            cap_mode,
            insert_planar_extra_limit_mm,
            parent_thickness_probe,
        )
        source_boundary_points = record["source_boundary_points"]
        expected_parent_socket_top = offset_points_along_conormals(
            source_boundary_points,
            interior_conormals,
            -parent_socket_overcut_mm,
        )
        distances, plane_record = reserve_flat_socket_travel_budget(
            child_fit_points=fit_points,
            child_distances=distances,
            child_directions=inward_directions,
            socket_top_points=expected_parent_socket_top,
            bottom_clearance_mm=bottom_clearance_mm,
            maximum_socket_travel_mm=max_extension_mm + max(float(planar_extra_limit_mm), 0.0),
            plane_record=plane_record,
        )
        lead_ids = []
        lead_taper_record: dict = {}
        if (
            lead_in_mm > 1e-9
            and float(record["effective_profile_inset_mm"])
            > top_edge_clearance_mm + 1e-9
        ):
            lead_points, lead_depths, effective_lead_directions, lead_taper_record = coherent_tapered_lead_ring(
                record["top_points"],
                fit_points,
                lead_in_directions,
                distances,
                lead_in_mm,
                reference_axis=inward,
            )
            for lead_point in lead_points:
                lead_ids.append(len(output_vertices))
                output_vertices.append(lead_point)
        bottom_ids = []
        for fit_point, direction, distance in zip(fit_points, inward_directions, distances):
            bottom_point = fit_point + direction * float(distance)
            bottom_ids.append(len(output_vertices))
            output_vertices.append(bottom_point)
        loops_lead.append(lead_ids)
        loops_bottom.append(bottom_ids)
        bottom_loop_points_2d.append(project_points(np.array([output_vertices[i] for i in bottom_ids]), component_center, u, v))
        group_extension_records.append(
            {
                "loop_index": loop_index,
                "vertices": len(top_ids),
                "extension_min_mm": float(distances.min()),
                "extension_max_mm": float(distances.max()),
                "lead_in_applied": bool(lead_ids),
                "visible_top_source_preserved": False,
                "visible_boundary_retopologized": True,
                "interface_retopology_surface_role": "shared_visible_boundary_and_surface_band",
                "lead_in_depth_min_mm": float(
                    min((np.linalg.norm(np.array(output_vertices[i]) - fit_points[pos]) for pos, i in enumerate(lead_ids)), default=0.0)
                ),
                "lead_in_depth_max_mm": float(
                    max((np.linalg.norm(np.array(output_vertices[i]) - fit_points[pos]) for pos, i in enumerate(lead_ids)), default=0.0)
                ),
                **lead_taper_record,
                **plane_record,
            }
        )

    side_faces = 0
    for record, lead_ids, bottom_ids in zip(insert_loop_records, loops_lead, loops_bottom):
        top_ring = record["loop"]
        if lead_ids:
            side_faces += add_side_faces_between_rings(
                output_faces, top_ring, lead_ids, vertices=output_vertices
            )
            side_faces += add_side_faces_between_rings(
                output_faces, lead_ids, bottom_ids, vertices=output_vertices
            )
        else:
            side_faces += add_side_faces_between_rings(
                output_faces, top_ring, bottom_ids, vertices=output_vertices
            )

    cap_faces = triangulate_cap(output_vertices, output_faces, loops_bottom, bottom_loop_points_2d, loop_groups, inward)

    mesh = trimesh.Trimesh(
        vertices=np.array(output_vertices, dtype=np.float64),
        faces=np.array(output_faces, dtype=np.int64),
        process=False,
        metadata={"name": part_id},
    )
    mesh = finalize_source_preserving_mesh(
        mesh,
        protected_source_face_count=len(local_faces),
    )

    color_info = COLOR_INFO.get(component.color_code, {"name": component.color_code, "hex": "", "rgba": [200, 200, 200, 255]})
    mesh.visual.face_colors = np.tile(np.array(color_info["rgba"], dtype=np.uint8), (len(mesh.faces), 1))

    bbox = mesh.bounds
    selected_cap_modes = sorted(
        {str(record.get("cap_mode", cap_mode)) for record in group_extension_records + socket_extension_records}
    )
    geometry_cap_mode = selected_cap_modes[0] if len(selected_cap_modes) == 1 else "mixed"
    stats = {
        "part_id": part_id,
        "processing_mode": "insert_inward_adaptive_bottom",
        "selected_processing_mode": "inward",
        "color_code": component.color_code,
        "color_name": color_info["name"],
        "color_hex": color_info["hex"],
        "source_faces": component.face_count,
        "output_faces": int(len(mesh.faces)),
        "output_vertices": int(len(mesh.vertices)),
        "boundary_loops": len(loops),
        "boundary_vertices": boundary_vertex_count,
        "interface_retopology_records": interface_retopology_records,
        "loop_groups": loop_groups,
        "side_faces_added": side_faces + socket_side_faces,
        "cap_faces_added": cap_faces + socket_cap_faces,
        "mesh_finalization": copy.deepcopy(
            mesh.metadata.get("source_preserving_finalization", {})
        ),
        "insert_side_faces_added": side_faces,
        "insert_cap_faces_added": cap_faces,
        "child_socket_side_faces_added": socket_side_faces,
        "child_socket_cap_faces_added": socket_cap_faces,
        "max_extension_limit_mm": max_extension_mm,
        "cap_mode": cap_mode,
        "geometry_cap_mode": geometry_cap_mode,
        "planar_extra_limit_mm": planar_extra_limit_mm,
        "selected_cap_modes": selected_cap_modes,
        "flat_clearance_mm": flat_clearance_mm,
        "clearance_mode": clearance_mode,
        "fit_clearance_mm": max(float(fit_clearance_mm), 0.0),
        "insert_shrink_mm": insert_shrink_mm,
        "socket_overcut_mm": 0.0,
        "child_socket_overcut_mm": socket_overcut_mm,
        "bottom_clearance_mm": 0.0,
        "child_socket_bottom_clearance_mm": bottom_clearance_mm,
        "lead_in_mm": lead_in_mm,
        "requested_top_edge_clearance_mm": requested_top_edge_clearance_mm,
        "top_edge_clearance_mm": top_edge_clearance_mm,
        "sibling_clearance_mm": sibling_clearance_mm,
        "parent_contact_only": bool(parent_contact_only),
        "visible_top_source_preserved": False,
        "visible_boundary_retopologized": True,
        "interface_retopology_surface_role": "shared_visible_boundary_and_surface_band",
        "inward_direction": inward.round(6).tolist(),
        "inward_direction_source": "assembly_parent" if inward_override is not None else "model_center",
        "local_inward_direction_records": local_inward_direction_records,
        "local_inward_outward_vertices_before": int(
            sum(record["global_outward_vertices_before"] for record in local_inward_direction_records)
        ),
        "local_inward_outward_vertices_after": int(
            sum(record["outward_vertices_after"] for record in local_inward_direction_records)
        ),
        "extension_max_observed_mm": float(max((r["extension_max_mm"] for r in group_extension_records), default=0.0)),
        "maximum_generated_inward_travel_mm": float(
            max(
                (r["extension_max_mm"] for r in group_extension_records + socket_extension_records),
                default=0.0,
            )
        ),
        "extension_min_observed_mm": float(min((r["extension_min_mm"] for r in group_extension_records), default=0.0)),
        "flat_bottom_required_max_extension_mm": float(max((r["flat_bottom_required_max_extension_mm"] for r in group_extension_records), default=0.0)),
        "flat_bottom_target_exceeded": bool(any(r["target_exceeded"] for r in group_extension_records)),
        "fixed_inward_depth_mm": max_extension_mm,
        "boundary_span_max_mm": float(max((r["flat_span_mm"] for r in group_extension_records), default=0.0)),
        "fixed_inward_depth_applied": bool(all(r.get("fixed_inward_depth_applied", False) for r in group_extension_records)) if group_extension_records else True,
        "child_socket_count": len(socket_extension_records),
        "child_socket_extensions": socket_extension_records,
        "child_socket_skipped_count": len(skipped_socket_records),
        "child_socket_skipped": skipped_socket_records,
        "skipped_non_parent_loop_count": len(skipped_non_parent_loop_records),
        "skipped_non_parent_loops": skipped_non_parent_loop_records,
        "dropped_non_parent_source_faces": int(len(dropped_non_parent_face_indices)),
        "loop_clearances": loop_clearance_records,
        "sibling_clearances": sibling_clearance_records,
        "loop_extensions": group_extension_records,
        "bbox_min": bbox[0].round(6).tolist(),
        "bbox_max": bbox[1].round(6).tolist(),
        "bbox_size_mm": (bbox[1] - bbox[0]).round(6).tolist(),
        **mesh_runtime_stats(mesh),
    }
    return mesh, stats

def make_body_cut_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    body_component: Component,
    part_id: str,
    cut_refs: list[dict],
    max_extension_mm: float,
    interface_retopology: PlanarArcRetopologyContext,
    flat_clearance_mm: float,
    fit_clearance_mm: float,
    socket_overcut_mm: float,
    bottom_clearance_mm: float,
    clearance_mode: str,
    model_center: np.ndarray,
    cap_mode: str,
    planar_extra_limit_mm: float,
    lead_in_mm: float,
    preserve_unmatched_source_geometry: bool = True,
    interface_geometry: str = "boundary-extrusion",
    defer_local_connector_boolean: bool = False,
) -> tuple[trimesh.Trimesh, dict]:
    if interface_geometry == "local-connector" and defer_local_connector_boolean:
        from .boolean_parent import build_complete_boolean_parent
        mesh, stats = build_complete_boolean_parent(
            vertices, faces, part_id=part_id, cut_refs=cut_refs,
            interface_retopology=interface_retopology,
        )
        stats['complete_recursive_source_faces'] = stats['source_faces']
        stats['source_faces'] = int(body_component.face_count)
        color_info = COLOR_INFO.get(body_component.color_code,
                                    {'name': body_component.color_code, 'hex': ''})
        stats.update(color_code=body_component.color_code,
                     color_name=color_info['name'], color_hex=color_info['hex'],
                     boundary_loops=len(cut_refs), cap_mode=cap_mode)
        return mesh, stats
    parent_thickness_probe = ParentThicknessProbe(vertices, faces)
    local_vertices, local_faces, _global_to_local, global_vertex_ids = build_local_mesh(vertices, faces, body_component)
    boundary_match_vertices = np.asarray(local_vertices, dtype=np.float64).copy()
    loops = boundary_loops(local_faces)
    boundary_vertex_count = int(sum(len(loop) for loop in loops))
    component_center = local_vertices[local_faces.reshape(-1)].mean(axis=0)
    fallback_inward = -average_outward_normal(local_vertices, local_faces, component_center, model_center)
    fallback_inward /= max(float(np.linalg.norm(fallback_inward)), 1e-12)
    vertex_inward_normals = mesh_vertex_inward_normals(local_vertices, local_faces)
    fallback_loop_directions: dict[int, np.ndarray] = {}
    fallback_direction_records: list[dict] = []
    for loop_index, loop in enumerate(loops):
        directions, direction_record = safe_boundary_inward_directions(
            vertex_inward_normals,
            loop,
            fallback_inward,
            loop_points=local_vertices[np.asarray(loop, dtype=np.int64)],
        )
        fallback_loop_directions[int(loop_index)] = directions
        fallback_direction_records.append({"loop_index": int(loop_index), **direction_record})

    loop_cut_refs, immutable_source_loop_records = classify_body_cut_loop_references(
        loops,
        global_vertex_ids,
        cut_refs,
        preserve_unmatched_source_geometry=preserve_unmatched_source_geometry,
    )
    mutable_loop_indices = [
        int(loop_index)
        for loop_index, ref in enumerate(loop_cut_refs)
        if not preserve_unmatched_source_geometry or ref is not None
    ]
    mutable_loops = [loops[loop_index] for loop_index in mutable_loop_indices]
    visible_source_vertices = np.asarray(local_vertices, dtype=np.float64).copy()
    retopology_vertices, interface_retopology_records = InterfaceRetopologyService.retopologize_local_loops(
        visible_source_vertices.copy(),
        mutable_loops,
        global_vertex_ids,
        interface_retopology,
        local_faces=local_faces,
    )
    visible_source_vertices = np.asarray(
        retopology_vertices, dtype=np.float64
    ).copy()
    for retopology_record, loop_index in zip(
        interface_retopology_records,
        mutable_loop_indices,
    ):
        retopology_record["loop_index"] = int(loop_index)
    hidden_geometry_vertices = generated_geometry_boundary_vertices(
        visible_source_vertices,
        mutable_loops,
        interface_retopology_records,
    )
    if immutable_source_loop_records:
        runtime_log(
            "递归几何",
            "inherited_parent_contact_shell_preserved",
            "上阶段父接触外壳的未匹配边界保持不可变，仅处理本阶段直属子件接口",
            part_id=str(part_id),
            immutable_loop_count=int(len(immutable_source_loop_records)),
            immutable_vertex_count=int(
                sum(record["vertices"] for record in immutable_source_loop_records)
            ),
            immutable_loops=immutable_source_loop_records,
            mutable_child_interface_loop_indices=mutable_loop_indices,
        )

    output_vertices: list[np.ndarray] = [p.copy() for p in visible_source_vertices]
    output_faces: list[list[int]] = local_faces.astype(int).tolist()
    component_center = visible_source_vertices[local_faces.reshape(-1)].mean(axis=0)

    side_faces = 0
    cap_faces = 0
    loop_extension_records = []
    local_socket_cutters: list[trimesh.Trimesh] = []
    local_socket_boolean_records: list[dict] = []
    deferred_local_connector_count = 0
    matched_refs = []
    applied_direction_records = []
    socket_overcut_mm = max(float(socket_overcut_mm), 0.0)
    bottom_clearance_mm = max(float(bottom_clearance_mm), 0.0)
    body_flat_clearance_mm = max(float(flat_clearance_mm), 0.0)
    lead_in_mm = max(float(lead_in_mm), 0.0)

    for loop_index, loop in enumerate(loops):
        body_loop_global = set(int(global_vertex_ids[i]) for i in loop)
        body_loop_global_ordered = [int(global_vertex_ids[i]) for i in loop]
        body_loop_global_edges = {
            tuple(sorted((body_loop_global_ordered[position], body_loop_global_ordered[(position + 1) % len(body_loop_global_ordered)])))
            for position in range(len(body_loop_global_ordered))
        }
        ref = loop_cut_refs[int(loop_index)]
        runtime_log(
            "递归诊断",
            "body_loop_cut_reference_selected",
            "记录父体边界环与直属子件帽底参考的匹配范围",
            part_id=str(part_id),
            loop_index=int(loop_index),
            parent_loop_vertex_count=int(len(body_loop_global_ordered)),
            matched_component_index=(
                None if ref is None else int(ref["component_index"])
            ),
            child_reference_vertex_count=(
                0 if ref is None else int(len(ref.get("global_loop", [])))
            ),
            shared_source_vertex_count=(
                0
                if ref is None
                else int(len(body_loop_global.intersection(ref["global_vertices"])))
            ),
            shared_source_edge_count=(
                0
                if ref is None
                else int(len(body_loop_global_edges.intersection(ref.get("global_edges", set()))))
            ),
            parent_contact_neighbor_counts=(
                {} if ref is None else dict(ref.get("parent_contact_neighbor_counts", {}))
            ),
        )
        if preserve_unmatched_source_geometry and ref is None:
            continue
        inward = ref["inward"] if ref else fallback_inward
        inward_directions = (
            reference_loop_inward_directions(ref, body_loop_global_ordered, inward)
            if ref is not None
            else fallback_loop_directions[int(loop_index)]
        )
        direction_record = dict(
            ref.get("local_inward_direction_record", {})
            if ref is not None
            else fallback_direction_records[int(loop_index)]
        )
        direction_record["loop_index"] = int(loop_index)
        direction_record["matched_component_index"] = ref.get("component_index") if ref else None
        applied_direction_records.append(direction_record)
        loop_cap_mode = str(ref.get("cap_mode", cap_mode)) if ref else cap_mode
        loop_planar_extra_limit_mm = float(
            ref.get("planar_extra_limit_mm")
            if ref and ref.get("planar_extra_limit_mm") is not None
            else planar_extra_limit_mm
        )
        u, v = orthonormal_basis(inward)
        loop_array = np.array(loop, dtype=np.int64)
        source_boundary_points = visible_source_vertices[loop_array]
        internal_socket_points = hidden_geometry_vertices[loop_array]
        loop_interior_conormals = boundary_loop_interior_conormals(
            hidden_geometry_vertices, local_faces, loop, reference_axis=inward
        )
        effective_socket_overcut_mm = max(
            float(ref.get("socket_overcut_mm", socket_overcut_mm)) if ref else socket_overcut_mm,
            0.0,
        )
        if effective_socket_overcut_mm > 1e-9:
            internal_socket_points = offset_points_along_conormals(
                internal_socket_points,
                loop_interior_conormals,
                effective_socket_overcut_mm,
            )
        shared_cap_decision = ref.get("cap_decision") if ref else None
        if shared_cap_decision is None:
            distances, inward_directions, plane_record = boundary_cap_distances(
                internal_socket_points,
                fallback_inward=inward,
                inward_directions=inward_directions,
                fixed_depth_mm=max_extension_mm,
                flat_clearance_mm=body_flat_clearance_mm,
                cap_mode=loop_cap_mode,
                planar_extra_limit_mm=loop_planar_extra_limit_mm,
                parent_thickness_probe=parent_thickness_probe,
            )
            socket_bottom_points = (
                internal_socket_points + inward_directions * distances[:, None]
            )
        else:
            child_insert_shrink_mm, _unused_socket_overcut_mm = clearance_offsets(
                clearance_mode,
                float(ref.get("fit_clearance_mm", fit_clearance_mm)),
            )
            socket_bottom_points, shared_record = matched_socket_bottom_geometry(
                source_boundary_points=source_boundary_points,
                socket_top_points=internal_socket_points,
                inward=inward,
                inward_directions=inward_directions,
                child_insert_shrink_mm=child_insert_shrink_mm,
                max_extension_mm=max_extension_mm,
                flat_clearance_mm=body_flat_clearance_mm,
                cap_mode=loop_cap_mode,
                planar_extra_limit_mm=loop_planar_extra_limit_mm,
                bottom_clearance_mm=bottom_clearance_mm,
                parent_thickness_probe=parent_thickness_probe,
                cap_decision=shared_cap_decision,
                source_vertex_ids=body_loop_global_ordered,
                boundary_match_points=boundary_match_vertices[loop_array],
                boundary_reconciliation_tolerance_mm=max(
                    float(interface_retopology.config.maximum_safe_target_offset_mm),
                    float(
                        ref.get(
                            "boundary_reconciliation_tolerance_mm",
                            ref.get("fit_clearance_mm", fit_clearance_mm),
                        )
                    ),
                ),
                error_context=(
                    f"{part_id} loop {int(loop_index)} matching child "
                    f"P{int(ref['component_index']):02d}"
                ),
                child_interior_conormals=reference_loop_interior_conormals(
                    ref, body_loop_global_ordered
                ),
            )
            plane_record = shared_record
        if interface_geometry == "local-connector":
            connector_fit_clearance_mm = float(
                ref.get("fit_clearance_mm", fit_clearance_mm)
                if ref is not None
                else fit_clearance_mm
            )
            if shared_cap_decision is not None:
                connector_boundary_distances = np.asarray(
                    shared_cap_decision.distances,
                    dtype=np.float64,
                )
            else:
                connector_boundary_distances = np.asarray(
                    distances,
                    dtype=np.float64,
                )
            precomputed_connector_safety = (
                shared_cap_decision.record.get("local_connector_safety")
                if shared_cap_decision is not None
                else None
            )
            connector_safety = (
                dict(precomputed_connector_safety)
                if isinstance(precomputed_connector_safety, dict)
                else local_connector_safe_depth_at_boundary(
                    boundary_points=source_boundary_points,
                    inward=inward,
                    fit_clearance_mm=connector_fit_clearance_mm,
                    bottom_clearance_mm=bottom_clearance_mm,
                    lead_in_mm=lead_in_mm,
                    boundary_distances=connector_boundary_distances,
                    parent_thickness_probe=parent_thickness_probe,
                )
            )
            safe_connector_depth_mm = float(
                connector_safety["local_connector_safety_budget_mm"]
            )
            connector_spec = local_connector_spec_for_interface(
                fit_clearance_mm=connector_fit_clearance_mm,
                bottom_clearance_mm=bottom_clearance_mm,
                lead_in_mm=lead_in_mm,
                safe_engagement_depth_mm=safe_connector_depth_mm,
                safe_backing_depth_mm=float(
                    connector_safety.get(
                        "local_connector_backing_safety_limit_mm",
                        connector_safety["boundary_safety_minimum_mm"],
                    )
                ),
                compact_peg_supported=bool(
                    connector_safety.get(
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
                visible_interface_simplification_tolerance=float(
                    interface_retopology.config.visible_interface_simplification_tolerance
                ),
            )
            preview_vertex_checkpoint = len(output_vertices)
            preview_face_checkpoint = len(output_faces)
            try:
                extension_record, socket_cutters = add_local_female_boolean_closure(
                    output_vertices=output_vertices,
                    output_faces=output_faces,
                    boundary_ids=[int(i) for i in loop],
                    boundary_points=source_boundary_points,
                    inward=inward,
                    spec=connector_spec,
                    inward_directions=inward_directions,
                    occupied_parent_edges=face_edges_among_vertices(
                        local_faces,
                        {int(index) for index in loop},
                        len(local_vertices),
                    ),
                    build_boolean_cutters=not defer_local_connector_boolean,
                )
            except ValueError as error:
                if os.environ.get("SPLIT3MF_FORCE_PREVIEW", "") != "1":
                    raise
                del output_vertices[preview_vertex_checkpoint:]
                del output_faces[preview_face_checkpoint:]
                socket_cutters = []
                extension_record = {
                    "status": "connector_loop_skipped_for_forced_preview",
                    "error": str(error),
                    "preview_not_printable": True,
                    "connector_side_faces_added": 0,
                    "backing_faces_added": 0,
                    "connector_cap_faces_added": 0,
                }
            local_socket_cutters.extend(socket_cutters)
            if defer_local_connector_boolean:
                deferred_local_connector_count += 1
            added_side = int(extension_record["connector_side_faces_added"])
            added_cap = int(
                extension_record["backing_faces_added"]
                + extension_record["connector_cap_faces_added"]
            )
            extension_record.update(
                {
                    **connector_safety,
                    "extension_min_mm": float(connector_spec.engagement_depth_mm),
                    "extension_max_mm": float(connector_spec.engagement_depth_mm),
                    "maximum_generated_inward_travel_mm": float(
                        connector_spec.full_boundary_backing_depth_mm
                        + connector_spec.engagement_depth_mm
                        + connector_spec.socket_bottom_clearance_mm
                    ),
                    "flat_bottom_required_max_extension_mm": float(
                        connector_spec.full_boundary_backing_depth_mm
                        + connector_spec.engagement_depth_mm
                        + connector_spec.socket_bottom_clearance_mm
                    ),
                    "target_exceeded": False,
                    "fixed_inward_depth_applied": True,
                    "flat_span_mm": 0.0,
                    "cap_mode": "local-connector",
                }
            )
        else:
            added_side, added_cap, extension_record = add_inward_lead_extrusion_and_cap(
                output_vertices=output_vertices,
                output_faces=output_faces,
                visible_top_ids=[int(i) for i in loop],
                visible_top_points=source_boundary_points,
                internal_fit_points=internal_socket_points,
                inward=inward,
                inward_directions=inward_directions,
                bottom_points=socket_bottom_points,
                lead_in_mm=lead_in_mm,
                max_extension_mm=max_extension_mm,
                flat_clearance_mm=body_flat_clearance_mm,
                cap_mode=loop_cap_mode,
                planar_extra_limit_mm=loop_planar_extra_limit_mm,
                plane_record=plane_record,
                parent_thickness_probe=parent_thickness_probe,
                lead_in_directions=inward_directions,
                cap_template_decision=shared_cap_decision,
            )
        side_faces += added_side
        cap_faces += added_cap
        extension_record["loop_index"] = loop_index
        extension_record["matched_cap_mode"] = loop_cap_mode
        extension_record["socket_overcut_mm"] = effective_socket_overcut_mm
        extension_record["bottom_clearance_mm"] = bottom_clearance_mm
        extension_record["matched_component_index"] = ref["component_index"] if ref else None
        extension_record["visible_top_source_preserved"] = False
        extension_record["visible_boundary_retopologized"] = True
        extension_record["interface_retopology_surface_role"] = (
            "shared_visible_boundary_and_surface_band"
        )
        extension_record["matched_color_code"] = ref["color_code"] if ref else None
        extension_record["matched_color_name"] = ref["color_name"] if ref else None
        loop_extension_records.append(extension_record)
        matched_refs.append(ref)

    mesh = trimesh.Trimesh(
        vertices=np.array(output_vertices, dtype=np.float64),
        faces=np.array(output_faces, dtype=np.int64),
        # The source body and every generated closure already share explicit
        # indices.  Generic welding can collapse distinct high-resolution
        # vendor boundary samples and reopen an otherwise closed parent just
        # before boolean subtraction.
        process=False,
        metadata={"name": part_id},
    )
    if local_socket_cutters and not defer_local_connector_boolean:
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
        try:
            mesh, boolean_record = subtract_socket_cutters(
                mesh,
                local_socket_cutters,
                cleanup_volume_envelope_cap_mm3=(
                    5e-5
                    if interface_retopology.config.surface_band_validation
                    == "advisory"
                    else 1e-5
                ),
                topology_healthy_export_volume_envelope_cap_mm3=(
                    5e-3
                    if interface_retopology.config.surface_band_validation
                    == "advisory"
                    else 0.0
                ),
            )
            mesh = finalize_boolean_difference_mesh(mesh)
            boolean_record["post_boolean_finalize_policy"] = str(
                mesh.metadata["boolean_finalize_policy"]
            )
            boolean_record["post_boolean_faces"] = int(len(mesh.faces))
            boolean_record["post_boolean_vertices"] = int(len(mesh.vertices))
            boolean_record["post_boolean_watertight"] = bool(mesh.is_watertight)
            boolean_record["post_boolean_winding_consistent"] = bool(
                mesh.is_winding_consistent
            )
            local_socket_boolean_records.append(boolean_record)
            mesh.metadata["name"] = part_id
        except ValueError as error:
            if os.environ.get("SPLIT3MF_FORCE_PREVIEW", "") != "1":
                raise
            # Manual-inspection escape hatch only.  Keep the pre-Boolean parent
            # so a failed first recursive layer can still be serialized for
            # visual diagnosis.  Normal and printable runs remain strict.
            local_socket_boolean_records.append(
                {
                    "status": "skipped_for_forced_preview",
                    "error": str(error),
                    "cutter_count": int(len(local_socket_cutters)),
                    "preview_not_printable": True,
                }
            )
            mesh.metadata["name"] = part_id
            mesh.metadata["preview_not_printable"] = True
    else:
        mesh = finalize_source_preserving_mesh(
            mesh,
            protected_source_face_count=len(local_faces),
        )
        if deferred_local_connector_count:
            local_socket_boolean_records.append(
                {
                    "status": "deferred_to_complete_emitted_direct_child_solids",
                    "cutter_count": 0,
                    "planned_interface_count": int(deferred_local_connector_count),
                    "discarded_cutter_construction_skipped": True,
                    "reason": "single_authoritative_boolean_geometry_path",
                }
            )

    color_info = COLOR_INFO.get(body_component.color_code, {"name": body_component.color_code, "hex": "", "rgba": [200, 200, 200, 255]})
    mesh.visual.face_colors = np.tile(np.array(color_info["rgba"], dtype=np.uint8), (len(mesh.faces), 1))
    bbox = mesh.bounds
    selected_cap_modes = sorted({str(record.get("cap_mode", cap_mode)) for record in loop_extension_records})
    geometry_cap_mode = selected_cap_modes[0] if len(selected_cap_modes) == 1 else "mixed"
    stats = {
        "part_id": part_id,
        "processing_mode": "body_cut_from_adjacent_insert_boundaries",
        "interface_geometry": str(interface_geometry),
        "selected_processing_mode": "body",
        "color_code": body_component.color_code,
        "color_name": color_info["name"],
        "color_hex": color_info["hex"],
        "source_faces": body_component.face_count,
        "output_faces": int(len(mesh.faces)),
        "output_vertices": int(len(mesh.vertices)),
        "boundary_loops": len(loops),
        "boundary_vertices": boundary_vertex_count,
        "interface_retopology_records": interface_retopology_records,
        "preserve_unmatched_source_geometry": bool(
            preserve_unmatched_source_geometry
        ),
        "immutable_source_loop_count": int(len(immutable_source_loop_records)),
        "immutable_source_loops": immutable_source_loop_records,
        "processed_body_cut_loop_indices": mutable_loop_indices,
        "side_faces_added": side_faces,
        "cap_faces_added": cap_faces,
        "mesh_finalization": copy.deepcopy(
            mesh.metadata.get("source_preserving_finalization", {})
        ),
        "max_extension_limit_mm": max_extension_mm,
        "cap_mode": cap_mode,
        "geometry_cap_mode": geometry_cap_mode,
        "planar_extra_limit_mm": planar_extra_limit_mm,
        "selected_cap_modes": selected_cap_modes,
        "flat_clearance_mm": flat_clearance_mm,
        "clearance_mode": clearance_mode,
        "fit_clearance_mm": max(float(fit_clearance_mm), 0.0),
        "insert_shrink_mm": 0.0,
        "socket_overcut_mm": socket_overcut_mm,
        "bottom_clearance_mm": bottom_clearance_mm,
        "body_flat_clearance_mm": body_flat_clearance_mm,
        "lead_in_mm": lead_in_mm,
        "top_edge_clearance_mm": 0.0,
        "visible_top_source_preserved": False,
        "visible_boundary_retopologized": True,
        "interface_retopology_surface_role": "shared_visible_boundary_and_surface_band",
        "local_inward_direction_records": applied_direction_records,
        "local_inward_outward_vertices_before": int(
            sum(record.get("global_outward_vertices_before", 0) for record in applied_direction_records)
        ),
        "local_inward_outward_vertices_after": int(
            sum(record.get("outward_vertices_after", 0) for record in applied_direction_records)
        ),
        "extension_max_observed_mm": float(max((r["extension_max_mm"] for r in loop_extension_records), default=0.0)),
        "maximum_generated_inward_travel_mm": float(max((r["extension_max_mm"] for r in loop_extension_records), default=0.0)),
        "extension_min_observed_mm": float(min((r["extension_min_mm"] for r in loop_extension_records), default=0.0)),
        "flat_bottom_required_max_extension_mm": float(max((r["flat_bottom_required_max_extension_mm"] for r in loop_extension_records), default=0.0)),
        "flat_bottom_target_exceeded": bool(any(r["target_exceeded"] for r in loop_extension_records)),
        "fixed_inward_depth_mm": max_extension_mm,
        "boundary_span_max_mm": float(max((r["flat_span_mm"] for r in loop_extension_records), default=0.0)),
        "fixed_inward_depth_applied": bool(all(r.get("fixed_inward_depth_applied", False) for r in loop_extension_records)) if loop_extension_records else True,
        "child_socket_count": len(loop_extension_records),
        "child_socket_extensions": loop_extension_records,
        "loop_extensions": loop_extension_records,
        "local_socket_boolean_records": local_socket_boolean_records,
        "local_connector_boolean_deferred": bool(
            deferred_local_connector_count and defer_local_connector_boolean
        ),
        "bbox_min": bbox[0].round(6).tolist(),
        "bbox_max": bbox[1].round(6).tolist(),
        "bbox_size_mm": (bbox[1] - bbox[0]).round(6).tolist(),
        **mesh_runtime_stats(mesh),
    }
    return mesh, stats

def make_layer_child_subassembly_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_colors: list[str],
    components: list[Component],
    component: Component,
    root_child_index: int,
    subtree_indices: list[int],
    parent_index: int,
    part_id: str,
    max_extension_mm: float,
    interface_retopology: PlanarArcRetopologyContext,
    flat_clearance_mm: float,
    fit_clearance_mm: float,
    insert_shrink_mm: float,
    lead_in_mm: float,
    top_edge_clearance_mm: float,
    boundary_neighbor_lookup: dict[tuple[int, int], set[int]],
    inward_override: np.ndarray | None,
    model_center: np.ndarray,
    cap_mode: str,
    planar_extra_limit_mm: float,
    cap_decisions_by_loop: dict[int, CapDecision] | None = None,
    bottom_clearance_mm: float = 0.0,
    interface_geometry: str = "boundary-extrusion",
) -> tuple[trimesh.Trimesh, dict]:
    cap_decisions_by_loop = cap_decisions_by_loop or {}
    parent_thickness_probe = ParentThicknessProbe(vertices, faces).excluding_triangles(
        np.asarray(component.global_faces, dtype=np.int64),
        filter_context={
            "parent_part_index": int(parent_index),
            "child_part_index": int(root_child_index),
            "excluded_subtree_part_indices": [
                int(index) for index in subtree_indices
            ],
            "final_child_build_probe": True,
        },
    )
    local_vertices, local_faces, _global_to_local, global_vertex_ids = build_local_mesh(vertices, faces, component)
    loops = boundary_loops(local_faces)
    root_component = components[root_child_index - 1]
    current_layer_component_set = {int(index) for index in subtree_indices}
    root_center = local_vertices[local_faces.reshape(-1)].mean(axis=0)
    inward = inward_override
    if inward is None:
        inward = component_inward_direction(vertices, faces, root_component, model_center)
    inward = inward / max(float(np.linalg.norm(inward)), 1e-12)
    root_vertex_inward_normals = mesh_vertex_inward_normals(local_vertices, local_faces)

    selected_loop_records = []
    ignored_loop_records = []
    for loop_index, root_loop in enumerate(loops):
        contact = boundary_loop_parent_contact(
            root_loop,
            global_vertex_ids,
            boundary_neighbor_lookup,
            parent_index,
            current_layer_component_set,
        )
        mapped_loop = [int(local_index) for local_index in root_loop]
        record = {
            "loop_index": int(loop_index),
            "vertices": int(len(mapped_loop)),
            "contact_source_component_index": int(root_child_index),
            **contact,
        }
        loop_directions, direction_record = safe_boundary_inward_directions(
            root_vertex_inward_normals,
            root_loop,
            inward,
            loop_points=local_vertices[np.asarray(root_loop, dtype=np.int64)],
        )
        if int(contact["parent_edges"]) >= 3:
            selected_loop_records.append(
                {
                    **record,
                    "loop": [int(value) for value in mapped_loop],
                    "global_loop": [
                        int(global_vertex_ids[int(local_index)])
                        for local_index in root_loop
                    ],
                    "inward_directions": loop_directions,
                    "lead_in_directions": loop_directions.copy(),
                    "local_inward_direction_record": direction_record,
                }
            )
        else:
            ignored_loop_records.append(record)

    from .micro_interfaces import filter_micro_interface_loops
    selected_loop_records, micro_loops = filter_micro_interface_loops(
        local_vertices, selected_loop_records, part_index=root_child_index)
    ignored_loop_records.extend(micro_loops)
    selected_loops = [record["loop"] for record in selected_loop_records]
    visible_source_vertices = np.asarray(local_vertices, dtype=np.float64).copy()
    retopology_vertices, interface_retopology_records = InterfaceRetopologyService.retopologize_local_loops(
        visible_source_vertices.copy(),
        selected_loops,
        global_vertex_ids,
        interface_retopology,
        local_faces=local_faces,
    )
    visible_source_vertices = np.asarray(
        retopology_vertices, dtype=np.float64
    ).copy()
    for retopology_record, selected_record in zip(interface_retopology_records, selected_loop_records):
        retopology_record["loop_index"] = int(selected_record["loop_index"])
    hidden_geometry_vertices = generated_geometry_boundary_vertices(
        visible_source_vertices,
        selected_loops,
        interface_retopology_records,
    )

    u, v = orthonormal_basis(inward)

    output_vertices: list[np.ndarray] = [point.copy() for point in visible_source_vertices]
    output_faces: list[list[int]] = local_faces.astype(int).tolist()
    color_info = COLOR_INFO.get(component.color_code, {"name": component.color_code, "hex": "", "rgba": [200, 200, 200, 255]})
    source_face_codes = [
        str(source_colors[int(global_face)])
        for global_face in component.global_faces
    ]
    for record in selected_loop_records:
        record["source_edge_color_codes"] = boundary_loop_source_color_codes(
            local_faces,
            source_face_codes,
            record["loop"],
            fallback_color_code=str(root_component.color_code),
        )
    loop_extension_records = []
    insert_loop_records = []
    loops_lead: list[list[int]] = []
    loops_bottom: list[list[int]] = []
    bottom_loop_points_2d: list[np.ndarray] = []
    local_generated_face_color_codes: list[str] = []
    local_connector_records: list[dict] = []
    runtime_local_connector_cutters: list[trimesh.Trimesh] = []

    insert_shrink_mm = max(float(insert_shrink_mm), 0.0)
    lead_in_mm = max(float(lead_in_mm), 0.0)
    requested_top_edge_clearance_mm = min(
        max(float(top_edge_clearance_mm), 0.0),
        insert_shrink_mm,
    )
    top_edge_clearance_mm = visible_top_edge_clearance(insert_shrink_mm)

    for record in selected_loop_records:
        loop = record["loop"]
        loop_array = np.array(loop, dtype=np.int64)
        source_boundary_points = visible_source_vertices[loop_array]
        cap_decision = cap_decisions_by_loop.get(int(record["loop_index"]))
        internal_boundary_points = hidden_geometry_vertices[loop_array]
        loop_interior_conormals = boundary_loop_interior_conormals(
            hidden_geometry_vertices,
            local_faces,
            loop,
            reference_axis=inward,
        )
        top_points = source_boundary_points.copy()
        fit_points = offset_points_along_conormals(
            internal_boundary_points,
            loop_interior_conormals,
            insert_shrink_mm,
        )
        inward_directions = record["inward_directions"]
        lead_in_directions = record["lead_in_directions"]
        if cap_decision is None:
            distances, inward_directions, plane_record = boundary_cap_distances(
                fit_points,
                inward,
                inward_directions,
                max_extension_mm,
                flat_clearance_mm,
                cap_mode,
                planar_extra_limit_mm,
                parent_thickness_probe,
            )
        else:
            (
                planned_fit_points,
                distances,
                inward_directions,
                plane_record,
            ) = remap_cap_decision(cap_decision, record["global_loop"])
            fit_points = reconcile_planned_internal_fit_points(
                fit_points,
                planned_fit_points,
                internal_boundary_points,
                loop_interior_conormals,
                insert_shrink_mm,
                plane_record,
                mismatch_label=(
                    f"P{int(root_child_index):02d} boundary fit points changed "
                    f"after cap planning on loop {int(record['loop_index'])}"
                ),
                allow_user_reviewed_reconciliation=(
                    interface_retopology.config.surface_band_validation
                    == "advisory"
                ),
                maximum_reconciliation_error_mm=(
                    interface_retopology.config.maximum_safe_target_offset_mm
                ),
            )
        if interface_geometry == "local-connector":
            precomputed_connector_safety = (
                cap_decision.record.get("local_connector_safety")
                if cap_decision is not None
                else None
            )
            connector_safety = (
                dict(precomputed_connector_safety)
                if isinstance(precomputed_connector_safety, dict)
                else local_connector_safe_depth_at_boundary(
                    boundary_points=source_boundary_points,
                    inward=inward,
                    fit_clearance_mm=fit_clearance_mm,
                    bottom_clearance_mm=bottom_clearance_mm,
                    lead_in_mm=lead_in_mm,
                    boundary_distances=np.asarray(
                        distances,
                        dtype=np.float64,
                    ),
                    parent_thickness_probe=parent_thickness_probe,
                )
            )
            connector_spec = local_connector_spec_for_interface(
                fit_clearance_mm=fit_clearance_mm,
                bottom_clearance_mm=bottom_clearance_mm,
                lead_in_mm=lead_in_mm,
                safe_engagement_depth_mm=float(
                    connector_safety["local_connector_safety_budget_mm"]
                ),
                safe_backing_depth_mm=float(
                    connector_safety.get(
                        "local_connector_backing_safety_limit_mm",
                        connector_safety["boundary_safety_minimum_mm"],
                    )
                ),
                compact_peg_supported=bool(
                    connector_safety.get(
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
                visible_interface_simplification_tolerance=float(
                    interface_retopology.config.visible_interface_simplification_tolerance
                ),
            )
            generated_face_start = len(output_faces)
            generated_vertex_start = len(output_vertices)
            try:
                connector_record = add_local_male_connector_and_backing(
                    output_vertices=output_vertices,
                    output_faces=output_faces,
                    boundary_ids=[int(value) for value in loop],
                    boundary_points=source_boundary_points,
                    inward=inward,
                    spec=connector_spec,
                    inward_directions=inward_directions,
                    boolean_cutters=runtime_local_connector_cutters,
                )
            except ValueError as error:
                if os.environ.get("SPLIT3MF_FORCE_PREVIEW", "") != "1":
                    raise
                del output_vertices[generated_vertex_start:]
                del output_faces[generated_face_start:]
                connector_record = {
                    "status": "connector_loop_skipped_for_forced_preview",
                    "error": str(error),
                    "preview_not_printable": True,
                    "backing_faces_added": 0,
                    "connector_side_faces_added": 0,
                    "connector_cap_faces_added": 0,
                }
            generated_count = int(len(output_faces) - generated_face_start)
            local_generated_face_color_codes.extend(
                local_connector_face_color_codes_from_boundary(
                    output_vertices,
                    output_faces[generated_face_start:],
                    source_boundary_points,
                    record["source_edge_color_codes"],
                    backing_face_count=int(connector_record["backing_faces_added"]),
                    fallback_color_code=str(root_component.color_code),
                )
            )
            if generated_count != (
                int(connector_record["backing_faces_added"])
                + int(connector_record["connector_side_faces_added"])
                + int(connector_record["connector_cap_faces_added"])
            ):
                raise ValueError("local connector face accounting does not match generated geometry")
            connector_record.update(
                {
                    **connector_safety,
                    "loop_index": int(record["loop_index"]),
                    "vertices": int(len(loop)),
                    "parent_edges": int(record["parent_edges"]),
                    "extension_min_mm": float(connector_spec.engagement_depth_mm),
                    "extension_max_mm": float(connector_spec.engagement_depth_mm),
                    "maximum_generated_inward_travel_mm": float(
                        connector_spec.full_boundary_backing_depth_mm
                        + connector_spec.engagement_depth_mm
                    ),
                    "flat_bottom_required_max_extension_mm": float(
                        connector_spec.full_boundary_backing_depth_mm
                        + connector_spec.engagement_depth_mm
                    ),
                    "target_exceeded": False,
                    "fixed_inward_depth_applied": True,
                    "flat_span_mm": 0.0,
                    "cap_mode": "local-connector",
                    "lead_in_applied": True,
                    "visible_top_source_preserved": False,
                    "visible_boundary_retopologized": True,
                    "interface_retopology_surface_role": "shared_visible_boundary_and_surface_band",
                    "interface_retopology_target_preserved": True,
                }
            )
            local_connector_records.append(connector_record)
            loop_extension_records.append(connector_record)
            continue
        lead_ids = []
        lead_taper_record: dict = {}
        effective_insert_shrink_mm = float(
            plane_record.get(
                "thin_boundary_effective_insert_shrink_mm",
                insert_shrink_mm,
            )
        )
        if (
            lead_in_mm > 1e-9
            and effective_insert_shrink_mm > top_edge_clearance_mm + 1e-9
        ):
            if bool(plane_record.get("guided_internal_cut_applied", False)):
                bottom_points = fit_points + inward_directions * distances[:, None]
                lead_points = (top_points + bottom_points) * 0.5
                lead_taper_record = {
                    "lead_profile": "guided_internal_midpoint_loft",
                    "guided_internal_midpoint_fraction": 0.5,
                }
            else:
                lead_points, lead_depths, effective_lead_directions, lead_taper_record = coherent_tapered_lead_ring(
                    top_points,
                    fit_points,
                    lead_in_directions,
                    distances,
                    lead_in_mm,
                    reference_axis=inward,
                )
            for lead_point in lead_points:
                lead_ids.append(len(output_vertices))
                output_vertices.append(lead_point)
        bottom_ids = []
        for fit_point, direction, distance in zip(fit_points, inward_directions, distances):
            bottom_ids.append(len(output_vertices))
            output_vertices.append(fit_point + direction * float(distance))
        loops_lead.append(lead_ids)
        loops_bottom.append(bottom_ids)
        bottom_loop_points_2d.append(project_points(np.array([output_vertices[i] for i in bottom_ids]), root_center, u, v))
        insert_loop_records.append(
            {
                "loop": loop,
                "top_points": top_points,
                "cap_decision": cap_decision,
                "source_edge_color_codes": list(
                    record["source_edge_color_codes"]
                ),
            }
        )
        loop_extension_records.append(
            {
                "loop_index": int(record["loop_index"]),
                "vertices": int(len(loop)),
                "parent_edges": int(record["parent_edges"]),
                "extension_min_mm": float(distances.min()),
                "extension_max_mm": float(distances.max()),
                "lead_in_applied": bool(lead_ids),
                "visible_top_source_preserved": False,
                "visible_boundary_retopologized": True,
                "interface_retopology_surface_role": "shared_visible_boundary_and_surface_band",
                "interface_retopology_target_preserved": True,
                **lead_taper_record,
                **plane_record,
            }
        )

    side_faces = 0
    generated_side_color_codes: list[str] = []
    for record, lead_ids, bottom_ids in zip(insert_loop_records, loops_lead, loops_bottom):
        top_ring = record["loop"]
        edge_color_codes = record["source_edge_color_codes"]
        if lead_ids:
            side_faces += add_side_faces_between_rings(
                output_faces, top_ring, lead_ids, vertices=output_vertices
            )
            generated_side_color_codes.extend(
                side_face_color_codes(edge_color_codes)
            )
            side_faces += add_side_faces_between_rings(
                output_faces, lead_ids, bottom_ids, vertices=output_vertices
            )
            generated_side_color_codes.extend(
                side_face_color_codes(edge_color_codes)
            )
        else:
            side_faces += add_side_faces_between_rings(
                output_faces, top_ring, bottom_ids, vertices=output_vertices
            )
            generated_side_color_codes.extend(
                side_face_color_codes(edge_color_codes)
            )

    loop_groups = group_loops(bottom_loop_points_2d) if bottom_loop_points_2d else []
    cap_face_start = len(output_faces)
    harmonic_records = [
        record
        for record in insert_loop_records
        if record["cap_decision"] is not None
        and record["cap_decision"].cap_template_points is not None
    ]
    if harmonic_records:
        if len(harmonic_records) != len(insert_loop_records):
            raise ValueError(
                "mixed harmonic and polygon cap strategies are not supported in one part"
            )
        if any(len(group) != 1 for group in loop_groups):
            raise ValueError(
                "harmonic source-patch caps require independent disk loops"
            )
        cap_faces = 0
        for record, bottom_ids in zip(insert_loop_records, loops_bottom):
            decision = record["cap_decision"]
            added, harmonic_quality = append_harmonic_cap_template(
                output_vertices,
                output_faces,
                np.asarray(decision.cap_template_points, dtype=np.float64),
                np.asarray(decision.cap_template_faces, dtype=np.int64),
                decision.cap_template_boundary_ids,
                bottom_ids,
                target_boundary_points=np.asarray(
                    [output_vertices[index] for index in bottom_ids],
                    dtype=np.float64,
                ),
            )
            if not bool(harmonic_quality["valid"]):
                raise ValueError(
                    "authoritative harmonic insert cap failed: "
                    + json.dumps(harmonic_quality, ensure_ascii=False, sort_keys=True)
                )
            cap_faces += int(added)
    else:
        cap_faces = triangulate_cap(
            output_vertices,
            output_faces,
            loops_bottom,
            bottom_loop_points_2d,
            loop_groups,
            inward,
        )
    generated_cap_color_codes = cap_face_color_codes_from_boundary(
        output_faces[cap_face_start:],
        loops_bottom,
        [record["source_edge_color_codes"] for record in insert_loop_records],
        fallback_color_code=str(root_component.color_code),
    )
    added_face_count = max(0, len(output_faces) - len(source_face_codes))
    generated_face_color_codes = (
        local_generated_face_color_codes
        + generated_side_color_codes
        + generated_cap_color_codes
    )
    if len(generated_face_color_codes) != added_face_count:
        raise ValueError(
            "generated recursive face-color count does not match added geometry"
        )
    face_color_codes = source_face_codes + generated_face_color_codes

    mesh = trimesh.Trimesh(
        vertices=np.array(output_vertices, dtype=np.float64),
        faces=np.array(output_faces, dtype=np.int64),
        process=False,
        metadata={"name": part_id},
    )
    face_filament_slots = [
        COLOR_INFO.get(code, {}).get("filament_slot")
        for code in face_color_codes
    ]
    mesh = finalize_recursive_colored_mesh(
        mesh,
        face_color_codes,
        face_filament_slots,
        str(root_component.color_code),
        protected_source_face_count=len(source_face_codes),
    )

    backing_validation = None
    if interface_geometry == 'local-connector' and selected_loop_records:
        from .backing_repair import ensure_backing
        source_patch = trimesh.Trimesh(visible_source_vertices, local_faces, process=False)
        complete_parent = trimesh.Trimesh(vertices, faces, process=False)
        mesh, backing_validation, repaired_codes = ensure_backing(
            source_patch, mesh, complete_parent, source_face_codes, inward, part_id=part_id)
        if repaired_codes is not None:
            generated_face_color_codes = repaired_codes[len(source_face_codes):]
            mesh = finalize_recursive_colored_mesh(
                mesh, repaired_codes,
                [COLOR_INFO.get(code, {}).get('filament_slot') for code in repaired_codes],
                str(root_component.color_code), protected_source_face_count=len(source_face_codes))
            from .backing_thickness import audit_local_backing
            from .print_tolerance import current as current_print_tolerance
            np.testing.assert_array_equal(mesh.triangles[:len(local_faces)], source_patch.triangles)
            backing_validation['after_finalization'] = audit_local_backing(
                source_patch, mesh, scale_factor=current_print_tolerance().backing_validation_scale,
                area_budget_mm2=current_print_tolerance().micro_area_mm2)
            if not backing_validation['after_finalization']['valid']:
                raise ValueError(f'{part_id}: finalized backing failed local thickness')
            # Exact parent subtraction consumes the complete new shell.
            runtime_local_connector_cutters = []
            local_connector_records = []
            side_faces, cap_faces = 0, len(mesh.faces)-len(source_face_codes)

    bbox = mesh.bounds
    selected_cap_modes = sorted({str(record.get("cap_mode", cap_mode)) for record in loop_extension_records})
    geometry_cap_mode = selected_cap_modes[0] if len(selected_cap_modes) == 1 else "mixed"
    local_side_faces = int(
        sum(record["connector_side_faces_added"] for record in local_connector_records)
    )
    local_cap_faces = int(
        sum(
            record["backing_faces_added"] + record["connector_cap_faces_added"]
            for record in local_connector_records
        )
    )
    stats = {
        "part_id": part_id,
        "processing_mode": "layer_subassembly_parent_contact_only",
        "interface_geometry": str(interface_geometry),
        "local_connector_count": int(len(local_connector_records)),
        "local_connectors": local_connector_records,
        "actual_backing_validation": backing_validation,
        "color_code": component.color_code,
        "color_name": color_info["name"],
        "color_hex": color_info["hex"],
        "source_faces": component.face_count,
        "output_faces": int(len(mesh.faces)),
        "output_vertices": int(len(mesh.vertices)),
        "mesh_finalization": copy.deepcopy(
            mesh.metadata.get("source_preserving_finalization", {})
        ),
        "boundary_loops": len(loops),
        "boundary_loops_total": len(loops),
        "union_boundary_loops_total": len(loops),
        "root_boundary_loops_total": len(loops),
        "parent_contact_source_component_index": int(root_child_index),
        "selected_parent_contact_loops": len(selected_loop_records),
        "generated_face_color_source": "local_parent_contact_boundary",
        "generated_face_color_counts": dict(
            sorted(collections.Counter(generated_face_color_codes).items())
        ),
        "interface_retopology_records": interface_retopology_records,
        "ignored_non_parent_loops": len(ignored_loop_records),
        "parent_contact_edges": int(sum(record["parent_edges"] for record in selected_loop_records)),
        "selected_loop_summary": [
            {
                key: value
                for key, value in record.items()
                if key not in {"loop", "global_loop", "inward_directions"}
            }
            for record in selected_loop_records
        ],
        "ignored_loop_summary": ignored_loop_records,
        "side_faces_added": side_faces + local_side_faces,
        "cap_faces_added": cap_faces + local_cap_faces,
        "child_socket_count": 0,
        "loop_groups": loop_groups,
        "max_extension_limit_mm": max_extension_mm,
        "cap_mode": cap_mode,
        "geometry_cap_mode": geometry_cap_mode,
        "planar_extra_limit_mm": planar_extra_limit_mm,
        "selected_cap_modes": selected_cap_modes,
        "shared_prevalidated_cap_decision_loops": sorted(
            int(loop_index) for loop_index in cap_decisions_by_loop
        ),
        "flat_clearance_mm": flat_clearance_mm,
        "fit_clearance_mm": max(float(fit_clearance_mm), 0.0),
        "insert_shrink_mm": insert_shrink_mm,
        "socket_overcut_mm": 0.0,
        "bottom_clearance_mm": 0.0,
        "lead_in_mm": lead_in_mm,
        "requested_top_edge_clearance_mm": requested_top_edge_clearance_mm,
        "top_edge_clearance_mm": top_edge_clearance_mm,
        "visible_top_source_preserved": False,
        "visible_boundary_retopologized": True,
        "interface_retopology_surface_role": "shared_visible_boundary_and_surface_band",
        "inward_direction": inward.round(6).tolist(),
        "local_inward_direction_records": [
            {"loop_index": int(record["loop_index"]), **record["local_inward_direction_record"]}
            for record in selected_loop_records
        ],
        "local_inward_outward_vertices_before": int(
            sum(
                record["local_inward_direction_record"]["global_outward_vertices_before"]
                for record in selected_loop_records
            )
        ),
        "local_inward_outward_vertices_after": int(
            sum(
                record["local_inward_direction_record"]["outward_vertices_after"]
                for record in selected_loop_records
            )
        ),
        "extension_max_observed_mm": float(max((record["extension_max_mm"] for record in loop_extension_records), default=0.0)),
        "maximum_generated_inward_travel_mm": float(
            max(
                (
                    record.get(
                        "maximum_generated_inward_travel_mm",
                        record["extension_max_mm"],
                    )
                    for record in loop_extension_records
                ),
                default=0.0,
            )
        ),
        "extension_min_observed_mm": float(min((record["extension_min_mm"] for record in loop_extension_records), default=0.0)),
        "boundary_span_max_mm": float(max((record["flat_span_mm"] for record in loop_extension_records), default=0.0)),
        "fixed_inward_depth_mm": max_extension_mm,
        "fixed_inward_depth_applied": bool(all(record.get("fixed_inward_depth_applied", False) for record in loop_extension_records)) if loop_extension_records else True,
        "loop_extensions": loop_extension_records,
        "bbox_min": bbox[0].round(6).tolist(),
        "bbox_max": bbox[1].round(6).tolist(),
        "bbox_size_mm": (bbox[1] - bbox[0]).round(6).tolist(),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "open_edges": open_edge_count(mesh),
        "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
        "root_child_index": int(root_child_index),
        "parent_index": int(parent_index),
        "contains": [int(index) for index in subtree_indices],
        # Runtime-only geometry consumed immediately by the recursive parent
        # Boolean.  The caller removes this private key before serializing
        # annotations or standalone 3MF state.
        "_local_connector_boolean_cutters": runtime_local_connector_cutters,
    }
    if backing_validation and backing_validation['repaired']:
        geometry = backing_validation['construction']
        stats.update(geometry_cap_mode='source_following_local_normals',
                     selected_cap_modes=['source_following_local_normals'],
                     extension_min_observed_mm=geometry['minimum_interior_depth_mm'],
                     extension_max_observed_mm=geometry['maximum_depth_mm'],
                     maximum_generated_inward_travel_mm=geometry['maximum_depth_mm'],
                     fixed_inward_depth_applied=False,
                     generated_face_color_source='source_following_triangle_owner',
                     loop_extension_records_role='superseded_plan_before_backing_repair')
    return mesh, stats
