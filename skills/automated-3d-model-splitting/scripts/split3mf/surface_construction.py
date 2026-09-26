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

def add_side_faces_between_rings(
    output_faces: list[list[int]],
    first_ids: list[int],
    second_ids: list[int],
    *,
    vertices: list[np.ndarray] | np.ndarray | None = None,
) -> int:
    """Triangulate a ring strip with the smoother diagonal for each quad.

    Fixed diagonals turn a mildly twisted retopology lead-in into an alternating
    row of bright and dark triangles in flat-shaded slicers.  When coordinates
    are available, choose the diagonal whose two triangle normals agree most
    closely, then prefer the healthier minimum triangle area and shorter
    diagonal.  The fallback preserves the historical winding and topology.
    """
    if len(first_ids) != len(second_ids):
        raise ValueError("Side rings must have the same vertex count")
    side_faces = 0
    count = len(first_ids)
    for index in range(count):
        a0 = int(first_ids[index])
        a1 = int(first_ids[(index + 1) % count])
        b0 = int(second_ids[index])
        b1 = int(second_ids[(index + 1) % count])
        first_split = ([a0, a1, b1], [a0, b1, b0])
        second_split = ([a0, a1, b0], [a1, b1, b0])
        selected = first_split
        if vertices is not None:
            point_array = np.asarray(vertices, dtype=np.float64)
            selected_index, _score, _areas = side_quad_triangulation_choice(
                point_array[a0],
                point_array[a1],
                point_array[b0],
                point_array[b1],
            )
            if selected_index == 1:
                selected = second_split
        output_faces.extend([list(selected[0]), list(selected[1])])
        side_faces += 2
    return side_faces

def boundary_loop_source_color_codes(
    local_faces: np.ndarray,
    source_face_codes: list[str],
    loop: list[int],
    fallback_color_code: str,
) -> list[str]:
    """Resolve each generated wall segment from its local source boundary.

    A recursive pending subassembly can contain several materials even though
    its wrapper has one default/root color.  Every loop edge has one owning
    source face on a manifold cut boundary; use that face instead of painting
    all generated geometry with the wrapper default.
    """
    face_array = np.asarray(local_faces, dtype=np.int64)
    if len(source_face_codes) != len(face_array):
        raise ValueError("source face colors do not match local faces")
    edge_codes: dict[tuple[int, int], list[str]] = collections.defaultdict(list)
    for face, code in zip(face_array, source_face_codes):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_codes[tuple(sorted((int(a), int(b))))].append(str(code))

    result = []
    for position, first in enumerate(loop):
        second = int(loop[(position + 1) % len(loop)])
        candidates = edge_codes.get(
            tuple(sorted((int(first), second))),
            [],
        )
        if not candidates:
            result.append(str(fallback_color_code))
            continue
        counts = collections.Counter(candidates)
        result.append(
            min(counts, key=lambda code: (-int(counts[code]), str(code)))
        )
    return result

def side_face_color_codes(edge_color_codes: list[str]) -> list[str]:
    return [
        str(code)
        for edge_code in edge_color_codes
        for code in (edge_code, edge_code)
    ]

def cap_face_color_codes_from_boundary(
    cap_faces: list[list[int]],
    loops_bottom: list[list[int]],
    loop_edge_color_codes: list[list[str]],
    fallback_color_code: str,
) -> list[str]:
    """Assign cap triangles from the nearest represented boundary material.

    Boundary-only triangulation reuses bottom-ring vertices.  Give each bottom
    vertex the majority material of its two incident boundary edges, then vote
    across the three vertices of every cap triangle.  Uniform loops remain
    uniform; mixed loops retain their local material regions instead of falling
    back to the pending wrapper color.
    """
    vertex_codes: dict[int, str] = {}
    for ring, edge_codes in zip(loops_bottom, loop_edge_color_codes):
        if len(ring) != len(edge_codes):
            raise ValueError("cap boundary color count does not match its ring")
        for position, vertex_id in enumerate(ring):
            candidates = [
                str(edge_codes[position - 1]),
                str(edge_codes[position]),
            ]
            counts = collections.Counter(candidates)
            vertex_codes[int(vertex_id)] = min(
                counts,
                key=lambda code: (-int(counts[code]), str(code)),
            )

    result = []
    for face in cap_faces:
        candidates = [
            vertex_codes[int(vertex_id)]
            for vertex_id in face
            if int(vertex_id) in vertex_codes
        ]
        if not candidates:
            result.append(str(fallback_color_code))
            continue
        counts = collections.Counter(candidates)
        result.append(
            min(counts, key=lambda code: (-int(counts[code]), str(code)))
        )
    return result

def local_connector_face_color_codes_from_boundary(
    output_vertices: list[np.ndarray] | np.ndarray,
    generated_faces: list[list[int]] | np.ndarray,
    boundary_points: np.ndarray,
    edge_color_codes: list[str],
    *,
    backing_face_count: int,
    fallback_color_code: str,
) -> list[str]:
    """Color connector backing from its boundary and the compact peg uniformly.

    The full-boundary backing belongs to the source faces incident to the cut
    loop.  Resolve each backing triangle from the nearest boundary vertex,
    whose material is the majority of its two incident loop edges.  The compact
    central peg is intentionally one printable feature, so give it the majority
    material of the complete local boundary instead of creating striped side
    walls or falling back to a multicolor wrapper's default material.
    """
    vertices = np.asarray(output_vertices, dtype=np.float64)
    faces = np.asarray(generated_faces, dtype=np.int64)
    boundary = np.asarray(boundary_points, dtype=np.float64)
    if len(boundary) != len(edge_color_codes):
        raise ValueError("local connector boundary colors do not match its points")
    if len(faces) == 0:
        return []
    backing_count = int(backing_face_count)
    if backing_count < 0 or backing_count > len(faces):
        raise ValueError("local connector backing face count is out of range")

    vertex_codes: list[str] = []
    for position in range(len(boundary)):
        candidates = [
            str(edge_color_codes[position - 1]),
            str(edge_color_codes[position]),
        ]
        counts = collections.Counter(candidates)
        vertex_codes.append(
            min(counts, key=lambda code: (-int(counts[code]), str(code)))
        )
    boundary_counts = collections.Counter(str(code) for code in edge_color_codes)
    connector_code = (
        min(
            boundary_counts,
            key=lambda code: (-int(boundary_counts[code]), str(code)),
        )
        if boundary_counts
        else str(fallback_color_code)
    )

    result: list[str] = []
    if backing_count:
        centroids = vertices[faces[:backing_count]].mean(axis=1)
        # Bound peak memory for vendor loops with thousands of source samples.
        for start in range(0, len(centroids), 256):
            block = centroids[start : start + 256]
            squared = np.sum(
                (block[:, None, :] - boundary[None, :, :]) ** 2,
                axis=2,
            )
            nearest = np.argmin(squared, axis=1)
            result.extend(vertex_codes[int(index)] for index in nearest)
    result.extend([connector_code] * (len(faces) - backing_count))
    return result

def add_loop_extrusion_and_cap(
    output_vertices: list[np.ndarray],
    output_faces: list[list[int]],
    top_ids: list[int],
    top_points: np.ndarray,
    inward: np.ndarray,
    origin: np.ndarray,
    max_extension_mm: float,
    flat_clearance_mm: float,
    cap_mode: str,
    planar_extra_limit_mm: float,
    inward_directions: np.ndarray | None = None,
    bottom_points_override: np.ndarray | None = None,
    plane_record_override: dict | None = None,
    parent_thickness_probe: ParentThicknessProbe | None = None,
    cap_template_decision: CapDecision | None = None,
) -> tuple[int, int, dict]:
    u, v = orthonormal_basis(inward)
    if inward_directions is None:
        inward_directions = np.tile(np.asarray(inward, dtype=np.float64), (len(top_points), 1))
    inward_directions = np.asarray(inward_directions, dtype=np.float64)
    inward_directions /= np.maximum(np.linalg.norm(inward_directions, axis=1)[:, None], 1e-12)
    if bottom_points_override is None:
        distances, inward_directions, plane_record = boundary_cap_distances(
            top_points,
            inward,
            inward_directions,
            max_extension_mm,
            flat_clearance_mm,
            cap_mode,
            planar_extra_limit_mm,
            parent_thickness_probe,
        )
        bottom_points = top_points + inward_directions * distances[:, None]
    else:
        bottom_points = np.asarray(bottom_points_override, dtype=np.float64)
        if bottom_points.shape != top_points.shape:
            raise ValueError("Socket bottom override must match the top boundary shape")
        displacement = bottom_points - top_points
        distances = np.linalg.norm(displacement, axis=1)
        inward_directions = displacement / np.maximum(distances[:, None], 1e-12)
        plane_record = dict(plane_record_override or {})

    bottom_ids = []
    for bottom_point in bottom_points:
        bottom_ids.append(len(output_vertices))
        output_vertices.append(bottom_point)

    side_faces = add_side_faces_between_rings(
        output_faces,
        top_ids,
        bottom_ids,
        vertices=output_vertices,
    )

    if (
        cap_template_decision is not None
        and cap_template_decision.cap_template_points is not None
        and cap_template_decision.cap_template_faces is not None
        and cap_template_decision.cap_template_boundary_ids is not None
    ):
        cap_faces, harmonic_quality = append_harmonic_cap_template(
            output_vertices,
            output_faces,
            np.asarray(cap_template_decision.cap_template_points, dtype=np.float64),
            np.asarray(cap_template_decision.cap_template_faces, dtype=np.int64),
            cap_template_decision.cap_template_boundary_ids,
            bottom_ids,
            target_boundary_points=bottom_points,
        )
        if not bool(harmonic_quality["valid"]):
            raise ValueError(
                "authoritative harmonic socket cap failed: "
                + json.dumps(harmonic_quality, ensure_ascii=False, sort_keys=True)
            )
        plane_record["harmonic_socket_cap_quality"] = harmonic_quality
    else:
        bottom_points_2d = [
            project_points(
                np.array([output_vertices[i] for i in bottom_ids]),
                origin,
                u,
                v,
            )
        ]
        cap_faces = triangulate_cap(
            output_vertices,
            output_faces,
            [bottom_ids],
            bottom_points_2d,
            [[0]],
            inward,
        )
    extension_record = {
        "vertices": len(top_ids),
        "extension_min_mm": float(distances.min()),
        "extension_max_mm": float(distances.max()),
        **plane_record,
    }
    return side_faces, cap_faces, extension_record

def add_inward_lead_extrusion_and_cap(
    output_vertices: list[np.ndarray],
    output_faces: list[list[int]],
    visible_top_ids: list[int],
    visible_top_points: np.ndarray,
    internal_fit_points: np.ndarray,
    inward: np.ndarray,
    inward_directions: np.ndarray,
    bottom_points: np.ndarray,
    lead_in_mm: float,
    max_extension_mm: float,
    flat_clearance_mm: float,
    cap_mode: str,
    planar_extra_limit_mm: float,
    plane_record: dict,
    parent_thickness_probe: ParentThicknessProbe | None,
    lead_in_directions: np.ndarray | None = None,
    cap_template_decision: CapDecision | None = None,
) -> tuple[int, int, dict]:
    """Continue from the shared visible planar-arc rim into the internal fit.

    The caller has already replaced the imperfect source seam and diffused the
    change through its visible surface band.  This builder therefore starts at
    that authoritative shared rim and adds only the subsurface fit transition.
    """
    visible_top_points = np.asarray(visible_top_points, dtype=np.float64)
    internal_fit_points = np.asarray(internal_fit_points, dtype=np.float64)
    inward_directions = np.asarray(inward_directions, dtype=np.float64)
    bottom_points = np.asarray(bottom_points, dtype=np.float64)
    if not (
        visible_top_points.shape
        == internal_fit_points.shape
        == inward_directions.shape
        == bottom_points.shape
    ):
        raise ValueError("lead-in rings must have matching shapes")
    if lead_in_directions is None:
        lead_directions = inward_directions.copy()
        lead_direction_source = "cap_direction_compatibility_fallback"
    else:
        lead_directions = np.asarray(lead_in_directions, dtype=np.float64)
        if lead_directions.shape != visible_top_points.shape:
            raise ValueError("local lead-in directions must match the visible top ring")
        lead_direction_source = "local_surface_inward_normal"
    lead_direction_lengths = np.linalg.norm(lead_directions, axis=1)
    if np.any(lead_direction_lengths <= 1e-12) or not np.all(
        np.isfinite(lead_direction_lengths)
    ):
        raise ValueError("local lead-in directions must be finite non-zero vectors")
    lead_directions = lead_directions / lead_direction_lengths[:, None]
    remaining = np.linalg.norm(bottom_points - internal_fit_points, axis=1)
    if bool(plane_record.get("guided_internal_cut_applied", False)):
        lead_points = (visible_top_points + bottom_points) * 0.5
        lead_depths = np.linalg.norm(lead_points - internal_fit_points, axis=1)
        lead_taper_record = {
            "lead_profile": "guided_internal_midpoint_loft",
            "guided_internal_midpoint_fraction": 0.5,
        }
    else:
        lead_points, lead_depths, lead_directions, lead_taper_record = coherent_tapered_lead_ring(
            visible_top_points,
            internal_fit_points,
            lead_directions,
            remaining,
            lead_in_mm,
            reference_axis=inward,
        )
    lead_ids: list[int] = []
    for point in lead_points:
        lead_ids.append(len(output_vertices))
        output_vertices.append(np.asarray(point, dtype=np.float64))
    lead_side_faces = add_side_faces_between_rings(
        output_faces,
        [int(value) for value in visible_top_ids],
        lead_ids,
        vertices=output_vertices,
    )
    lower_side_faces, cap_faces, extension_record = add_loop_extrusion_and_cap(
        output_vertices=output_vertices,
        output_faces=output_faces,
        top_ids=lead_ids,
        top_points=lead_points,
        inward=inward,
        origin=internal_fit_points.mean(axis=0),
        max_extension_mm=max_extension_mm,
        flat_clearance_mm=flat_clearance_mm,
        cap_mode=cap_mode,
        planar_extra_limit_mm=planar_extra_limit_mm,
        inward_directions=inward_directions,
        bottom_points_override=bottom_points,
        plane_record_override=plane_record,
        parent_thickness_probe=parent_thickness_probe,
        cap_template_decision=cap_template_decision,
    )
    full_travel = np.linalg.norm(bottom_points - visible_top_points, axis=1)
    lateral_shift = np.linalg.norm(internal_fit_points - visible_top_points, axis=1)
    extension_record.update(
        {
            "extension_min_mm": float(full_travel.min()),
            "extension_max_mm": float(full_travel.max()),
            "lead_in_applied": True,
            "lead_in_depth_min_mm": float(lead_depths.min()),
            "lead_in_depth_max_mm": float(lead_depths.max()),
            "visible_top_source_preserved": False,
            "visible_boundary_retopologized": True,
            "lead_direction_source": lead_direction_source,
            "internal_fit_lateral_shift_min_mm": float(lateral_shift.min()),
            "internal_fit_lateral_shift_max_mm": float(lateral_shift.max()),
            **lead_taper_record,
        }
    )
    return int(lead_side_faces + lower_side_faces), int(cap_faces), extension_record

def matched_socket_bottom_geometry(
    source_boundary_points: np.ndarray,
    socket_top_points: np.ndarray,
    inward: np.ndarray,
    inward_directions: np.ndarray,
    child_insert_shrink_mm: float,
    max_extension_mm: float,
    flat_clearance_mm: float,
    cap_mode: str,
    planar_extra_limit_mm: float,
    bottom_clearance_mm: float,
    parent_thickness_probe: ParentThicknessProbe | None = None,
    cap_decision: CapDecision | None = None,
    source_vertex_ids: list[int] | tuple[int, ...] | None = None,
    boundary_match_points: np.ndarray | None = None,
    boundary_reconciliation_tolerance_mm: float = 0.001,
    error_context: str | None = None,
    child_interior_conormals: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Reuse the child's actual cap field, then place the socket bottom behind it."""
    inward = np.asarray(inward, dtype=np.float64)
    inward /= max(float(np.linalg.norm(inward)), 1e-12)
    source_boundary_points = np.asarray(source_boundary_points, dtype=np.float64)
    socket_top_points = np.asarray(socket_top_points, dtype=np.float64)
    if child_interior_conormals is None:
        u, v = orthonormal_basis(inward)
        child_fit_points = radial_offset_points(
            source_boundary_points,
            source_boundary_points.mean(axis=0),
            u,
            v,
            inward,
            -max(float(child_insert_shrink_mm), 0.0),
        )
    else:
        child_fit_points = offset_points_along_conormals(
            source_boundary_points,
            child_interior_conormals,
            max(float(child_insert_shrink_mm), 0.0),
        )
    if cap_decision is not None:
        if source_vertex_ids is None:
            raise ValueError("a shared cap decision requires source vertex ids")
        try:
            (
                child_fit_points,
                child_distances,
                child_directions,
                child_record,
            ) = remap_cap_decision(
                cap_decision,
                source_vertex_ids,
                requested_fit_points=child_fit_points,
                requested_source_points=boundary_match_points,
                geometric_tolerance_mm=boundary_reconciliation_tolerance_mm,
            )
        except ValueError as exc:
            if error_context:
                raise ValueError(f"{error_context}: {exc}") from exc
            raise
    else:
        child_distances, child_directions, child_record = boundary_cap_distances(
            child_fit_points,
            inward,
            inward_directions,
            max_extension_mm,
            flat_clearance_mm,
            cap_mode,
            max(
                float(planar_extra_limit_mm)
                - max(float(bottom_clearance_mm), 0.0),
                0.0,
            ),
            parent_thickness_probe,
        )
        child_distances, child_record = reserve_flat_socket_travel_budget(
            child_fit_points=child_fit_points,
            child_distances=child_distances,
            child_directions=child_directions,
            socket_top_points=socket_top_points,
            bottom_clearance_mm=bottom_clearance_mm,
            maximum_socket_travel_mm=max_extension_mm
            + max(float(planar_extra_limit_mm), 0.0),
            plane_record=child_record,
        )
    child_bottom_points = child_fit_points + child_directions * child_distances[:, None]
    clearance = max(float(bottom_clearance_mm), 0.0)
    if str(child_record.get("cap_mode")) == "flat":
        plane_normal = np.asarray(child_directions[0], dtype=np.float64)
        plane_normal /= max(float(np.linalg.norm(plane_normal)), 1e-12)
        child_plane_s = float(np.mean(child_bottom_points @ plane_normal))
        socket_distances = (
            child_plane_s
            + clearance
            - socket_top_points @ plane_normal
        )
        socket_bottom_points = socket_top_points + socket_distances[:, None] * plane_normal
    else:
        socket_bottom_points = socket_top_points + child_directions * (
            child_distances + clearance
        )[:, None]
    record = dict(child_record)
    record.update(
        {
            "socket_cap_field_source": (
                "prevalidated_shared_child_cap"
                if cap_decision is not None
                else "matched_child_cap"
            ),
            "socket_bottom_clearance_applied_mm": clearance,
            "child_insert_shrink_mm": max(float(child_insert_shrink_mm), 0.0),
        }
    )
    return socket_bottom_points, record

def reserve_flat_socket_travel_budget(
    child_fit_points: np.ndarray,
    child_distances: np.ndarray,
    child_directions: np.ndarray,
    socket_top_points: np.ndarray,
    bottom_clearance_mm: float,
    maximum_socket_travel_mm: float,
    plane_record: dict,
) -> tuple[np.ndarray, dict]:
    """Shift a child plane outward so its matching socket, including clearance, stays in budget."""
    record = dict(plane_record)
    if str(record.get("cap_mode")) != "flat" or not len(child_distances):
        return child_distances, record
    child_fit_points = np.asarray(child_fit_points, dtype=np.float64)
    child_directions = np.asarray(child_directions, dtype=np.float64)
    child_distances = np.asarray(child_distances, dtype=np.float64)
    child_bottom_points = (
        child_fit_points + child_directions * child_distances[:, None]
    )
    plane_normal = fit_plane_normal(
        child_bottom_points,
        np.asarray(child_directions, dtype=np.float64).mean(axis=0),
    )
    ray_dot_plane = child_directions @ plane_normal
    if float(ray_dot_plane.min()) <= 0.05:
        raise ValueError(
            "flat socket budget shift has a near-tangent inward ray: "
            f"minimum_ray_dot_plane={float(ray_dot_plane.min()):.9f}"
        )
    child_plane_s = float(np.mean(child_bottom_points @ plane_normal))
    required_socket_distances = (
        child_plane_s
        + max(float(bottom_clearance_mm), 0.0)
        - np.asarray(socket_top_points, dtype=np.float64) @ plane_normal
    )
    socket_budget_shift = max(
        0.0,
        float(required_socket_distances.max()) - float(maximum_socket_travel_mm),
    )
    shifted_plane_s = child_plane_s - socket_budget_shift
    shifted_distances = (
        shifted_plane_s - child_fit_points @ plane_normal
    ) / ray_dot_plane
    effective_minimum = float(
        record.get("effective_minimum_inward_depth_mm", MINIMUM_INWARD_DEPTH_MM)
    )
    if float(shifted_distances.min()) < effective_minimum - 1e-9:
        record.update(
            {
                "socket_budget_shift_mm": socket_budget_shift,
                "socket_budget_shift_applied": False,
                "socket_budget_shift_reason": "minimum_inward_depth_would_be_violated",
                "socket_budget_effective_minimum_mm": effective_minimum,
            }
        )
        return child_distances, record
    record.update(
        {
            "socket_budget_shift_mm": socket_budget_shift,
            "socket_budget_shift_applied": bool(socket_budget_shift > 0.0),
            "socket_budget_shift_reason": "matching_socket_total_travel_limit",
            "socket_budget_shift_strategy": (
                "parallel_plane_ray_reintersection"
            ),
            "maximum_matching_socket_travel_mm": float(maximum_socket_travel_mm),
            "socket_budget_effective_minimum_mm": effective_minimum,
            "maximum_generated_inward_travel_mm": float(shifted_distances.max()),
            "minimum_generated_inward_travel_mm": float(shifted_distances.min()),
        }
    )
    return shifted_distances, record

def _append_points(
    output_vertices: list[np.ndarray],
    points: np.ndarray,
) -> list[int]:
    ids: list[int] = []
    for point in np.asarray(points, dtype=np.float64):
        ids.append(len(output_vertices))
        output_vertices.append(np.asarray(point, dtype=np.float64))
    return ids

def _resample_closed_ring(points: np.ndarray, sample_count: int) -> np.ndarray:
    """Resample a spatial loop at equal arc-length intervals."""
    ring = np.asarray(points, dtype=np.float64)
    if len(ring) < 3 or int(sample_count) < 3:
        raise ValueError("closed ring resampling needs at least three points")
    following = np.roll(ring, -1, axis=0)
    segment_lengths = np.linalg.norm(following - ring, axis=1)
    perimeter = float(segment_lengths.sum())
    if perimeter <= 1e-9:
        raise ValueError("cannot resample a zero-length connector boundary")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    targets = np.linspace(0.0, perimeter, int(sample_count), endpoint=False)
    result = []
    for target in targets:
        segment = min(
            int(np.searchsorted(cumulative, target, side="right") - 1),
            len(ring) - 1,
        )
        length = float(segment_lengths[segment])
        ratio = 0.0 if length <= 1e-12 else (float(target) - cumulative[segment]) / length
        result.append(ring[segment] + ratio * (following[segment] - ring[segment]))
    return np.asarray(result, dtype=np.float64)

def _aligned_ring_indices(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the cyclic/reversal order of target closest to reference."""
    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(reference) != len(target):
        raise ValueError("ring alignment requires equal sample counts")
    best_score = None
    best_indices = None
    base = np.arange(len(target), dtype=np.int64)
    for candidate in (base, base[::-1]):
        for shift in range(len(target)):
            indices = np.roll(candidate, shift)
            score = float(np.sum((reference - target[indices]) ** 2))
            if best_score is None or score < best_score:
                best_score = score
                best_indices = indices.copy()
    if best_indices is None:
        raise RuntimeError("connector ring alignment produced no candidate")
    return best_indices

def _add_faces_between_unequal_rings(
    output_faces: list[list[int]],
    outer_ids: list[int],
    inner_ids: list[int],
    *,
    vertices: list[np.ndarray] | np.ndarray | None = None,
) -> int:
    """Zipper two loops while consuming every edge exactly once.

    The former normalized-parameter zipper ignored the actual geometry.  A
    dense sampled source curve joined to a simplified Clipper contour could
    therefore spend hundreds of steps at one inner vertex and create long,
    nearly zero-area spikes.  The dynamic program below chooses the monotone
    non-crossing strip with the healthiest triangles while preserving every
    edge of both rings exactly once.
    """
    outer = [int(value) for value in outer_ids]
    inner = [int(value) for value in inner_ids]
    if len(outer) < 3 or len(inner) < 3:
        raise ValueError("connector backing rings need at least three vertices")
    if vertices is None:
        raise ValueError("geometry-aware unequal-ring zipper needs vertices")
    points = np.asarray(vertices, dtype=np.float64)
    outer_points = points[np.asarray(outer, dtype=np.int64)]
    inner_points = points[np.asarray(inner, dtype=np.int64)]

    # Put the seam at the closest ring pair.  This keeps the unwrapped dynamic
    # program away from a needless long diagonal at its start/end boundary.
    distances = np.linalg.norm(
        outer_points[:, None, :] - inner_points[None, :, :], axis=2
    )
    seam_outer, seam_inner = np.unravel_index(int(np.argmin(distances)), distances.shape)
    outer = outer[seam_outer:] + outer[:seam_outer]
    inner = inner[seam_inner:] + inner[:seam_inner]
    outer_points = points[np.asarray(outer, dtype=np.int64)]
    inner_points = points[np.asarray(inner, dtype=np.int64)]
    n = len(outer)
    m = len(inner)

    def triangle_cost(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        edges = np.array(
            [np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)],
            dtype=np.float64,
        )
        double_area = float(np.linalg.norm(np.cross(b - a, c - a)))
        if double_area <= 1e-14:
            return 1e30
        # max_edge^2 / double_area is a scale-free sliver measure.  A small
        # diagonal term breaks ties without preferring a geometrically awful
        # triangle merely because it is short.
        aspect = float(edges.max() ** 2 / double_area)
        return aspect * aspect + 1e-6 * float(np.sum(edges * edges))

    costs = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    previous = np.full((n + 1, m + 1), -1, dtype=np.int8)
    costs[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            current = float(costs[i, j])
            if not np.isfinite(current):
                continue
            if i < n:
                value = current + triangle_cost(
                    outer_points[i % n],
                    outer_points[(i + 1) % n],
                    inner_points[j % m],
                )
                if value < costs[i + 1, j]:
                    costs[i + 1, j] = value
                    previous[i + 1, j] = 0
            if j < m:
                value = current + triangle_cost(
                    outer_points[i % n],
                    inner_points[(j + 1) % m],
                    inner_points[j % m],
                )
                if value < costs[i, j + 1]:
                    costs[i, j + 1] = value
                    previous[i, j + 1] = 1
    if not np.isfinite(costs[n, m]):
        raise ValueError("unequal connector rings cannot form a valid monotone strip")

    moves: list[int] = []
    i, j = n, m
    while i or j:
        move = int(previous[i, j])
        if move == 0:
            moves.append(0)
            i -= 1
        elif move == 1:
            moves.append(1)
            j -= 1
        else:
            raise ValueError("unequal connector strip backtracking failed")
    moves.reverse()
    i = j = 0
    face_start = len(output_faces)
    for move in moves:
        if move == 0:
            output_faces.append(
                [outer[i % n], outer[(i + 1) % n], inner[j % m]]
            )
            i += 1
        else:
            output_faces.append(
                [outer[i % n], inner[(j + 1) % m], inner[j % m]]
            )
            j += 1
    generated = np.asarray(output_faces[face_start:], dtype=np.int64)
    triangles = points[generated]
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if np.any(double_areas <= 1e-12):
        del output_faces[face_start:]
        raise ValueError("unequal connector strip contains a degenerate face")
    return int(len(outer) + len(inner))

def _add_constrained_connector_annulus(
    *,
    output_vertices: list[np.ndarray],
    output_faces: list[list[int]],
    outer_ids: list[int],
    outer_points: np.ndarray,
    inner_points: np.ndarray,
    plan: dict,
    outward_normal: np.ndarray,
) -> tuple[list[int], np.ndarray, int]:
    """Build and audit one exact outer-minus-inner connector annulus.

    Geometry strategy, topology auditing, and face-orientation propagation live
    in connector_topology.  This adapter owns only mesh-buffer mutation and the
    connector-specific printable-face quality gate.
    """

    outer = np.asarray(outer_points, dtype=np.float64)
    inner = np.asarray(inner_points, dtype=np.float64)
    if len(outer_ids) != len(outer):
        raise ValueError("connector annulus outer ids and points must match")
    if len(outer) < 3 or len(inner) < 3:
        raise ValueError("connector annulus needs two valid closed rings")

    outer_2d = _project_connector_points(outer, plan)
    inner_2d = _project_connector_points(inner, plan)
    if not all(
        point_in_poly(point, outer_2d)
        or point_on_poly_boundary(point, outer_2d)
        for point in inner_2d
    ):
        raise ValueError("compact connector ring must remain inside its backing")

    order = np.arange(len(inner), dtype=np.int64)
    inner_ids = _append_points(output_vertices, inner)
    face_start = len(output_faces)
    try:
        constrained_faces, topology_audit = triangulate_connector_annulus(
            [int(value) for value in outer_ids],
            outer_2d,
            [int(value) for value in inner_ids],
            inner_2d,
        )
        if not topology_audit.valid or not constrained_faces:
            raise ValueError(
                "constrained connector backing annulus triangulation failed: "
                + str(topology_audit.reason or "unknown topology failure")
            )

        point_by_id = {
            int(vertex_id): np.asarray(output_vertices[int(vertex_id)], dtype=np.float64)
            for vertex_id in set(int(value) for value in outer_ids + inner_ids)
        }
        oriented_faces = orient_face_patch_consistently(
            constrained_faces,
            point_by_id,
            np.asarray(outward_normal, dtype=np.float64),
        )
        output_faces.extend([list(face) for face in oriented_faces])

        perimeter_edges = {
            tuple(sorted((int(ring[index]), int(ring[(index + 1) % len(ring)]))))
            for ring in (list(outer_ids), inner_ids)
            for index in range(len(ring))
        }
        annulus_quality = _validate_connector_wedge_faces(
            output_vertices,
            output_faces,
            face_start,
            perimeter_edges=perimeter_edges,
            planar_boundary_rings=[set(outer_ids), set(inner_ids)],
            planar_surface=True,
        )
    except (RuntimeError, ValueError):
        del output_faces[face_start:]
        del output_vertices[-len(inner_ids):]
        raise

    plan["backing_annulus_triangulation_method"] = str(topology_audit.strategy)
    plan["backing_annulus_topology_audit"] = {
        "face_count": int(topology_audit.face_count),
        "boundary_edge_count": int(topology_audit.boundary_edge_count),
        "covered_area": float(topology_audit.covered_area),
        "expected_area": float(topology_audit.expected_area),
    }
    plan["backing_annulus_oriented_components"] = 1
    plan["backing_annulus_repaired_boundary_ears"] = 0
    plan["backing_annulus_max_aspect"] = float(
        annulus_quality["backing_wedge_max_aspect"]
    )
    plan["backing_annulus_dense_sampling_faces"] = int(
        annulus_quality["backing_wedge_dense_sampling_faces"]
    )
    plan["backing_annulus_planar_boundary_ear_faces"] = int(
        annulus_quality["backing_wedge_planar_boundary_ear_faces"]
    )
    return inner_ids, order, int(len(oriented_faces))

def _connector_record(plan: dict, role: str, backing_faces: int, side_faces: int, cap_faces: int) -> dict:
    return {
        "interface_geometry": "local_connector",
        "connector_role": str(role),
        "peg_width_mm": float(plan["peg_width_mm"]),
        "peg_length_mm": float(plan["peg_length_mm"]),
        "engagement_depth_mm": float(plan["engagement_depth_mm"]),
        "socket_depth_mm": float(plan["socket_depth_mm"]),
        "total_clearance_mm": float(plan["total_clearance_mm"]),
        "socket_bottom_clearance_mm": float(
            plan.get("socket_bottom_clearance_mm", 0.0)
        ),
        "backing_clearance_per_side_mm": float(
            plan.get("backing_clearance_per_side_mm", 0.0)
        ),
        "backing_bottom_clearance_mm": float(
            plan.get("backing_bottom_clearance_mm", 0.0)
        ),
        "backing_clearance_shift_mm": float(
            plan.get("backing_clearance_shift_mm", 0.0)
        ),
        "backing_clearance_strategy": str(
            plan.get("backing_clearance_strategy", "none")
        ),
        "socket_mouth_chamfer_mm": float(plan["socket_mouth_chamfer_mm"]),
        "mouth_slope_degrees": float(plan["mouth_slope_degrees"]),
        "footprint_area_ratio": float(plan["footprint_area_ratio"]),
        "interface_area_mm2": float(plan["interface_area_mm2"]),
        "edge_clearance_mm": float(plan["edge_clearance_mm"]),
        "fit_scale": float(plan["fit_scale"]),
        "full_boundary_backing_depth_mm": float(
            plan.get("full_boundary_backing_depth_mm", 0.0)
        ),
        "elastic_shrink_applied": bool(
            plan.get("elastic_shrink_applied", False)
        ),
        "backing_elastic_shrink_applied": bool(
            plan.get("backing_elastic_shrink_applied", False)
        ),
        "engagement_elastic_shrink_applied": bool(
            plan.get("engagement_elastic_shrink_applied", False)
        ),
        "elastic_backing_scale": float(
            plan.get("elastic_backing_scale", 1.0)
        ),
        "elastic_engagement_scale": float(
            plan.get("elastic_engagement_scale", 1.0)
        ),
        "elastic_lateral_scale": float(
            plan.get("elastic_lateral_scale", 1.0)
        ),
        "nominal_full_boundary_backing_depth_mm": float(
            plan.get(
                "nominal_full_boundary_backing_depth_mm",
                plan.get("full_boundary_backing_depth_mm", 0.0),
            )
        ),
        "minimum_elastic_backing_depth_mm": float(
            plan.get("minimum_elastic_backing_depth_mm", 0.0)
        ),
        "nominal_engagement_depth_mm": float(
            plan.get("nominal_engagement_depth_mm", 5.0)
        ),
        "backing_safety_limit_mm": float(
            plan.get(
                "backing_safety_limit_mm",
                plan.get("full_boundary_backing_depth_mm", 0.0),
            )
        ),
        "total_safety_limit_mm": float(
            plan.get("total_safety_limit_mm", 0.0)
        ),
        "compact_peg_enabled": bool(
            plan.get("compact_peg_enabled", True)
        ),
        "backing_taper_requested_mm": float(
            plan.get("backing_taper_requested_mm", 0.0)
        ),
        "backing_taper_shape_backoff_applied": bool(
            plan.get("backing_taper_shape_backoff_applied", False)
        ),
        "backing_inward_direction_mode": str(
            plan.get("backing_inward_direction_mode", "single_global_flat_direction")
        ),
        "backing_inset_method": str(
            plan.get("backing_inset_method", "source_vertex_miter_offset")
        ),
        "backing_ring_correspondence": str(
            plan.get("backing_ring_correspondence", "unrecorded")
        ),
        "backing_ring_outer_vertices": int(
            plan.get("backing_ring_outer_vertices", 0)
        ),
        "backing_ring_inner_vertices": int(
            plan.get("backing_ring_inner_vertices", 0)
        ),
        "backing_ring_scale": float(plan.get("backing_ring_scale", 1.0)),
        "backing_source_ring_vertices": int(
            plan.get("backing_source_ring_vertices", 0)
        ),
        "backing_floor_ring_vertices": int(
            plan.get("backing_floor_ring_vertices", 0)
        ),
        "backing_annulus_max_aspect": float(
            plan.get("backing_annulus_max_aspect", 0.0)
        ),
        "backing_annulus_dense_sampling_faces": int(
            plan.get("backing_annulus_dense_sampling_faces", 0)
        ),
        "backing_annulus_planar_boundary_ear_faces": int(
            plan.get("backing_annulus_planar_boundary_ear_faces", 0)
        ),
        "backing_annulus_triangulation_method": str(
            plan.get("backing_annulus_triangulation_method", "unknown")
        ),
        "backing_annulus_repaired_boundary_ears": int(
            plan.get("backing_annulus_repaired_boundary_ears", 0)
        ),
        "backing_annulus_oriented_components": int(
            plan.get("backing_annulus_oriented_components", 0)
        ),
        "center": np.asarray(plan["center"], dtype=np.float64).round(6).tolist(),
        "inward": np.asarray(plan["inward"], dtype=np.float64).round(6).tolist(),
        "backing_faces_added": int(backing_faces),
        "connector_side_faces_added": int(side_faces),
        "connector_cap_faces_added": int(cap_faces),
    }

def _rings_coincident(first: np.ndarray, second: np.ndarray) -> bool:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    return bool(
        first_array.shape == second_array.shape
        and np.max(np.linalg.norm(first_array - second_array, axis=1)) <= 1e-9
    )

def _join_connector_rings(
    output_vertices: list[np.ndarray],
    output_faces: list[list[int]],
    first_ids: list[int],
    second_ids: list[int],
    *,
    first_points: np.ndarray,
    second_points: np.ndarray,
    plan: dict,
    ) -> tuple[int, dict, np.ndarray]:
    """Join one backing layer with bounded many-to-one correspondence."""

    strip = triangulate_bounded_ring_strip(
        [int(value) for value in first_ids],
        np.asarray(first_points, dtype=np.float64),
        [int(value) for value in second_ids],
        np.asarray(second_points, dtype=np.float64),
        _project_connector_points(first_points, plan),
        _project_connector_points(second_points, plan),
        maximum_fanout=4,
        allow_projection_bridge_passthrough=(
            str(plan.get("backing_surface_validation_mode", "strict"))
            == "advisory"
        ),
    )
    if not strip.audit.valid or not strip.faces:
        raise ValueError(
            "connector backing layer strip failed quality audit: "
            + str(strip.audit.reason or "unknown strip failure")
        )
    returned_ids = [int(value) for value in strip.inner_ids]
    if returned_ids != [int(value) for value in second_ids]:
        original_count = len(second_ids)
        refined_points = np.asarray(strip.inner_points, dtype=np.float64)
        if len(refined_points) <= original_count:
            raise ValueError("connector strip returned an invalid refined ring")
        replacement: dict[int, int] = {}
        next_local_id = len(output_vertices)
        original_id_set = set(int(value) for value in second_ids)
        for returned_id, point in zip(returned_ids, refined_points):
            if returned_id in original_id_set:
                replacement[returned_id] = returned_id
            else:
                replacement[returned_id] = int(next_local_id)
                output_vertices.append(np.asarray(point, dtype=np.float64))
                next_local_id += 1
        mapped_faces = [
            tuple(replacement.get(int(value), int(value)) for value in face)
            for face in strip.faces
        ]
        second_ids[:] = [replacement[int(value)] for value in returned_ids]
    else:
        mapped_faces = list(strip.faces)
    output_faces.extend([list(face) for face in mapped_faces])
    return (
        int(len(mapped_faces)),
        dict(strip.audit.__dict__),
        np.asarray(strip.inner_points, dtype=np.float64),
    )

def _validate_connector_wedge_faces(
    output_vertices: list[np.ndarray],
    output_faces: list[list[int]],
    face_start: int,
    *,
    perimeter_edges: set[tuple[int, int]] | None = None,
    planar_boundary_rings: list[set[int]] | None = None,
    coherent_boundary_rings: list[set[int]] | None = None,
    coherent_boundary_pairs: list[set[int]] | None = None,
    heightfield_surface_vertices: set[int] | None = None,
    planar_surface: bool = False,
    shallow_minimal_closure_plan: dict | None = None,
) -> dict:
    generated = np.asarray(output_faces[face_start:], dtype=np.int64)
    if not len(generated):
        return {
            "backing_wedge_min_double_area_mm2": 0.0,
            "backing_wedge_max_aspect": 0.0,
        }
    points = np.asarray(output_vertices, dtype=np.float64)
    triangles = points[generated]
    edges = np.stack(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if np.any(~np.isfinite(double_areas)) or np.any(double_areas <= 1e-12):
        raise ValueError("connector backing wedge contains a degenerate face")
    aspects = edges.max(axis=1) ** 2 / double_areas
    maximum_aspect = float(aspects.max())
    dense_sampling_faces = 0
    planar_boundary_ear_faces = 0
    coherent_boundary_ear_faces = 0
    invalid_needle_faces: list[int] = []
    invalid_needle_metrics: list[dict] = []
    perimeter_edges = perimeter_edges or set()
    planar_boundary_rings = planar_boundary_rings or []
    coherent_boundary_rings = coherent_boundary_rings or []
    coherent_boundary_pairs = coherent_boundary_pairs or []
    heightfield_surface_vertices = heightfield_surface_vertices or set()
    cross_layer_coincident_ear_faces = 0
    cross_layer_dense_perimeter_ear_faces = 0
    for local_index in np.flatnonzero(aspects > 750.0):
        face = generated[int(local_index)]
        face_vertices = {int(value) for value in face}
        edge_pairs = (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        )
        edge_lengths = edges[int(local_index)]
        shortest_index = int(np.argmin(edge_lengths))
        shortest_edge = tuple(sorted(edge_pairs[shortest_index]))
        shortest_length = float(edge_lengths[shortest_index])
        other_lengths = np.delete(edge_lengths, shortest_index)
        altitude = float(double_areas[int(local_index)] / max(shortest_length, 1e-12))
        long_edge_ratio = float(
            other_lengths.max() / max(other_lengths.min(), 1e-12)
        )
        # Aspect ratio is not a spike criterion on the planar, constrained
        # annulus: every face is coplanar, while edge topology, orientation and
        # exact covered area are validated by the caller.  Dense source curves
        # and the two legal bridge cuts can therefore create thin 2-D faces
        # without creating any 3-D tent or folded wall.
        if planar_surface:
            if any(face_vertices.issubset(ring) for ring in planar_boundary_rings):
                planar_boundary_ear_faces += 1
            else:
                dense_sampling_faces += 1
            continue
        # A dense source boundary can contain sub-millimetre perimeter edges
        # beside a several-millimetre backing depth.  Its paired strip
        # triangles are unavoidably slender, but they do not form geometric
        # spikes: the short edge remains on a real ring, the opposing altitude
        # is printable, and the two cross-ring edges are nearly symmetric.
        # Continue rejecting slivers caused by a cross-ring short edge,
        # near-collinearity, or a long tangential bridge.
        if (
            shortest_edge in perimeter_edges
            and altitude >= 0.10
            and long_edge_ratio <= 2.0
        ):
            dense_sampling_faces += 1
            continue
        # A source perimeter can be sampled much more densely than the local
        # physical layer spacing.  At a trimmed-contour event the paired point
        # on the next layer may approach the perimeter edge closely, producing
        # a shallow triangle with two balanced cross-layer sides.  Accept it
        # only when the face spans one audited adjacent-ring pair and is not a
        # same-ring planar sliver; the latter remains a blocking defect.
        if (
            shortest_edge in perimeter_edges
            and long_edge_ratio <= 1.20
            and any(
                face_vertices.issubset(pair)
                for pair in coherent_boundary_pairs
            )
            and not any(
                face_vertices.issubset(ring)
                for ring in coherent_boundary_rings
            )
        ):
            cross_layer_dense_perimeter_ear_faces += 1
            continue
        # When a trimmed inset passes a medial-axis event, one vertex on the
        # next physical-depth layer can land almost directly below a vertex on
        # the previous layer.  The resulting triangle has a tiny *cross-layer*
        # edge and two balanced 45-degree edges.  It is a local strip ear, not
        # a blade: the constrained strip audit already proves that its edges
        # stay inside the annulus.  Limit this exception to one known adjacent
        # ring pair, printable altitude, and nearly equal long edges.
        if (
            shortest_edge not in perimeter_edges
            and altitude >= 0.10
            and long_edge_ratio <= 1.10
            and any(
                face_vertices.issubset(pair)
                for pair in coherent_boundary_pairs
            )
        ):
            cross_layer_coincident_ear_faces += 1
            continue
        # Conforming backing repair/refinement can inherit a very short edge
        # from a densely sampled source rim.  Its two long sides remain almost
        # equal and every vertex belongs to the audited backing patch (also
        # including its bounded inward source-chord collars), so this is a
        # narrow surface cell rather than a knife.
        if (
            long_edge_ratio <= 1.10
            and face_vertices.issubset(heightfield_surface_vertices)
        ):
            cross_layer_coincident_ear_faces += 1
            continue
        # A topology-preserving offset may retain a microscopic, almost
        # collinear cell beside the collapsed inner contour.  It is harmless
        # only when the complete triangle is sub-nozzle scale; this absolute
        # cap cannot admit the multi-millimetre blades the audit targets.
        if (
            float(edge_lengths.max()) <= 0.25 + 1e-12
            and face_vertices.issubset(heightfield_surface_vertices)
        ):
            cross_layer_coincident_ear_faces += 1
            continue
        # Constrained annulus triangulation can clip one almost-collinear ear
        # from a smooth boundary ring.  All three vertices then belong to the
        # same coherent layer, so the face cannot span backing depth or become
        # a cross-layer knife.  Keep the exception narrow: a real perimeter
        # edge and two balanced chord edges are both required.
        if (
            shortest_edge in perimeter_edges
            and long_edge_ratio <= 2.0
            and any(
                face_vertices.issubset(ring)
                for ring in coherent_boundary_rings
            )
        ):
            coherent_boundary_ear_faces += 1
            continue
        # A planar constrained annulus may need a very shallow ear between
        # three samples of the same smooth, densely tessellated boundary.  It
        # cannot form a 3-D needle because every vertex belongs to one coplanar
        # ring, and the annulus topology/area audits independently prove that
        # the chord stays inside the selected surface.  Do not grant this
        # exception to cross-ring faces or to the 45-degree side wedge.
        if (
            shortest_edge in perimeter_edges
            and any(face_vertices.issubset(ring) for ring in planar_boundary_rings)
        ):
            planar_boundary_ear_faces += 1
            continue
        invalid_needle_faces.append(int(local_index))
        if len(invalid_needle_metrics) < 5:
            invalid_needle_metrics.append(
                {
                    "face": int(local_index),
                    "shortest_is_perimeter": bool(shortest_edge in perimeter_edges),
                    "altitude_mm": round(altitude, 6),
                    "maximum_edge_mm": round(float(edge_lengths.max()), 6),
                    "long_edge_ratio": round(long_edge_ratio, 6),
                    "aspect": round(float(aspects[int(local_index)]), 6),
                }
            )
    needle_ratio = float(len(invalid_needle_faces) / max(len(generated), 1))
    ratio_accepted_needle_faces = bool(
        invalid_needle_faces
        and needle_ratio <= 0.001 + 1e-12
        and len(invalid_needle_faces) <= 8
        and all(
            float(sample["altitude_mm"]) >= 0.01 - 1e-12
            and float(sample["maximum_edge_mm"]) <= 2.0 + 1e-12
            for sample in invalid_needle_metrics
        )
    )
    from .print_tolerance import small_patch_report
    micro_needle_report = small_patch_report(
        output_vertices, generated[np.asarray(invalid_needle_faces,dtype=np.int64)],
        maximum_span_mm=5.0, separate_components=True)
    strip_records = (shallow_minimal_closure_plan or {}).get('backing_strip_audits', [])
    strips_safe = all(record.get('valid', False) and not record.get('invalid_bridge_count',0)
                      and not record.get('degenerate_face_count',0) for record in strip_records)
    if invalid_needle_faces and micro_needle_report['accepted'] and np.isfinite(maximum_aspect) and strips_safe:
        ratio_accepted_needle_faces = True
        runtime_log('打印容错','micro_needle_patch_retained',
                    '小面积非退化细长面记录后保留', **micro_needle_report)
    maximum_invalid_edge_mm = max(
        (
            float(sample["maximum_edge_mm"])
            for sample in invalid_needle_metrics
        ),
        default=0.0,
    )
    shallow_minimal_needle_advisory_accepted = bool(
        shallow_minimal_closure_plan is not None
        and user_reviewed_shallow_minimal_needle_advisory_is_eligible(
            shallow_minimal_closure_plan,
            invalid_face_count=len(invalid_needle_faces),
            maximum_edge_mm=maximum_invalid_edge_mm,
        )
    )
    if not np.isfinite(maximum_aspect):
        raise ValueError(
            "connector backing wedge contains needle faces: "
            f"maximum_aspect={maximum_aspect:.3f}, "
            f"invalid_faces={len(invalid_needle_faces)}, "
            f"samples={invalid_needle_metrics}"
        )
    if (
        invalid_needle_faces
        and not ratio_accepted_needle_faces
        and not shallow_minimal_needle_advisory_accepted
    ):
        raise ValueError(
            "connector backing wedge contains needle faces: "
            f"maximum_aspect={maximum_aspect:.3f}, "
            f"invalid_faces={len(invalid_needle_faces)}, "
            f"samples={invalid_needle_metrics}, "
            "shallow_minimal_advisory_checks="
            f"{None if shallow_minimal_closure_plan is None else shallow_minimal_closure_plan.get('shallow_minimal_closure_needle_advisory_checks')}"
        )
    if shallow_minimal_needle_advisory_accepted:
        runtime_log(
            "local-connector",
            "shallow_minimal_closure_needle_advisory_accepted",
            "User-reviewed shallow minimal closure passed bounded needle advisory",
            invalid_faces=int(len(invalid_needle_faces)),
            maximum_edge_mm=float(maximum_invalid_edge_mm),
            maximum_aspect=float(maximum_aspect),
            samples=invalid_needle_metrics,
        )
    elif invalid_needle_faces:
        runtime_log(
            "local-connector",
            "connector_needle_faces_ratio_accepted",
            "Localized connector needle faces accepted by ratio validation",
            invalid_faces=int(len(invalid_needle_faces)),
            defect_ratio=needle_ratio,
            maximum_aspect=float(maximum_aspect),
            samples=invalid_needle_metrics,
        )

    # Only intermediate layer seams are expected to be geometrically smooth.
    # Their ring edges are shared by the strips immediately above and below;
    # cross-layer edges are deliberately excluded because they may preserve a
    # real corner in the painted source outline.  A large angle here is the
    # signature of a cyclic height-field mismatch: the visible crater/knife
    # defect can still be watertight and therefore escaped the old audit.
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    face_normals /= np.linalg.norm(face_normals, axis=1)[:, None]
    seam_owners: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for local_index, face in enumerate(generated):
        for left, right in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(left), int(right))))
            if edge in perimeter_edges:
                seam_owners[edge].append(int(local_index))
    seam_angles: list[float] = []
    for owners in seam_owners.values():
        if len(owners) != 2:
            continue
        cosine = float(
            np.clip(
                np.dot(face_normals[owners[0]], face_normals[owners[1]]),
                -1.0,
                1.0,
            )
        )
        seam_angles.append(float(np.degrees(np.arccos(cosine))))
    seam_array = np.asarray(seam_angles, dtype=np.float64)
    seam_p95 = float(np.percentile(seam_array, 95.0)) if len(seam_array) else None
    seam_maximum = float(seam_array.max()) if len(seam_array) else None
    seam_over_30 = int(np.count_nonzero(seam_array > 30.0))
    seam_over_60 = int(np.count_nonzero(seam_array > 60.0))
    seam_over_60_ratio = float(seam_over_60 / max(len(seam_array), 1))
    if seam_p95 is not None and (seam_p95 > 30.0 + 1e-9 or seam_over_60_ratio > 0.01 + 1e-12):
        raise ValueError(
            "connector backing layer seams are not smooth: "
            f"p95_dihedral_degrees={seam_p95:.3f}, "
            f"maximum_dihedral_degrees={seam_maximum:.3f}, "
            f"over_60_degrees={seam_over_60}, "
            f"over_60_ratio={seam_over_60_ratio:.6f}"
        )
    from .backing_shape import internal_dihedral_report
    return {
        "backing_wedge_internal_dihedral": internal_dihedral_report(
            points, generated, perimeter_edges),
        "backing_wedge_smooth_seam_status": "measured" if len(seam_array) else "not_evaluated",
        "backing_wedge_min_double_area_mm2": float(double_areas.min()),
        "backing_wedge_max_aspect": maximum_aspect,
        "backing_wedge_ratio_accepted_needle_faces": int(
            len(invalid_needle_faces) if ratio_accepted_needle_faces else 0
        ),
        "backing_wedge_shallow_minimal_advisory_needle_faces": int(
            len(invalid_needle_faces)
            if shallow_minimal_needle_advisory_accepted
            else 0
        ),
        "backing_wedge_ratio_accepted_needle_face_samples": (
            invalid_needle_metrics if ratio_accepted_needle_faces else []
        ),
        "backing_wedge_needle_face_ratio": needle_ratio,
        "backing_wedge_dense_sampling_faces": int(dense_sampling_faces),
        "backing_wedge_planar_boundary_ear_faces": int(
            planar_boundary_ear_faces
        ),
        "backing_wedge_coherent_boundary_ear_faces": int(
            coherent_boundary_ear_faces
        ),
        "backing_wedge_cross_layer_coincident_ear_faces": int(
            cross_layer_coincident_ear_faces
        ),
        "backing_wedge_cross_layer_dense_perimeter_ear_faces": int(
            cross_layer_dense_perimeter_ear_faces
        ),
        "backing_wedge_smooth_seam_edges": int(len(seam_array)),
        "backing_wedge_smooth_seam_p95_dihedral_degrees": seam_p95,
        "backing_wedge_smooth_seam_maximum_dihedral_degrees": seam_maximum,
        "backing_wedge_smooth_seam_over_30_degrees": seam_over_30,
        "backing_wedge_smooth_seam_over_60_degrees": seam_over_60,
        "backing_wedge_smooth_seam_over_60_ratio": seam_over_60_ratio,
        "backing_wedge_aspect_policy": (
            "strict_750_or_aligned_perimeter_dense_sampling"
        ),
    }

def _finalize_local_male_record(
    *,
    plan: dict,
    boundary: np.ndarray,
    backing_boundary: np.ndarray,
    backing_taper_depth: float,
    generated_ring_count: int,
    backing_faces: int,
    side_faces: int,
    cap_faces: int,
    wedge_quality: dict,
    backing_triangulation: str,
) -> dict:
    """Attach the common audit contract to both peg and backing-only males."""

    record = _connector_record(
        plan,
        "male",
        backing_faces,
        side_faces,
        cap_faces,
    )
    record["backing_taper_target_degrees"] = float(
        plan.get("backing_taper_target_degrees", 45.0)
    )
    record["backing_taper_lateral_inset_mm"] = float(
        plan.get("backing_taper_lateral_inset_mm", 0.0)
    )
    record["backing_taper_angle_search"] = list(
        plan.get("backing_taper_angle_search", [])
    )
    record["backing_taper_selected_measured_statistics"] = dict(
        plan.get("backing_taper_selected_measured_statistics", {})
    )
    taper_audit = _backing_taper_angle_audit(
        boundary,
        backing_boundary,
        plan,
    )
    measured_minimum = taper_audit["minimum_degrees"]
    measured_median = taper_audit["median_degrees"]
    measured_maximum = taper_audit["maximum_degrees"]
    record["backing_taper_measured_minimum_degrees"] = measured_minimum
    record["backing_taper_measured_median_degrees"] = measured_median
    record["backing_taper_measured_maximum_degrees"] = measured_maximum
    record["backing_taper_measured_p05_degrees"] = taper_audit["p05_degrees"]
    record["backing_taper_measured_p95_degrees"] = taper_audit["p95_degrees"]
    record["backing_taper_outside_30_75_count"] = taper_audit["outside_count"]
    record["backing_taper_outside_30_75_ratio"] = taper_audit["outside_ratio"]
    record["backing_taper_maximum_cyclic_outlier_run"] = taper_audit[
        "maximum_cyclic_outlier_run"
    ]
    record["backing_taper_maximum_allowed_outlier_run"] = taper_audit[
        "maximum_allowed_outlier_run"
    ]
    record["backing_taper_sparse_outliers_accepted"] = taper_audit[
        "accepted_sparse_outliers"
    ]
    record["backing_taper_acceptance_policy"] = taper_audit.get("policy")
    slope_validation_mode = str(
        plan.get("backing_slope_validation_mode", "strict")
    )
    record["backing_taper_validation_mode"] = slope_validation_mode
    from .tolerance_policy import slope_warning
    finding = slope_warning(taper_audit["outside_count"], taper_audit["sample_count"])
    record["backing_taper_warning_policy"] = finding
    record["backing_taper_advisory_finding"] = bool(
        finding["warning"] and slope_validation_mode == "advisory"
    )
    if record["backing_taper_advisory_finding"]:
        from .reporting import runtime_log
        runtime_log("警告", "backing_slope_ratio", "当前接口背衬坡度超界比例大于 1%（仅警告）", **finding)
    if not bool(taper_audit["valid"]) and slope_validation_mode == "strict":
        raise ValueError(
            "connector backing slope has a sustained departure from the "
            "printable 30-75 degree interval: "
            f"minimum={measured_minimum}, "
            f"p05={taper_audit['p05_degrees']}, "
            f"median={measured_median}, "
            f"p95={taper_audit['p95_degrees']}, "
            f"maximum={measured_maximum}, "
            f"outside_ratio={taper_audit['outside_ratio']:.6f}, "
            "maximum_cyclic_outlier_run="
            f"{taper_audit['maximum_cyclic_outlier_run']}"
        )
    record["backing_taper_depth_mm"] = float(backing_taper_depth)
    record["backing_profile"] = "outer_large_inner_small"
    record["backing_profile_layer_count"] = int(generated_ring_count)
    record["backing_strip_maximum_fanout"] = int(
        plan["backing_strip_maximum_fanout"]
    )
    record["backing_strip_maximum_cross_edge_mm"] = float(
        plan["backing_strip_maximum_cross_edge_mm"]
    )
    record["backing_triangulation"] = str(backing_triangulation)
    record["source_direction_audit"] = plan.get("source_direction_audit")
    record["backing_strip_projection_audits"] = [
        audit.get("projection") for audit in plan.get("backing_strip_audits", [])]
    record["backing_rim_chord_repairs"] = [
        audit.get("source_chord_repair") for audit in plan.get("backing_strip_audits", [])]
    record["compact_peg_omitted"] = not bool(plan["compact_peg_enabled"])
    record.update(wedge_quality)
    return record

def _orient_new_cap_against_existing_rim(
    output_faces: list[list[int]],
    face_start: int,
    rim_ids: list[int],
) -> bool:
    """Flip a new cap when its shared rim follows the existing wall direction."""

    rim_edges = {
        tuple(sorted((int(rim_ids[index]), int(rim_ids[(index + 1) % len(rim_ids)]))))
        for index in range(len(rim_ids))
    }
    existing_direction: dict[tuple[int, int], tuple[int, int]] = {}
    for face in output_faces[:face_start]:
        for edge in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            key = tuple(sorted(edge))
            if key in rim_edges:
                existing_direction[key] = edge
    same = 0
    opposite = 0
    for face in output_faces[face_start:]:
        for edge in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            key = tuple(sorted(edge))
            previous = existing_direction.get(key)
            if previous is None:
                continue
            if edge == previous:
                same += 1
            else:
                opposite += 1
    if same > opposite:
        for face in output_faces[face_start:]:
            face[1], face[2] = face[2], face[1]
        return True
    return False
