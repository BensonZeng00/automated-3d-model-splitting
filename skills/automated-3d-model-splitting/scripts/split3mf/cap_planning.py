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

def plan_cap_decision(
    points: np.ndarray,
    source_vertex_ids: list[int] | tuple[int, ...],
    fallback_inward: np.ndarray,
    inward_directions: np.ndarray,
    fixed_depth_mm: float,
    flat_clearance_mm: float,
    cap_mode: str,
    planar_extra_limit_mm: float,
    parent_thickness_probe: ParentThicknessProbe | None = None,
    source_points: np.ndarray | None = None,
    safety_ceiling_mm: float | None = None,
) -> CapDecision:
    """Plan one reusable child cap decision keyed by source vertex id."""
    vertex_ids = tuple(int(value) for value in source_vertex_ids)
    if len(vertex_ids) != len(set(vertex_ids)):
        raise ValueError("cap decision source vertex ids must be unique")
    if len(vertex_ids) != len(points):
        raise ValueError("cap decision vertex-id count does not match boundary points")
    decision_source_points = np.asarray(
        points if source_points is None else source_points,
        dtype=np.float64,
    )
    if decision_source_points.shape != np.asarray(points).shape:
        raise ValueError("cap decision source points do not match boundary points")
    if not np.isfinite(decision_source_points).all():
        raise ValueError("cap decision source points must be finite")
    distances, directions, record = boundary_cap_distances(
        points,
        fallback_inward,
        inward_directions,
        fixed_depth_mm,
        flat_clearance_mm,
        cap_mode,
        planar_extra_limit_mm,
        parent_thickness_probe,
        safety_ceiling_mm,
    )
    return CapDecision(
        mode=str(record["cap_mode"]),
        source_vertex_ids=vertex_ids,
        fit_points=np.asarray(points, dtype=np.float64).copy(),
        directions=np.asarray(directions, dtype=np.float64).copy(),
        distances=np.asarray(distances, dtype=np.float64).copy(),
        record=dict(record),
        source_points=decision_source_points.copy(),
    )

def cap_decision_patch_quality(
    decision: CapDecision,
    fallback_normal: np.ndarray,
    visible_top_points: np.ndarray,
    lead_in_directions: np.ndarray,
    lead_in_mm: float,
    lead_in_applied: bool,
    source_mesh_vertices: np.ndarray | None = None,
    source_mesh_faces: np.ndarray | None = None,
) -> dict:
    """Preflight all generated triangles from the visible ring to the cap."""
    fit_points = np.asarray(decision.fit_points, dtype=np.float64)
    directions = np.asarray(decision.directions, dtype=np.float64)
    distances = np.asarray(decision.distances, dtype=np.float64)
    top_points = np.asarray(visible_top_points, dtype=np.float64)
    lead_directions = np.asarray(lead_in_directions, dtype=np.float64)
    bottom_points = fit_points + directions * distances[:, None]
    count = int(len(fit_points))
    if not (
        top_points.shape == fit_points.shape == directions.shape
        and lead_directions.shape == fit_points.shape
        and distances.shape == (count,)
    ):
        raise ValueError("cap patch rings must have matching shapes")

    if source_mesh_vertices is None:
        vertices: list[np.ndarray] = [point.copy() for point in top_points]
        faces: list[list[int]] = []
        top_ids = list(range(count))
        source_face_count = 0
    else:
        source_vertex_array = np.asarray(source_mesh_vertices, dtype=np.float64)
        source_face_array = np.asarray(source_mesh_faces, dtype=np.int64)
        vertices = [point.copy() for point in source_vertex_array]
        faces = source_face_array.astype(int).tolist()
        source_face_count = int(len(faces))
        top_ids = list(range(len(vertices), len(vertices) + count))
        vertices.extend(point.copy() for point in top_points)
    side_rings = [top_points]
    if lead_in_applied:
        if bool(decision.record.get("guided_internal_cut_applied", False)):
            lead_points = (top_points + bottom_points) * 0.5
        else:
            lead_points, _lead_depths, _lead_directions, _lead_taper_record = coherent_tapered_lead_ring(
                top_points,
                fit_points,
                lead_directions,
                distances,
                lead_in_mm,
                reference_axis=fallback_normal,
            )
        lead_ids = list(range(len(vertices), len(vertices) + count))
        vertices.extend(point.copy() for point in lead_points)
        add_side_faces_between_rings(faces, top_ids, lead_ids, vertices=vertices)
        side_top_ids = lead_ids
        side_rings.append(lead_points)
    else:
        side_top_ids = top_ids
    side_rings.append(bottom_points)
    side_wall_quality = cap_side_wall_quality(side_rings)
    bottom_ids = list(range(len(vertices), len(vertices) + count))
    vertices.extend(point.copy() for point in bottom_points)
    add_side_faces_between_rings(faces, side_top_ids, bottom_ids, vertices=vertices)
    side_face_count = int(len(faces) - source_face_count)

    if (
        decision.cap_template_points is not None
        and decision.cap_template_faces is not None
        and decision.cap_template_boundary_ids is not None
    ):
        cap_face_count, cap_surface_quality = append_harmonic_cap_template(
            vertices,
            faces,
            np.asarray(decision.cap_template_points, dtype=np.float64),
            np.asarray(decision.cap_template_faces, dtype=np.int64),
            decision.cap_template_boundary_ids,
            bottom_ids,
            target_boundary_points=bottom_points,
        )
        cap_triangles = [None] * int(cap_face_count)
        expected_cap_faces = int(len(decision.cap_template_faces))
        triangulation_record = {
            "status": "source_patch_harmonic_template",
            "face_count": int(cap_face_count),
        }
    else:
        cap_triangles, _normal, triangulation_record = triangulate_ordered_loop_3d(
            bottom_points,
            np.asarray(fallback_normal, dtype=np.float64),
        )
        cap_surface_quality = inward_cap_surface_quality(
            bottom_points,
            cap_triangles,
            _normal,
        )
        faces.extend(
            [bottom_ids[int(a)], bottom_ids[int(b)], bottom_ids[int(c)]]
            for a, b, c in cap_triangles
        )
        expected_cap_faces = max(count - 2, 0)
    probe_mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    total_faces = int(len(probe_mesh.faces))
    probe_mesh.merge_vertices(digits_vertex=6)
    after_vertex_merge = int(len(probe_mesh.faces))
    probe_faces = np.asarray(probe_mesh.faces, dtype=np.int64)
    if len(probe_faces):
        _unique_faces, keep_indices = np.unique(
            np.sort(probe_faces, axis=1),
            axis=0,
            return_index=True,
        )
        if len(keep_indices) != len(probe_faces):
            probe_mesh.update_faces(np.sort(keep_indices))
    after_duplicate_removal = int(len(probe_mesh.faces))
    nondegenerate_mask = np.asarray(probe_mesh.nondegenerate_faces(), dtype=bool)
    nondegenerate_faces = int(np.count_nonzero(nondegenerate_mask))
    degenerate_indices = np.flatnonzero(~nondegenerate_mask)
    source_end = int(source_face_count)
    side_end = int(source_face_count + side_face_count)
    degenerate_source_faces = int(np.count_nonzero(degenerate_indices < source_end))
    degenerate_side_faces = int(
        np.count_nonzero(
            (degenerate_indices >= source_end) & (degenerate_indices < side_end)
        )
    )
    degenerate_cap_faces = int(np.count_nonzero(degenerate_indices >= side_end))
    degenerate_cap_samples = [
        {
            "face_index": int(face_index),
            "vertex_ids": [
                int(value) for value in probe_mesh.faces[int(face_index)].tolist()
            ],
            "points": np.asarray(
                probe_mesh.vertices[probe_mesh.faces[int(face_index)]],
                dtype=np.float64,
            ).round(9).tolist(),
        }
        for face_index in degenerate_indices[degenerate_indices >= side_end][:8]
    ]
    duplicate_faces = max(after_vertex_merge - after_duplicate_removal, 0)
    degenerate_faces = max(
        after_duplicate_removal - nondegenerate_faces,
        0,
    )
    generated_faces = max(total_faces - source_face_count, 0)
    return {
        "valid": bool(
            len(cap_triangles) == expected_cap_faces
            and nondegenerate_faces == after_duplicate_removal
            and after_duplicate_removal == after_vertex_merge
            and cap_surface_quality["valid"]
            and side_wall_quality["valid"]
        ),
        "boundary_vertices": count,
        "side_faces": side_face_count,
        "source_faces": source_face_count,
        "generated_faces": generated_faces,
        "expected_cap_faces": int(expected_cap_faces),
        "cap_faces": int(len(cap_triangles)),
        "total_faces": total_faces,
        "after_vertex_merge": after_vertex_merge,
        "after_duplicate_removal": after_duplicate_removal,
        "nondegenerate_faces": nondegenerate_faces,
        "degenerate_faces": degenerate_faces,
        "degenerate_source_faces": degenerate_source_faces,
        "degenerate_side_faces": degenerate_side_faces,
        "degenerate_cap_faces": degenerate_cap_faces,
        "degenerate_cap_samples": degenerate_cap_samples,
        "duplicate_faces": duplicate_faces,
        "invalid_faces": degenerate_faces
        + duplicate_faces
        + abs(int(len(cap_triangles)) - int(expected_cap_faces))
        + (0 if cap_surface_quality["valid"] else 1)
        + (0 if side_wall_quality["valid"] else 1),
        "lead_in_applied": bool(lead_in_applied),
        "coordinate_precision_decimals": 6,
        "triangulation": dict(triangulation_record),
        "cap_surface_quality": cap_surface_quality,
        "side_wall_quality": side_wall_quality,
    }

def side_quad_triangulation_choice(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> tuple[int, tuple[float, float, float], tuple[float, float]]:
    """Choose the same smooth diagonal used by production side-wall output."""
    points = np.asarray([a0, a1, b0, b1], dtype=np.float64)
    splits = (
        ((0, 1, 3), (0, 3, 2), (0, 3)),
        ((0, 1, 2), (1, 3, 2), (1, 2)),
    )
    scored: list[tuple[tuple[float, float, float], tuple[float, float]]] = []
    for first, second, diagonal in splits:
        normals: list[np.ndarray] = []
        areas: list[float] = []
        for face in (first, second):
            triangle = points[np.asarray(face, dtype=np.int64)]
            normal = np.cross(
                triangle[1] - triangle[0],
                triangle[2] - triangle[0],
            )
            area = float(np.linalg.norm(normal))
            areas.append(area)
            normals.append(normal / area if area > 1e-14 else np.zeros(3))
        agreement = float(np.dot(normals[0], normals[1]))
        diagonal_length = float(
            np.linalg.norm(points[int(diagonal[1])] - points[int(diagonal[0])])
        )
        scored.append(
            (
                (agreement, min(areas), -diagonal_length),
                (areas[0], areas[1]),
            )
        )
    selected = int(scored[1][0] > scored[0][0])
    score, areas = scored[selected]
    return selected, score, areas

def cap_side_wall_quality(rings: list[np.ndarray]) -> dict:
    """Audit the cheap but critical ring strips independently of cap triangulation.

    This stays linear in boundary size, so dense vendor loops do not get a free
    pass.  It catches the long circumferential spikes and collapsed quads which
    can remain topologically watertight while rendering as an obvious hole.
    """
    arrays = [np.asarray(ring, dtype=np.float64) for ring in rings]
    count = int(len(arrays[0])) if arrays else 0
    shapes_match = bool(
        count >= 3
        and all(ring.shape == (count, 3) for ring in arrays)
        and len(arrays) >= 2
    )
    finite = bool(shapes_match and all(np.isfinite(ring).all() for ring in arrays))
    if not finite:
        return {
            "valid": False,
            "reason": "invalid_or_mismatched_side_rings",
            "degenerate_triangles": 1,
            "long_circumferential_edges": 0,
            "circumferential_edge_stretch_max": float("inf"),
            "conflicting_quad_normals": 0,
        }

    source_edges = np.linalg.norm(np.roll(arrays[0], -1, axis=0) - arrays[0], axis=1)
    positive_source = source_edges[source_edges > 1e-9]
    median_source_edge = float(np.median(positive_source)) if len(positive_source) else 0.0
    long_threshold = max(5.0, 10.0 * median_source_edge)
    degenerate = 0
    conflicting = 0
    long_edges = 0
    maximum_stretch = 1.0
    minimum_area2 = float("inf")
    maximum_circumferential_edge = 0.0
    coincident_layers_skipped = 0
    strip_records: list[dict] = []

    for strip_index, (upper, lower) in enumerate(zip(arrays[:-1], arrays[1:])):
        if float(np.max(np.linalg.norm(lower - upper, axis=1))) <= 1e-9:
            # The production profile may intentionally collapse an unused
            # zero-clearance lead layer.  It contributes no geometric strip;
            # audit the next non-coincident layer instead.
            coincident_layers_skipped += 1
            strip_records.append(
                {"strip_index": int(strip_index), "coincident": True}
            )
            continue
        upper_next = np.roll(upper, -1, axis=0)
        lower_next = np.roll(lower, -1, axis=0)
        upper_edges = np.linalg.norm(upper_next - upper, axis=1)
        lower_edges = np.linalg.norm(lower_next - lower, axis=1)
        maximum_circumferential_edge = max(
            maximum_circumferential_edge,
            float(upper_edges.max(initial=0.0)),
            float(lower_edges.max(initial=0.0)),
        )
        long_edges += int(np.count_nonzero(lower_edges > long_threshold + 1e-9))
        ratios = np.maximum(
            lower_edges / np.maximum(upper_edges, 1e-12),
            upper_edges / np.maximum(lower_edges, 1e-12),
        )
        maximum_stretch = max(maximum_stretch, float(ratios.max(initial=1.0)))

        selected_quality = [
            side_quad_triangulation_choice(a0, a1, b0, b1)
            for a0, a1, b0, b1 in zip(
                upper,
                upper_next,
                lower,
                lower_next,
            )
        ]
        area2_a = np.asarray(
            [quality[2][0] for quality in selected_quality],
            dtype=np.float64,
        )
        area2_b = np.asarray(
            [quality[2][1] for quality in selected_quality],
            dtype=np.float64,
        )
        minimum_area2 = min(
            minimum_area2,
            float(area2_a.min(initial=np.inf)),
            float(area2_b.min(initial=np.inf)),
        )
        degenerate += int(np.count_nonzero(area2_a <= 1e-10))
        degenerate += int(np.count_nonzero(area2_b <= 1e-10))
        normal_dot = np.asarray(
            [quality[1][0] for quality in selected_quality],
            dtype=np.float64,
        )
        conflicting += int(np.count_nonzero(normal_dot < -0.25))
        strip_records.append(
            {
                "strip_index": int(strip_index),
                "coincident": False,
                "maximum_upper_edge_mm": float(upper_edges.max(initial=0.0)),
                "maximum_lower_edge_mm": float(lower_edges.max(initial=0.0)),
                "long_lower_edges": int(
                    np.count_nonzero(lower_edges > long_threshold + 1e-9)
                ),
                "maximum_edge_stretch": float(ratios.max(initial=1.0)),
                "conflicting_quad_normals": int(
                    np.count_nonzero(normal_dot < -0.25)
                ),
            }
        )

    audited_quad_count = count * max(len(arrays) - 1 - coincident_layers_skipped, 0)
    conflicting_limit = max(2, int(math.ceil(audited_quad_count * 0.01)))
    valid = bool(
        degenerate == 0
        and long_edges == 0
        and maximum_stretch <= 10.0 + 1e-9
        and conflicting <= conflicting_limit
    )
    return {
        "valid": valid,
        "boundary_vertices": count,
        "ring_count": int(len(arrays)),
        "median_source_boundary_edge_mm": median_source_edge,
        "long_edge_threshold_mm": long_threshold,
        "maximum_circumferential_edge_mm": maximum_circumferential_edge,
        "long_circumferential_edges": int(long_edges),
        "circumferential_edge_stretch_max": float(maximum_stretch),
        "minimum_triangle_double_area_mm2": float(minimum_area2),
        "degenerate_triangles": int(degenerate),
        "conflicting_quad_normals": int(conflicting),
        "conflicting_quad_normals_limit": int(conflicting_limit),
        "audited_quad_count": int(audited_quad_count),
        "coincident_ring_layers_skipped": int(coincident_layers_skipped),
        "strips": strip_records,
    }

def cap_decision_patch_quality_preflight(
    decision: CapDecision,
    fallback_normal: np.ndarray,
    visible_top_points: np.ndarray,
    lead_in_directions: np.ndarray,
    lead_in_mm: float,
    lead_in_applied: bool,
    source_mesh_vertices: np.ndarray | None = None,
    source_mesh_faces: np.ndarray | None = None,
    defer_cap_triangulation: bool = False,
) -> dict:
    """Preflight a connector patch without duplicating authoritative large-cap work.

    Dense vendor boundaries can contain roughly a thousand samples.  Running the
    general concave ear-clipper across every candidate projection is cubic in the
    worst case and used to stall before the actual connector builder was reached.
    The production builder already triangulates and validates the completed solid,
    so large-loop planning only needs a bounded structural check here.  Ordinary
    loops keep the complete patch-quality preflight and retopology validation.
    """
    if not defer_cap_triangulation:
        return cap_decision_patch_quality(
            decision,
            fallback_normal,
            visible_top_points,
            lead_in_directions,
            lead_in_mm,
            lead_in_applied,
            source_mesh_vertices=source_mesh_vertices,
            source_mesh_faces=source_mesh_faces,
        )

    fit_points = np.asarray(decision.fit_points, dtype=np.float64)
    directions = np.asarray(decision.directions, dtype=np.float64)
    distances = np.asarray(decision.distances, dtype=np.float64)
    top_points = np.asarray(visible_top_points, dtype=np.float64)
    lead_directions = np.asarray(lead_in_directions, dtype=np.float64)
    count = int(len(fit_points))
    shapes_match = bool(
        top_points.shape == fit_points.shape == directions.shape
        and lead_directions.shape == fit_points.shape
        and distances.shape == (count,)
    )
    finite = bool(
        shapes_match
        and np.isfinite(top_points).all()
        and np.isfinite(fit_points).all()
        and np.isfinite(directions).all()
        and np.isfinite(distances).all()
    )
    positive_directions = bool(
        finite and np.all(np.linalg.norm(directions, axis=1) > 1e-12)
    )
    bottom_points = (
        fit_points + directions * distances[:, None]
        if finite
        else np.empty((0, 3), dtype=np.float64)
    )
    side_rings = [top_points]
    if lead_in_applied and finite:
        if bool(decision.record.get("guided_internal_cut_applied", False)):
            lead_points = (top_points + bottom_points) * 0.5
        else:
            lead_points, _lead_depths, _lead_directions, _lead_taper_record = (
                coherent_tapered_lead_ring(
                    top_points,
                    fit_points,
                    lead_directions,
                    distances,
                    lead_in_mm,
                    reference_axis=fallback_normal,
                )
            )
        side_rings.append(lead_points)
    side_rings.append(bottom_points)
    side_wall_quality = cap_side_wall_quality(side_rings)
    harmonic_template = bool(
        decision.cap_template_points is not None
        and decision.cap_template_faces is not None
        and decision.cap_template_boundary_ids is not None
    )
    if harmonic_template:
        _template_points, cap_surface_quality = harmonic_patch_deformation(
            np.asarray(decision.cap_template_points, dtype=np.float64),
            np.asarray(decision.cap_template_faces, dtype=np.int64),
            decision.cap_template_boundary_ids,
            bottom_points,
        )
    else:
        bottom_normal = (
            fit_plane_normal(bottom_points, fallback_normal)
            if len(bottom_points) >= 3
            else np.asarray(fallback_normal, dtype=np.float64)
        )
        cap_surface_quality = inward_cap_surface_quality(
            bottom_points,
            [],
            bottom_normal,
        )
    side_layers = 2 if lead_in_applied else 1
    side_faces = int(2 * count * side_layers)
    source_face_count = int(
        0 if source_mesh_faces is None else len(np.asarray(source_mesh_faces))
    )
    structurally_valid = bool(
        count >= 3
        and finite
        and positive_directions
        and bool(cap_surface_quality["valid"])
        and bool(side_wall_quality["valid"])
    )
    return {
        "valid": structurally_valid,
        "boundary_vertices": count,
        "side_faces": side_faces,
        "source_faces": source_face_count,
        "generated_faces": side_faces,
        "expected_cap_faces": (
            int(len(decision.cap_template_faces))
            if harmonic_template
            else max(count - 2, 0)
        ),
        "cap_faces": 0,
        "total_faces": source_face_count + side_faces,
        "after_vertex_merge": source_face_count + side_faces,
        "after_duplicate_removal": source_face_count + side_faces,
        "nondegenerate_faces": source_face_count + side_faces,
        "degenerate_faces": 0 if structurally_valid else 1,
        # Keep the deferred large-loop record schema identical to the full
        # triangulation record.  The cap itself has deliberately not been
        # triangulated yet, so only the bounded side-wall audit can report
        # concrete degenerate triangles at this stage.
        "degenerate_source_faces": 0,
        "degenerate_side_faces": int(
            side_wall_quality.get("degenerate_triangles", 0)
        ),
        "degenerate_cap_faces": 0,
        "degenerate_cap_samples": [],
        "duplicate_faces": 0,
        "invalid_faces": 0 if structurally_valid else 1,
        "lead_in_applied": bool(lead_in_applied),
        "coordinate_precision_decimals": 6,
        "triangulation": {
            "status": (
                "source_patch_harmonic_template_prevalidated"
                if harmonic_template
                else "deferred_to_authoritative_connector_builder"
            ),
            "reason": "large_vendor_boundary_preflight_budget",
        },
        "cap_surface_quality": cap_surface_quality,
        "side_wall_quality": side_wall_quality,
        "cap_triangulation_deferred": True,
        "authoritative_solid_validation_required": True,
    }

def remap_cap_decision(
    decision: CapDecision,
    source_vertex_ids: list[int] | tuple[int, ...],
    requested_fit_points: np.ndarray | None = None,
    requested_source_points: np.ndarray | None = None,
    geometric_tolerance_mm: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return a shared cap decision in the caller's boundary order.

    Vendor paint topology can leave a parent loop with an extra T-joint
    vertex, a coordinate alias, or a short alternate triangle-edge route.
    Exact ids remain the preferred path.  When caller points are supplied,
    reconcile only bounded source-loop differences within the caller-provided
    tolerance while preserving the already planned child cap field.
    """
    requested_ids = tuple(int(value) for value in source_vertex_ids)
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("matching cap boundary contains duplicate source vertex ids")
    requested_set = set(requested_ids)
    source_index = {
        int(global_id): index
        for index, global_id in enumerate(decision.source_vertex_ids)
    }
    missing = [global_id for global_id in requested_ids if global_id not in source_index]
    extra = [
        global_id
        for global_id in decision.source_vertex_ids
        if global_id not in requested_set
    ]
    if (missing or extra) and requested_fit_points is None:
        raise ValueError(
            "matching parent socket and child cap source vertices differ: "
            f"missing={missing}, extra={extra}"
        )

    decision_points = np.asarray(decision.fit_points, dtype=np.float64)
    decision_directions = np.asarray(decision.directions, dtype=np.float64)
    decision_distances = np.asarray(decision.distances, dtype=np.float64)
    if not (len(decision_points) == len(decision_directions) == len(decision_distances)):
        raise ValueError("cap decision geometry arrays have different lengths")

    reconciliation_record = None
    if missing or extra:
        tolerance = float(geometric_tolerance_mm)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("cap boundary geometric tolerance must be positive and finite")
        requested_points = np.asarray(requested_fit_points, dtype=np.float64)
        if requested_points.shape != (len(requested_ids), 3):
            raise ValueError("matching cap boundary points do not match source vertex ids")
        if not np.isfinite(requested_points).all():
            raise ValueError("matching cap boundary points must be finite")

        decision_source_points = np.asarray(
            decision_points
            if decision.source_points is None
            else decision.source_points,
            dtype=np.float64,
        )
        requested_matching_points = np.asarray(
            requested_points
            if requested_source_points is None
            else requested_source_points,
            dtype=np.float64,
        )
        if decision_source_points.shape != decision_points.shape:
            raise ValueError("cap decision source points do not match fit points")
        if requested_matching_points.shape != requested_points.shape:
            raise ValueError("matching source boundary points do not match fit points")
        if not (
            np.isfinite(decision_source_points).all()
            and np.isfinite(requested_matching_points).all()
        ):
            raise ValueError("cap boundary matching points must be finite")

        segment_starts = decision_source_points
        segment_ends = np.roll(decision_source_points, -1, axis=0)
        segment_vectors = segment_ends - segment_starts
        segment_lengths_sq = np.sum(segment_vectors * segment_vectors, axis=1)

        print_match_record = None
        print_match_attempted = False
        def within_correspondence_limit(error):
            nonlocal tolerance, print_match_record, print_match_attempted
            if error <= tolerance:
                return True
            if not print_match_attempted:
                from .boundary_correspondence import source_loop_print_match
                print_match_attempted = True
                print_match_record = source_loop_print_match(
                    decision_source_points, requested_matching_points, tolerance)
                if print_match_record is not None:
                    tolerance = print_match_record["tolerance_mm"]
            return error <= tolerance

        def nearest_segment(point: np.ndarray) -> tuple[int, float, float]:
            offsets = point[None, :] - segment_starts
            parameters = np.divide(
                np.sum(offsets * segment_vectors, axis=1),
                segment_lengths_sq,
                out=np.zeros(len(segment_starts), dtype=np.float64),
                where=segment_lengths_sq > 1e-24,
            )
            parameters = np.clip(parameters, 0.0, 1.0)
            projections = segment_starts + parameters[:, None] * segment_vectors
            errors = np.linalg.norm(projections - point[None, :], axis=1)
            segment_index = int(np.argmin(errors))
            return segment_index, float(parameters[segment_index]), float(errors[segment_index])

        fit_points = np.empty_like(requested_points)
        directions = np.empty_like(requested_points)
        distances = np.empty(len(requested_ids), dtype=np.float64)
        alias_count = 0
        projected_vertex_count = 0
        maximum_error = 0.0

        decision_bottom_points = (
            decision_points + decision_directions * decision_distances[:, None]
        )
        flat_mode = str(decision.mode) == "flat"
        flat_normal = None
        flat_plane_s = None
        if flat_mode:
            flat_normal = np.asarray(decision_directions[0], dtype=np.float64)
            flat_normal /= max(float(np.linalg.norm(flat_normal)), 1e-12)
            flat_plane_s = float(np.mean(decision_bottom_points @ flat_normal))

        for requested_index, (global_id, requested_point, matching_point) in enumerate(
            zip(requested_ids, requested_points, requested_matching_points)
        ):
            direct_index = source_index.get(int(global_id))
            if direct_index is not None:
                fit_points[requested_index] = decision_points[direct_index]
                directions[requested_index] = decision_directions[direct_index]
                distances[requested_index] = decision_distances[direct_index]
                continue

            vertex_errors = np.linalg.norm(
                decision_source_points - matching_point[None, :],
                axis=1,
            )
            nearest_vertex = int(np.argmin(vertex_errors))
            nearest_vertex_error = float(vertex_errors[nearest_vertex])
            if nearest_vertex_error <= float(geometric_tolerance_mm):
                fit_points[requested_index] = decision_points[nearest_vertex]
                directions[requested_index] = decision_directions[nearest_vertex]
                distances[requested_index] = decision_distances[nearest_vertex]
                alias_count += 1
                maximum_error = max(maximum_error, nearest_vertex_error)
                continue

            segment_index, parameter, projection_error = nearest_segment(matching_point)
            if not within_correspondence_limit(projection_error):
                raise ValueError(
                    "matching parent socket and child cap source vertices differ beyond "
                    f"{tolerance:.6f} mm: missing={missing}, extra={extra}, "
                    f"vertex={global_id}, nearest_error_mm={projection_error:.9f}"
                )
            next_index = (segment_index + 1) % len(decision_points)
            direction = (
                (1.0 - parameter) * decision_directions[segment_index]
                + parameter * decision_directions[next_index]
            )
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            fit_points[requested_index] = (
                (1.0 - parameter) * decision_points[segment_index]
                + parameter * decision_points[next_index]
            )
            directions[requested_index] = direction
            if flat_mode:
                distances[requested_index] = (
                    float(flat_plane_s)
                    - float(
                        np.dot(
                            fit_points[requested_index],
                            np.asarray(flat_normal),
                        )
                    )
                )
            else:
                distances[requested_index] = (
                    (1.0 - parameter) * decision_distances[segment_index]
                    + parameter * decision_distances[next_index]
                )
            projected_vertex_count += 1
            maximum_error = max(maximum_error, projection_error)

        requested_segment_starts = requested_matching_points
        requested_segment_ends = np.roll(requested_matching_points, -1, axis=0)
        requested_segment_vectors = requested_segment_ends - requested_segment_starts
        requested_segment_lengths_sq = np.sum(
            requested_segment_vectors * requested_segment_vectors,
            axis=1,
        )
        uncovered_extra_ids = []
        for global_id in extra:
            decision_index = source_index[int(global_id)]
            point = decision_source_points[decision_index]
            offsets = point[None, :] - requested_segment_starts
            parameters = np.divide(
                np.sum(offsets * requested_segment_vectors, axis=1),
                requested_segment_lengths_sq,
                out=np.zeros(len(requested_segment_starts), dtype=np.float64),
                where=requested_segment_lengths_sq > 1e-24,
            )
            parameters = np.clip(parameters, 0.0, 1.0)
            projections = (
                requested_segment_starts
                + parameters[:, None] * requested_segment_vectors
            )
            coverage_error = float(
                np.min(np.linalg.norm(projections - point[None, :], axis=1))
            )
            maximum_error = max(maximum_error, coverage_error)
            if not within_correspondence_limit(coverage_error):
                uncovered_extra_ids.append(int(global_id))
        if uncovered_extra_ids:
            raise ValueError(
                "matching parent socket and child cap source vertices differ beyond "
                f"{tolerance:.6f} mm: missing={missing}, "
                f"uncovered_extra={uncovered_extra_ids}"
            )
        if np.any(distances <= 0.0) or not (
            np.isfinite(fit_points).all()
            and np.isfinite(directions).all()
            and np.isfinite(distances).all()
        ):
            raise ValueError("reconciled cap decision contains unsafe geometry")
        reconciliation_record = {
            "status": "bounded_source_loop_reconciled",
            "tolerance_mm": tolerance,
            "missing_source_vertex_ids": [int(value) for value in missing],
            "extra_source_vertex_ids": [int(value) for value in extra],
            "coordinate_alias_count": int(alias_count),
            "projected_source_vertex_count": int(projected_vertex_count),
            "maximum_source_projection_error_mm": float(maximum_error),
            "print_correspondence_proof": print_match_record,
        }
    else:
        order = np.asarray(
            [source_index[global_id] for global_id in requested_ids],
            dtype=np.int64,
        )
        fit_points = decision_points[order].copy()
        directions = decision_directions[order].copy()
        distances = decision_distances[order].copy()

    record = dict(decision.record)
    if str(record.get("cap_mode")) != str(decision.mode):
        raise ValueError("cap decision mode and record disagree")
    record["cap_decision_source"] = "prevalidated_shared_child"
    if reconciliation_record is not None:
        record["cap_boundary_reconciliation"] = reconciliation_record
    return fit_points, distances, directions, record

def reconcile_planned_internal_fit_points(
    recomputed_fit_points: np.ndarray,
    planned_fit_points: np.ndarray,
    internal_boundary_points: np.ndarray,
    interior_conormals: np.ndarray,
    base_insert_shrink_mm: float,
    plane_record: dict,
    *,
    mismatch_label: str = "boundary fit points changed after cap planning",
    allow_user_reviewed_reconciliation: bool = False,
    maximum_reconciliation_error_mm: float = 0.0,
) -> np.ndarray:
    """Accept only a prevalidated non-default internal fit inset.

    Ordinary cap plans must still reproduce the build-stage fit ring exactly.
    Thin-boundary fallback and an explicit guided internal cut are the only
    exceptions: both deliberately recess the hidden fit ring farther inward
    while preserving the visible source ring.  Recompute the documented offset
    before trusting the planned ring so a stale or unrelated cap decision
    cannot bypass the geometry invariant.
    """
    recomputed = np.asarray(recomputed_fit_points, dtype=np.float64)
    planned = np.asarray(planned_fit_points, dtype=np.float64)
    if recomputed.shape != planned.shape:
        raise ValueError(mismatch_label)
    if np.allclose(recomputed, planned, atol=1e-9, rtol=0.0):
        return planned.copy()
    point_errors = np.linalg.norm(planned - recomputed, axis=1)
    maximum_error = float(point_errors.max(initial=0.0))
    reconciliation_limit = max(float(maximum_reconciliation_error_mm), 0.0)
    if (
        allow_user_reviewed_reconciliation
        and np.isfinite(planned).all()
        and np.isfinite(maximum_error)
        and maximum_error <= reconciliation_limit + 1e-9
    ):
        runtime_log(
            "可视验证",
            "planned_fit_ring_reconciled",
            "用户已确认的边界采用预验证隐藏配合环",
            maximum_fit_point_difference_mm=maximum_error,
            reconciliation_limit_mm=reconciliation_limit,
            fit_point_count=int(len(planned)),
        )
        return planned.copy()
    base_shrink_mm = max(float(base_insert_shrink_mm), 0.0)
    authorized_internal_inset = bool(
        plane_record.get("thin_boundary_adaptive_inset_applied", False)
        or (
            plane_record.get("guided_internal_cut_applied", False)
            and plane_record.get(
                "guided_internal_cut_visible_boundary_locked", False
            )
        )
    )
    if not authorized_internal_inset:
        raise ValueError(mismatch_label)
    if bool(plane_record.get("guided_internal_cut_applied", False)):
        inset_limit = float(
            plane_record.get(
                "guided_internal_cut_entry_inset_limit_mm",
                float("nan"),
            )
        )
        reported_minimum = float(
            plane_record.get("guided_entry_inset_min_mm", float("nan"))
        )
        reported_maximum = float(
            plane_record.get("guided_entry_inset_max_mm", float("nan"))
        )
        if (
            not np.isfinite(inset_limit)
            or not np.isfinite(reported_minimum)
            or not np.isfinite(reported_maximum)
            or reported_minimum < base_shrink_mm - 1e-9
            or reported_maximum > inset_limit + 1e-9
            or not np.isfinite(planned).all()
        ):
            raise ValueError(mismatch_label)
        return planned.copy()
    effective_shrink_mm = float(
        plane_record.get("thin_boundary_effective_insert_shrink_mm", float("nan"))
    )
    if not np.isfinite(effective_shrink_mm) or effective_shrink_mm <= base_shrink_mm:
        raise ValueError(mismatch_label)
    expected = offset_points_along_conormals(
        np.asarray(internal_boundary_points, dtype=np.float64),
        np.asarray(interior_conormals, dtype=np.float64),
        effective_shrink_mm,
    )
    if expected.shape != planned.shape or not np.allclose(
        expected,
        planned,
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError(mismatch_label)
    return planned.copy()
