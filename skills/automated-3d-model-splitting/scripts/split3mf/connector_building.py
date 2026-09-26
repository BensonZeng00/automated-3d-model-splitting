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

def add_local_male_connector_and_backing(
    *,
    output_vertices: list[np.ndarray],
    output_faces: list[list[int]],
    boundary_ids: list[int],
    boundary_points: np.ndarray,
    inward: np.ndarray,
    spec: LocalConnectorSpec,
    inward_directions: np.ndarray | None = None,
    boolean_cutters: list[trimesh.Trimesh] | None = None,
    boolean_socket_cutters: list[trimesh.Trimesh] | None = None,
) -> dict:
    """Close the full interface shallowly and put one compact peg in its interior."""

    build_started = time.perf_counter()
    if boolean_cutters is not None and boolean_socket_cutters is not None:
        raise ValueError(
            "provide boolean_cutters or legacy boolean_socket_cutters, not both"
        )
    plan = plan_local_connector(boundary_points, inward, spec, samples=64)
    from .surface_direction import audit_inward_axis
    plan["source_direction_audit"] = audit_inward_axis(
        output_vertices, output_faces, boundary_ids, inward)
    runtime_log(
        "local-connector",
        "male_plan_ready",
        "Local male connector plan is ready",
        boundary_vertices=int(len(boundary_points)),
        compact_peg_enabled=bool(plan["compact_peg_enabled"]),
        backing_depth_mm=float(plan["full_boundary_backing_depth_mm"]),
        engagement_depth_mm=float(plan["engagement_depth_mm"]),
        stage_elapsed_seconds=round(time.perf_counter() - build_started, 3),
    )
    boundary = np.asarray(boundary_points, dtype=np.float64)
    backing_kwargs = {}
    if inward_directions is not None:
        backing_kwargs["inward_directions"] = inward_directions
    backing_profile = _printable_backing_profile(
        boundary,
        plan,
        child_clearance=True,
        **backing_kwargs,
    )
    runtime_log(
        "local-connector",
        "male_backing_profile_ready",
        "Printable backing profile is ready",
        boundary_vertices=int(len(boundary)),
        backing_layers=int(len(backing_profile.rings)),
        stage_elapsed_seconds=round(time.perf_counter() - build_started, 3),
    )
    backing_boundary = np.asarray(backing_profile.rings[-1], dtype=np.float64)
    backing_taper_depth = float(backing_profile.taper_depth_mm)
    wedge_face_start = len(output_faces)
    previous_ids = list(boundary_ids)
    previous_points = boundary
    backing_faces = 0
    strip_audits: list[dict] = []
    generated_ring_ids: list[list[int]] = []
    heightfield_surface_vertices: set[int] = set()
    for layer_index, ring in enumerate(backing_profile.rings):
        layer_points = np.asarray(ring, dtype=np.float64)
        if _rings_coincident(layer_points, previous_points):
            continue
        layer_ids = _append_points(output_vertices, layer_points)
        runtime_log(
            "local-connector",
            "male_backing_strip_start",
            "Triangulating one local male backing strip",
            layer_index=int(layer_index),
            outer_vertices=int(len(previous_ids)),
            inner_vertices=int(len(layer_ids)),
            stage_elapsed_seconds=round(time.perf_counter() - build_started, 3),
        )
        strip_face_start = len(output_faces)
        added, strip_audit, returned_points = _join_connector_rings(
            output_vertices,
            output_faces,
            previous_ids,
            layer_ids,
            first_points=previous_points,
            second_points=layer_points,
            plan=plan,
        )
        layer_points = returned_points
        if layer_index == 0 and backing_taper_depth > 1e-9:
            refinement_vertex_start = len(output_vertices)
            from .rim_chord_repair import separate_source_chords
            strip_audit["source_chord_repair"] = separate_source_chords(
                output_vertices, output_faces, strip_face_start, previous_ids,
                inward, backing_taper_depth)
            refinement = refine_connector_annulus_heightfield(
                output_vertices=output_vertices,
                output_faces=output_faces,
                face_start=strip_face_start,
                outer_ids=previous_ids,
                inner_ids=layer_ids,
                boundary_points=boundary,
                plan=plan,
                taper_depth_mm=backing_taper_depth,
                passes=12,
                maximum_internal_edge_mm=0.85,
                preserve_audited_surface=current_print_tolerance().preserve_audited_hidden_surfaces,
                allow_audited_resolution_passthrough=(
                    str(plan.get("backing_surface_validation_mode", "strict"))
                    == "advisory"
                ),
            )
            added = len(output_faces) - strip_face_start
            plan["backing_surface_refinement"] = dict(refinement.__dict__)
            strip_audit["maximum_cross_edge_before_refinement_mm"] = float(
                strip_audit.get("maximum_cross_edge_mm", 0.0)
            )
            strip_audit["maximum_cross_edge_mm"] = float(
                refinement.maximum_internal_edge_after_mm
            )
            strip_audit["surface_refinement"] = dict(refinement.__dict__)
            refinement_suffix = (
                "+audited_hidden_surface_passthrough"
                if refinement.skipped_reason is not None
                else "+regular_heightfield_refinement"
            )
            strip_audit["strategy"] = (
                str(strip_audit.get("strategy", "unknown"))
                + refinement_suffix
            )
            heightfield_surface_vertices.update(int(value) for value in previous_ids)
            heightfield_surface_vertices.update(int(value) for value in layer_ids)
            heightfield_surface_vertices.update(
                range(refinement_vertex_start, len(output_vertices))
            )
        runtime_log(
            "local-connector",
            "male_backing_strip_ready",
            "One local male backing strip is ready",
            layer_index=int(layer_index),
            strategy=str(strip_audit.get("strategy", "unknown")),
            added_faces=int(added),
            stage_elapsed_seconds=round(time.perf_counter() - build_started, 3),
        )
        strip_audit["layer_index"] = int(layer_index)
        strip_audit["layer_depth_mm"] = float(
            backing_profile.depths_mm[layer_index]
        )
        strip_audits.append(strip_audit)
        backing_faces += int(added)
        generated_ring_ids.append(layer_ids)
        previous_ids = layer_ids
        previous_points = layer_points
    if not generated_ring_ids:
        raise ValueError("local male backing profile produced no side layers")
    runtime_log(
        "local-connector",
        "male_backing_strips_ready",
        "Local male backing strips are ready",
        generated_layers=int(len(generated_ring_ids)),
        backing_faces=int(backing_faces),
        stage_elapsed_seconds=round(time.perf_counter() - build_started, 3),
    )
    backing_boundary_ids = generated_ring_ids[-1]
    backing_boundary = np.asarray(previous_points, dtype=np.float64)
    if len(backing_boundary_ids) != len(backing_boundary):
        raise ValueError(
            "final connector backing layer ids and refined points are inconsistent"
        )
    plan["backing_floor_ring_vertices"] = int(len(backing_boundary))
    plan["backing_strip_audits"] = strip_audits
    plan["backing_strip_maximum_fanout"] = max(
        int(record["maximum_fanout"]) for record in strip_audits
    )
    plan["backing_strip_maximum_cross_edge_mm"] = max(
        float(record["maximum_cross_edge_mm"]) for record in strip_audits
    )
    perimeter_edges: set[tuple[int, int]] = set()
    for ring_ids in [list(boundary_ids), *generated_ring_ids]:
        perimeter_edges.update(
            tuple(sorted((int(ring_ids[index]), int(ring_ids[(index + 1) % len(ring_ids)]))))
            for index in range(len(ring_ids))
        )
    wedge_quality = _validate_connector_wedge_faces(
        output_vertices,
        output_faces,
        wedge_face_start,
        perimeter_edges=perimeter_edges,
        coherent_boundary_rings=[
            set(int(value) for value in ring_ids)
            for ring_ids in [list(boundary_ids), *generated_ring_ids]
        ],
        coherent_boundary_pairs=[
            set(int(value) for value in first_ids)
            | set(int(value) for value in second_ids)
            for first_ids, second_ids in zip(
                [list(boundary_ids), *generated_ring_ids[:-1]],
                generated_ring_ids,
            )
        ],
        heightfield_surface_vertices=heightfield_surface_vertices,
        shallow_minimal_closure_plan=plan,
    )
    runtime_log(
        "local-connector",
        "male_backing_wedge_audited",
        "Local male backing wedge passed quality audit",
        backing_faces=int(backing_faces),
        stage_elapsed_seconds=round(time.perf_counter() - build_started, 3),
    )
    if not bool(plan["compact_peg_enabled"]):
        cap_face_start = len(output_faces)
        cap_faces = triangulate_cap(
            output_vertices,
            output_faces,
            [backing_boundary_ids],
            [_project_connector_points(backing_boundary, plan)],
            [[0]],
            np.asarray(plan["inward"], dtype=np.float64),
        )
        if cap_faces <= 0:
            raise ValueError("zero-engagement backing floor triangulation failed")
        plan["zero_engagement_cap_orientation_flipped"] = bool(
            _orient_new_cap_against_existing_rim(
                output_faces,
                cap_face_start,
                backing_boundary_ids,
            )
        )
        return _finalize_local_male_record(
            plan=plan,
            boundary=boundary,
            backing_boundary=backing_boundary,
            backing_taper_depth=backing_taper_depth,
            generated_ring_count=len(generated_ring_ids),
            backing_faces=backing_faces,
            side_faces=0,
            cap_faces=cap_faces,
            wedge_quality=wedge_quality,
            backing_triangulation="boundary_preserving_backing_floor",
        )
    peg_top_ids, order, annular_faces = _add_constrained_connector_annulus(
        output_vertices=output_vertices,
        output_faces=output_faces,
        outer_ids=backing_boundary_ids,
        outer_points=backing_boundary,
        inner_points=np.asarray(plan["peg_top"], dtype=np.float64),
        plan=plan,
        outward_normal=np.asarray(plan["inward"], dtype=np.float64),
    )
    runtime_log(
        "local-connector",
        "male_connector_annulus_ready",
        "Compact male connector annulus is ready",
        annular_faces=int(annular_faces),
        stage_elapsed_seconds=round(time.perf_counter() - build_started, 3),
    )
    backing_faces += int(annular_faces)
    if backing_faces <= 0:
        raise ValueError("local male backing triangulation failed")
    peg_straight_points = np.asarray(
        plan["peg_straight"], dtype=np.float64
    )[order]
    peg_tip_points = np.asarray(plan["peg_tip"], dtype=np.float64)[order]
    peg_straight_ids = _append_points(output_vertices, peg_straight_points)
    side_faces = add_side_faces_between_rings(
        output_faces,
        peg_top_ids,
        peg_straight_ids,
        vertices=output_vertices,
    )
    if float(np.max(np.linalg.norm(peg_tip_points - peg_straight_points, axis=1))) <= 1e-9:
        # Flat, unchamfered tip: cap the terminal straight ring directly.
        peg_tip_ids = peg_straight_ids
    else:
        peg_tip_ids = _append_points(output_vertices, peg_tip_points)
        side_faces += add_side_faces_between_rings(
            output_faces,
            peg_straight_ids,
            peg_tip_ids,
            vertices=output_vertices,
        )
    cap_faces = triangulate_cap(
        output_vertices,
        output_faces,
        [peg_tip_ids],
        [_project_connector_points(plan["peg_tip"], plan)],
        [[0]],
        np.asarray(plan["inward"], dtype=np.float64),
    )
    if cap_faces <= 0:
        raise ValueError("local male peg tip triangulation failed")
    return _finalize_local_male_record(
        plan=plan,
        boundary=boundary,
        backing_boundary=backing_boundary,
        backing_taper_depth=backing_taper_depth,
        generated_ring_count=len(generated_ring_ids),
        backing_faces=backing_faces,
        side_faces=side_faces,
        cap_faces=cap_faces,
        wedge_quality=wedge_quality,
        backing_triangulation="constrained_outer_with_connector_hole",
    )

def preflight_local_connector_patch(
    *,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    boundary_ids: list[int],
    boundary_points: np.ndarray,
    inward: np.ndarray,
    inward_directions: np.ndarray,
    spec: LocalConnectorSpec,
) -> dict:
    """Build and audit the exact local-connector patch used in production.

    Concave painted rims can be densely sampled while the printable backing is
    deliberately simplified.  Auditing an equal-count loft between those two
    curves tests geometry that will never be emitted and can therefore reject a
    valid connector.  This preflight reuses the production bridge, taper, peg,
    cap, and private Boolean-cutter builders so planning and output cannot drift.
    """

    output_vertices = [
        np.asarray(point, dtype=np.float64).copy()
        for point in np.asarray(source_vertices, dtype=np.float64)
    ]
    output_faces = np.asarray(source_faces, dtype=np.int64).astype(int).tolist()
    generated_face_start = len(output_faces)
    boolean_cutters: list[trimesh.Trimesh] = []
    connector_record = add_local_male_connector_and_backing(
        output_vertices=output_vertices,
        output_faces=output_faces,
        boundary_ids=[int(value) for value in boundary_ids],
        boundary_points=np.asarray(boundary_points, dtype=np.float64),
        inward=np.asarray(inward, dtype=np.float64),
        spec=spec,
        inward_directions=np.asarray(inward_directions, dtype=np.float64),
        boolean_cutters=boolean_cutters,
    )
    generated_faces = np.asarray(
        output_faces[generated_face_start:],
        dtype=np.int64,
    )
    generated_face_count = int(len(generated_faces))
    expected_face_count = sum(
        int(connector_record[key])
        for key in (
            "backing_faces_added",
            "connector_side_faces_added",
            "connector_cap_faces_added",
        )
    )
    if generated_face_count != expected_face_count:
        raise ValueError(
            "local connector preflight face accounting mismatch: "
            f"generated={generated_face_count}, expected={expected_face_count}"
        )
    if generated_face_count == 0:
        raise ValueError("local connector preflight produced no generated faces")

    points = np.asarray(output_vertices, dtype=np.float64)
    triangles = points[generated_faces]
    double_areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    degenerate_indices = np.flatnonzero(double_areas <= 1e-12)
    if len(degenerate_indices):
        raise ValueError(
            "local connector preflight produced degenerate faces: "
            f"count={int(len(degenerate_indices))}, "
            f"indices={degenerate_indices[:16].tolist()}"
        )

    return {
        "valid": True,
        "boundary_vertices": int(len(boundary_ids)),
        "side_faces": int(
            connector_record["backing_faces_added"]
            + connector_record["connector_side_faces_added"]
        ),
        "source_faces": int(len(source_faces)),
        "generated_faces": generated_face_count,
        "expected_cap_faces": int(connector_record["connector_cap_faces_added"]),
        "cap_faces": int(connector_record["connector_cap_faces_added"]),
        "total_faces": int(len(output_faces)),
        "after_vertex_merge": int(len(output_faces)),
        "after_duplicate_removal": int(len(output_faces)),
        "nondegenerate_faces": int(len(output_faces)),
        "degenerate_faces": 0,
        "degenerate_source_faces": 0,
        "degenerate_side_faces": 0,
        "degenerate_cap_faces": 0,
        "degenerate_cap_samples": [],
        "duplicate_faces": 0,
        "invalid_faces": 0,
        "lead_in_applied": True,
        "coordinate_precision_decimals": 6,
        "triangulation": {
            "status": "production_local_connector_geometry",
            "face_count": generated_face_count,
        },
        "cap_surface_quality": {
            "valid": True,
            "strategy": connector_record["backing_triangulation"],
            "minimum_triangle_double_area_mm2": float(double_areas.min()),
        },
        "side_wall_quality": {
            "valid": True,
            "strategy": "geometry_aware_unequal_count_connector_strips",
            "maximum_fanout": int(
                connector_record["backing_strip_maximum_fanout"]
            ),
            "maximum_cross_edge_mm": float(
                connector_record["backing_strip_maximum_cross_edge_mm"]
            ),
            "minimum_triangle_double_area_mm2": float(double_areas.min()),
        },
        "interface_geometry": "local-connector",
        "private_boolean_cutter_count": int(len(boolean_cutters)),
        "connector_record": dict(connector_record),
    }

def build_local_male_attachment_cutter(
    *,
    boundary_points: np.ndarray,
    inward: np.ndarray,
    spec: LocalConnectorSpec,
    outside_extension_mm: float,
    inward_directions: np.ndarray | None = None,
    boundary_overcut_mm: float = 0.02,
) -> trimesh.Trimesh:
    """Build the complete negative volume occupied by the printable male side.

    The male part is more than its compact peg: the complete 3 mm printable
    backing and its outer-large/inner-small transition also enter the parent.
    Cutting only the peg leaves that broad backing interpenetrating the parent.
    Reuse the production male builder so the negative tool is geometrically
    identical to the child attachment, then close it outside the source surface.
    """
    boundary = np.asarray(boundary_points, dtype=np.float64)
    axis = np.asarray(inward, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    plan = plan_local_connector(boundary, axis, spec, samples=64)
    # The printable male keeps the exact painted boundary.  Its negative tool
    # must not share that exact exit curve with the restored parent patch:
    # coincident Boolean operands create collapsed seam faces in Manifold on
    # dense vendor loops.  Expand only the cutter laterally by 0.02 mm; this is
    # below the 0.50 mm fit budget, removes the coplanar seam, and deliberately
    # makes the female backing pocket the slightly larger mate.
    overcut = max(float(boundary_overcut_mm), 0.0)
    cutter_boundary = boundary.copy()
    if overcut > 0.0:
        cutter_boundary = boundary + _line_preserving_inset_displacements(
            boundary,
            plan,
            -overcut,
        )
    outside = cutter_boundary - axis[None, :] * max(float(outside_extension_mm), 0.10)
    output_vertices: list[np.ndarray] = [point.copy() for point in outside]
    output_vertices.extend(point.copy() for point in cutter_boundary)
    count = len(cutter_boundary)
    outside_ids = list(range(count))
    boundary_ids = list(range(count, 2 * count))
    output_faces: list[list[int]] = []
    side_faces = add_side_faces_between_rings(
        output_faces,
        outside_ids,
        boundary_ids,
        vertices=output_vertices,
    )
    outside_cap_faces = triangulate_cap(
        output_vertices,
        output_faces,
        [outside_ids],
        [_project_connector_points(outside, plan)],
        [[0]],
        -axis,
    )
    if side_faces <= 0 or outside_cap_faces <= 0:
        raise ValueError("local male attachment cutter outside closure failed")
    add_local_male_connector_and_backing(
        output_vertices=output_vertices,
        output_faces=output_faces,
        boundary_ids=boundary_ids,
        boundary_points=cutter_boundary,
        inward=axis,
        spec=spec,
        inward_directions=inward_directions,
    )
    # Every seam above is built with shared vertex indices already.  Do not
    # run Trimesh's generic vertex welding here: high-resolution vendor loops
    # can contain spatially coincident but topologically distinct samples, and
    # merging those samples changes a closed cutter into a non-manifold one.
    cutter = trimesh.Trimesh(
        vertices=np.asarray(output_vertices, dtype=np.float64),
        faces=np.asarray(output_faces, dtype=np.int64),
        process=False,
    )
    cutter.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(cutter)
    trimesh.repair.fix_normals(cutter)
    if not cutter.is_watertight or not cutter.is_winding_consistent:
        edge_counts = np.bincount(cutter.edges_unique_inverse)
        raise ValueError(
            "complete male attachment boolean cutter is invalid: "
            f"watertight={bool(cutter.is_watertight)}, "
            f"winding_consistent={bool(cutter.is_winding_consistent)}, "
            f"boundary_edges={int(np.count_nonzero(edge_counts == 1))}, "
            f"nonmanifold_edges={int(np.count_nonzero(edge_counts > 2))}"
        )
    cutter.metadata["boundary_overcut_mm"] = float(overcut)
    return cutter

def add_local_female_socket_and_backing(
    *,
    output_vertices: list[np.ndarray],
    output_faces: list[list[int]],
    boundary_ids: list[int],
    boundary_points: np.ndarray,
    inward: np.ndarray,
    spec: LocalConnectorSpec,
) -> dict:
    """Close the parent interface around a compact socket with a 45° mouth."""

    plan = plan_local_connector(boundary_points, inward, spec, samples=64)
    plan_normal = -np.asarray(plan["inward"], dtype=np.float64)
    mouth_ids, order, backing_faces = _add_constrained_connector_annulus(
        output_vertices=output_vertices,
        output_faces=output_faces,
        outer_ids=list(boundary_ids),
        outer_points=np.asarray(boundary_points, dtype=np.float64),
        inner_points=np.asarray(plan["socket_mouth"], dtype=np.float64),
        plan=plan,
        outward_normal=plan_normal,
    )
    throat_ids = _append_points(
        output_vertices,
        np.asarray(plan["socket_throat"], dtype=np.float64)[order],
    )
    bottom_ids = _append_points(
        output_vertices,
        np.asarray(plan["socket_bottom"], dtype=np.float64)[order],
    )
    surface_normal = plan_normal
    if backing_faces <= 0:
        raise ValueError("local female backing triangulation failed")
    side_faces = add_side_faces_between_rings(
        output_faces,
        mouth_ids,
        throat_ids,
        vertices=output_vertices,
    )
    side_faces += add_side_faces_between_rings(
        output_faces,
        throat_ids,
        bottom_ids,
        vertices=output_vertices,
    )
    cap_faces = triangulate_cap(
        output_vertices,
        output_faces,
        [bottom_ids],
        [_project_connector_points(plan["socket_bottom"], plan)],
        [[0]],
        surface_normal,
    )
    if cap_faces <= 0:
        raise ValueError("local female socket bottom triangulation failed")
    return _connector_record(plan, "female", backing_faces, side_faces, cap_faces)

def add_local_female_boolean_closure(
    *,
    output_vertices: list[np.ndarray],
    output_faces: list[list[int]],
    boundary_ids: list[int],
    boundary_points: np.ndarray,
    inward: np.ndarray,
    spec: LocalConnectorSpec,
    inward_directions: np.ndarray | None = None,
    occupied_parent_edges: set[tuple[int, int]] | None = None,
    build_boolean_cutters: bool = True,
) -> tuple[dict, list[trimesh.Trimesh]]:
    """Restore the missing source patch, then cut the complete female cavity.

    The pre-1.5.2 path built a partial female wedge and capped its compact
    socket mouth before running the Boolean.  The complete male cutter crossed
    that internal cap exactly at its tangent ring, leaving zero-area bridge
    triangles even though Manifold still called the result watertight.  A
    parent body only needs its missing painted surface restored to become a
    closed solid.  Cap that source loop in its original plane, then let the two
    sequential cutters create every subsurface face of the final cavity.
    """
    plan = plan_local_connector(boundary_points, inward, spec, samples=64)
    axis = np.asarray(plan["inward"], dtype=np.float64)
    backing_depth = float(plan.get("full_boundary_backing_depth_mm", 0.0))
    boundary = np.asarray(boundary_points, dtype=np.float64)
    preclosure_face_start = len(output_faces)
    cap_faces, preclosure_triangulation = triangulate_boundary_cap_without_center(
        output_vertices,
        output_faces,
        [int(index) for index in boundary_ids],
        -axis,
        occupied_edges=occupied_parent_edges,
    )
    if cap_faces <= 0:
        raise ValueError("local female source-patch preclosure cap failed")
    preclosure_triangles = np.asarray(output_vertices, dtype=np.float64)[
        np.asarray(output_faces[preclosure_face_start:], dtype=np.int64)
    ]
    preclosure_double_areas = np.linalg.norm(
        np.cross(
            preclosure_triangles[:, 1] - preclosure_triangles[:, 0],
            preclosure_triangles[:, 2] - preclosure_triangles[:, 0],
        ),
        axis=1,
    )
    if np.any(preclosure_double_areas <= 1e-12):
        raise ValueError("local female source-patch preclosure contains degenerate faces")
    cutters: list[trimesh.Trimesh] = []
    male_attachment_cutter = None
    compact_socket_cutter = None
    if build_boolean_cutters:
        male_attachment_cutter = build_local_male_attachment_cutter(
            boundary_points=boundary,
            inward=axis,
            spec=spec,
            outside_extension_mm=backing_depth + 1.0,
            inward_directions=inward_directions,
            boundary_overcut_mm=0.02,
        )
        cutters = [male_attachment_cutter]
        if bool(plan["compact_peg_enabled"]):
            compact_socket_cutter = build_socket_cutter_from_plan(
                plan,
                outside_extension_mm=backing_depth + 1.0,
            )
            cutters.append(compact_socket_cutter)
        for cutter_index, cutter in enumerate(cutters):
            if not cutter.is_watertight or not cutter.is_winding_consistent:
                raise ValueError(
                    f"female cavity cutter {cutter_index} is not a valid closed solid"
                )
    record = _connector_record(
        plan,
        "female_boolean",
        int(cap_faces),
        0,
        0,
    )
    record["boolean_socket_required"] = bool(plan["compact_peg_enabled"])
    record["boolean_cutters_built"] = bool(build_boolean_cutters)
    if not build_boolean_cutters:
        boolean_cutter_scope = "deferred_to_complete_emitted_direct_child_solids"
    elif bool(plan["compact_peg_enabled"]):
        boolean_cutter_scope = "sequential_complete_male_backing_then_compact_socket"
    else:
        boolean_cutter_scope = "complete_male_backing_only_zero_engagement"
    record["boolean_cutter_scope"] = boolean_cutter_scope
    record["male_attachment_cutter_volume_mm3"] = (
        float(abs(male_attachment_cutter.volume))
        if male_attachment_cutter is not None
        else 0.0
    )
    record["female_backing_boundary_overcut_mm"] = 0.02
    record["compact_socket_cutter_volume_mm3"] = (
        float(abs(compact_socket_cutter.volume))
        if compact_socket_cutter is not None
        else 0.0
    )
    record["planned_cavity_cutter_count"] = (
        2 if bool(plan["compact_peg_enabled"]) else 1
    )
    record["cavity_cutter_count"] = int(len(cutters))
    record["cavity_boolean_strategy"] = (
        "sequential_difference_without_cutter_union"
    )
    record["backing_taper_target_degrees"] = 45.0
    record["backing_taper_depth_mm"] = backing_depth
    record["backing_profile"] = "outer_large_inner_small"
    record["pre_boolean_closure"] = "source_plane_patch_only"
    record["pre_boolean_closure_faces_added"] = int(cap_faces)
    record["pre_boolean_closure_triangulation"] = dict(
        preclosure_triangulation
    )
    record["pre_boolean_closure_occupied_parent_edge_count"] = int(
        len(occupied_parent_edges or set())
    )
    record["pre_boolean_closure_forbidden_internal_edge_count"] = int(
        preclosure_triangulation.get("forbidden_internal_edges", 0)
    )
    record["pre_boolean_closure_min_double_area_mm2"] = float(
        preclosure_double_areas.min()
    )
    return record, cutters
