from __future__ import annotations

from scipy.ndimage import convolve1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .cap_template import (
    fit_affine_cap_inside_parent,
    progressive_boundary_deformation,
    refined_harmonic_heightfield_cap,
)

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


def visible_top_edge_clearance(insert_shrink_mm: float) -> float:
    """Keep the exterior cut ring coincident; start fit clearance below it.

    ``insert_shrink_mm`` remains part of the signature so every caller routes
    the requested internal fit through this explicit visible-surface policy.
    The lead-in ring still transitions to the full shrink below the source
    surface, while the original/fared top ring is never radially offset.
    """
    _ = max(float(insert_shrink_mm), 0.0)
    return 0.0
from .interface_retopology import (
    InterfaceRetopologyService,
    generated_geometry_boundary_vertices,
)
from .domain import PlanarArcRetopologyContext, CapDecision
from .hidden_interface import (
    HIDDEN_INTERFACE_MINIMUM_LOAD_BEARING_DEPTH_MM,
    HiddenInterfacePlanner,
)
from .guided_internal_cut import (
    GuidedInternalCutPlanner,
    GuidedInternalCutSpec,
    adaptive_guided_entry_ring,
)
from .local_connectors import (
    LocalConnectorSpec,
    build_socket_cutter_from_plan,
    plan_local_connector,
    subtract_socket_cutters,
)
from .connector_planning import (
    local_connector_safe_depth_at_footprint,
    local_connector_safe_depth_from_field,
    local_connector_spec_for_interface,
)
from .connector_geometry import (
    backing_taper_angle_audit as _backing_taper_angle_audit,
    line_preserving_inset_displacements as _line_preserving_inset_displacements,
    printable_backing_profile as _printable_backing_profile,
    printable_backing_rings as _printable_backing_rings,
    project_connector_points as _project_connector_points,
    user_reviewed_shallow_minimal_needle_advisory_is_eligible,
)
from .connector_topology import (
    orient_face_patch_consistently,
    triangulate_bounded_ring_strip,
    triangulate_connector_annulus,
)
from .connector_surface import refine_connector_annulus_heightfield
from .mesh_finalization import (
    finalize_source_preserving_mesh,
)
from .reporting import runtime_log


def recursive_face_geometry_key(
    vertex_array: np.ndarray,
    face: np.ndarray,
) -> tuple:
    points = np.round(
        vertex_array[np.asarray(face, dtype=np.int64)],
        decimals=6,
    )
    return tuple(
        sorted(tuple(float(value) for value in point) for point in points)
    )


def finalize_recursive_colored_mesh(
    mesh: trimesh.Trimesh,
    face_color_codes: list[str],
    face_filament_slots: list[int | None],
    default_color_code: str,
    protected_source_face_count: int = 0,
) -> trimesh.Trimesh:
    """Finalize geometry while preserving each surviving triangle's material."""
    if len(face_color_codes) != len(mesh.faces):
        raise ValueError("recursive face-color count does not match mesh faces")
    if len(face_filament_slots) != len(mesh.faces):
        raise ValueError("recursive face-slot count does not match mesh faces")
    vertex_array = np.asarray(mesh.vertices, dtype=np.float64)
    color_votes_by_geometry: dict[tuple, collections.Counter] = (
        collections.defaultdict(collections.Counter)
    )
    for face, code, slot in zip(
        np.asarray(mesh.faces),
        face_color_codes,
        face_filament_slots,
    ):
        color_votes_by_geometry[
            recursive_face_geometry_key(vertex_array, face)
        ][(str(code), None if slot is None else int(slot))] += 1

    def select_material(votes: collections.Counter) -> tuple[str, int | None]:
        return min(
            votes,
            key=lambda material: (
                -int(votes[material]),
                str(material[0]),
                -1 if material[1] is None else int(material[1]),
            ),
        )
    mesh.visual.face_colors = np.asarray(
        [
            COLOR_INFO.get(code, {"rgba": [200, 200, 200, 255]})["rgba"]
            for code in face_color_codes
        ],
        dtype=np.uint8,
    )
    # Painted vendor meshes may contain coordinate-coincident vertices whose
    # identities keep distinct source triangles and material boundaries
    # intact.  Recursive parts must preserve those source faces just like the
    # large root partition does; welding all vertices first can collapse valid
    # vendor triangles and turn a closed insert into an open one.
    # Remove only exact topological duplicate faces (same vertex ids).  Do not
    # conflate coordinate-coincident faces backed by distinct vendor ids.
    protected_count = min(
        max(int(protected_source_face_count), 0),
        int(len(mesh.faces)),
    )
    recursive_faces = np.asarray(mesh.faces, dtype=np.int64)
    keep_mask = np.ones(len(recursive_faces), dtype=bool)
    seen_topology: set[tuple[int, int, int]] = set()
    for face_index, face in enumerate(recursive_faces):
        key = tuple(sorted(int(value) for value in face))
        if face_index < protected_count:
            seen_topology.add(key)
            continue
        if key in seen_topology:
            keep_mask[int(face_index)] = False
        else:
            seen_topology.add(key)
    if not np.all(keep_mask):
        mesh.update_faces(keep_mask)

    # Protect every original vendor triangle from area-based cleanup.  Only
    # generated wall/cap faces are eligible for removal.
    nondegenerate_mask = np.asarray(mesh.nondegenerate_faces(), dtype=bool)
    nondegenerate_mask[:protected_count] = True
    if not np.all(nondegenerate_mask):
        mesh.update_faces(nondegenerate_mask)
    generated_orientation = {"applied": False, "changed_generated_faces": 0}
    if protected_count < len(mesh.faces):
        source_faces = np.asarray(mesh.faces[:protected_count], dtype=np.int64).copy()
        generated_before = np.asarray(mesh.faces[protected_count:], dtype=np.int64).copy()
        orient_mesh_faces_consistently(mesh)
        mesh.faces[:protected_count] = source_faces
        generated_after = np.asarray(mesh.faces[protected_count:], dtype=np.int64)
        generated_orientation = {
            "applied": True,
            "changed_generated_faces": int(
                np.count_nonzero(np.any(generated_before != generated_after, axis=1))
            ),
            "source_faces_changed": 0,
        }
    # Do not weld, close, orient, or remove disconnected source shells here.
    # Source topology is diagnostic; interface builders validate their own new
    # faces before this assembly step.
    mesh.metadata["source_topology_diagnostics"] = validate_mesh_in_memory(mesh)
    mesh.metadata["protected_source_face_count"] = protected_count
    mesh.metadata["source_geometry_mutation"] = "none"
    mesh.metadata["generated_interface_orientation"] = generated_orientation
    runtime_log(
        "mesh-finalize",
        "recursive_mesh_finalize_face_counts",
        "Recursive mesh cleanup face-count stages completed",
        part_id=str(mesh.metadata.get("name", "")),
        **dict(mesh.metadata.get("finalize_face_counts", {})),
    )
    final_vertex_array = np.asarray(mesh.vertices, dtype=np.float64)
    default_slot = COLOR_INFO.get(default_color_code, {}).get("filament_slot")
    final_faces = np.asarray(mesh.faces, dtype=np.int64)
    final_materials: list[tuple[str, int | None] | None] = []
    for face in final_faces:
        votes = color_votes_by_geometry.get(
            recursive_face_geometry_key(final_vertex_array, face)
        )
        final_materials.append(select_material(votes) if votes else None)

    # ``finalize_mesh`` may close a small residual boundary after generated
    # walls/caps are assembled.  Those new triangles have no pre-finalization
    # geometry key.  Propagate material across shared edges so a closure inside
    # a child region inherits that child instead of becoming a wrapper-colored
    # island which is cut again at the next recursive step.
    unresolved = {
        int(index)
        for index, material in enumerate(final_materials)
        if material is None
    }
    edge_faces: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for face_index, face in enumerate(final_faces):
        for left, right in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge_faces[tuple(sorted((int(left), int(right))))].append(
                int(face_index)
            )
    while unresolved:
        updates: dict[int, tuple[str, int | None]] = {}
        for face_index in sorted(unresolved):
            votes: collections.Counter = collections.Counter()
            face = final_faces[int(face_index)]
            for left, right in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            ):
                for neighbor in edge_faces.get(
                    tuple(sorted((int(left), int(right)))),
                    [],
                ):
                    material = final_materials[int(neighbor)]
                    if int(neighbor) != int(face_index) and material is not None:
                        votes[material] += 1
            if votes:
                updates[int(face_index)] = select_material(votes)
        if not updates:
            break
        for face_index, material in updates.items():
            final_materials[int(face_index)] = material
            unresolved.discard(int(face_index))

    fallback_material = (str(default_color_code), default_slot)
    inferred_face_count = int(
        sum(
            1
            for face in final_faces
            if recursive_face_geometry_key(final_vertex_array, face)
            not in color_votes_by_geometry
        )
    )
    final_materials = [
        material if material is not None else fallback_material
        for material in final_materials
    ]
    final_codes = [str(material[0]) for material in final_materials]
    final_slots = [
        None if material[1] is None else int(material[1])
        for material in final_materials
    ]
    mesh.visual.face_colors = np.asarray(
        [
            COLOR_INFO.get(code, {"rgba": [200, 200, 200, 255]})["rgba"]
            for code in final_codes
        ],
        dtype=np.uint8,
    )
    mesh.metadata["face_color_codes"] = final_codes
    mesh.metadata["face_filament_slot_indices"] = final_slots
    mesh.metadata["recursive_finalize_inferred_face_materials"] = int(
        inferred_face_count
    )
    mesh.metadata["recursive_finalize_defaulted_face_materials"] = int(
        len(unresolved)
    )
    return mesh


def component_inward_direction(
    vertices: np.ndarray,
    faces: np.ndarray,
    component: Component,
    model_center: np.ndarray,
) -> np.ndarray:
    local_vertices, local_faces, _, _ = build_local_mesh(vertices, faces, component)
    component_center = local_vertices[local_faces.reshape(-1)].mean(axis=0)
    return -average_outward_normal(local_vertices, local_faces, component_center, model_center)


def mesh_vertex_inward_normals(local_vertices: np.ndarray, local_faces: np.ndarray) -> np.ndarray:
    """Return area-weighted local inward normals for every source vertex."""
    triangles = local_vertices[local_faces]
    face_cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    accumulated = np.zeros_like(local_vertices, dtype=np.float64)
    for column in range(3):
        np.add.at(accumulated, local_faces[:, column], face_cross)
    lengths = np.linalg.norm(accumulated, axis=1)
    valid = lengths > 1e-12
    accumulated[valid] /= lengths[valid, None]
    accumulated[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return -accumulated


def mesh_vertex_conormal_evidence(
    local_vertices: np.ndarray,
    local_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate face-normal and interior votes once for every mesh vertex."""
    vertices = np.asarray(local_vertices, dtype=np.float64)
    faces = np.asarray(local_faces, dtype=np.int64)
    triangles = vertices[faces]
    face_cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    face_weights = np.maximum(np.linalg.norm(face_cross, axis=1), 1e-12)
    face_centroids = triangles.mean(axis=1)
    accumulated_normal = np.zeros_like(vertices, dtype=np.float64)
    accumulated_interior = np.zeros_like(vertices, dtype=np.float64)
    accumulated_weight = np.zeros(len(vertices), dtype=np.float64)
    for column in range(3):
        vertex_ids = faces[:, column]
        np.add.at(accumulated_normal, vertex_ids, face_cross)
        np.add.at(
            accumulated_interior,
            vertex_ids,
            (face_centroids - vertices[vertex_ids]) * face_weights[:, None],
        )
        np.add.at(accumulated_weight, vertex_ids, face_weights)
    return accumulated_normal, accumulated_interior, accumulated_weight


def boundary_loop_interior_conormals(
    local_vertices: np.ndarray,
    local_faces: np.ndarray,
    loop: list[int],
    reference_axis: np.ndarray | None = None,
    vertex_evidence: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Return surface-tangent directions from a cut boundary into its owner.

    The direction is inferred from the actual incident triangles, so concave
    loops and hole boundaries do not depend on a loop centroid or projected
    polygon winding.
    """
    vertices = np.asarray(local_vertices, dtype=np.float64)
    faces = np.asarray(local_faces, dtype=np.int64)
    loop_ids = np.asarray(loop, dtype=np.int64)
    if not len(loop_ids):
        return np.empty((0, 3), dtype=np.float64)
    if vertex_evidence is None:
        vertex_evidence = mesh_vertex_conormal_evidence(vertices, faces)
    accumulated_normal, accumulated_interior, accumulated_weight = vertex_evidence
    if (
        accumulated_normal.shape != vertices.shape
        or accumulated_interior.shape != vertices.shape
        or accumulated_weight.shape != (len(vertices),)
    ):
        raise ValueError("vertex conormal evidence does not match local mesh")

    points = vertices[loop_ids]
    normals = accumulated_normal[loop_ids]
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    interior = accumulated_interior[loop_ids] / np.maximum(
        accumulated_weight[loop_ids, None], 1e-12
    )
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangents -= normals * np.einsum("ij,ij->i", tangents, normals)[:, None]
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1)[:, None], 1e-12)
    conormals = interior.copy()
    conormals -= normals * np.einsum("ij,ij->i", conormals, normals)[:, None]
    conormals -= tangents * np.einsum("ij,ij->i", conormals, tangents)[:, None]
    conormal_lengths = np.linalg.norm(conormals, axis=1)
    invalid = conormal_lengths <= 1e-10
    if np.any(invalid):
        fallback = np.cross(normals[invalid], tangents[invalid])
        reverse = np.einsum("ij,ij->i", fallback, interior[invalid]) < 0.0
        fallback[reverse] *= -1.0
        conormals[invalid] = fallback
    conormals /= np.maximum(np.linalg.norm(conormals, axis=1)[:, None], 1e-12)
    # A reloaded recursive subassembly can expose a seam vertex whose incident
    # face-centroid vectors cancel exactly (notably where several body loops
    # meet).  That makes the local inside vote zero, not evidence that the
    # conormal points outside.  Correct genuinely negative votes pointwise and
    # let the ordered-loop field below resolve ambiguous zeros continuously.
    agreement = np.einsum("ij,ij->i", conormals, interior)
    reverse = agreement < -1e-12
    conormals[reverse] *= -1.0
    return smooth_ordered_loop_interior_conormals(
        points,
        conormals,
        reference_axis=reference_axis,
    )


def smooth_ordered_loop_interior_conormals(
    loop_points: np.ndarray,
    raw_interior_conormals: np.ndarray,
    tangent_radius_mm: float = 1.20,
    reference_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Build a stable in-plane inward field from the closed curve itself.

    Per-triangle surface conormals are useful for deciding which side is inside,
    but directly multiplying them by a multi-millimetre taper turns tiny normal
    noise into long spikes.  Use a best-fit loop plane and millimetre-scale chord
    tangents for geometry; use the raw field only to choose the global inward
    orientation.  One sign is used for the entire ordered loop, so it cannot
    alternate from vertex to vertex.
    """
    points = np.asarray(loop_points, dtype=np.float64)
    raw = np.asarray(raw_interior_conormals, dtype=np.float64)
    if points.shape != raw.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("ordered loop points and raw conormals must be matching Nx3 arrays")
    if len(points) < 3:
        return raw.copy()

    centered = points - points.mean(axis=0)
    _u_singular, _singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if reference_axis is None:
        plane_normal = vh[-1]
    else:
        plane_normal = np.asarray(reference_axis, dtype=np.float64).copy()
        if plane_normal.shape != (3,) or not np.all(np.isfinite(plane_normal)):
            raise ValueError("loop conormal reference axis must be one finite 3-vector")
        if float(np.linalg.norm(plane_normal)) <= 1e-12:
            raise ValueError("loop conormal reference axis must be non-zero")
    raw_axis = np.cross(
        np.roll(points, -1, axis=0) - points,
        raw,
    ).mean(axis=0)
    if float(np.dot(plane_normal, raw_axis)) < 0.0:
        plane_normal *= -1.0
    plane_normal /= max(float(np.linalg.norm(plane_normal)), 1e-12)

    edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    target = max(float(tangent_radius_mm), float(np.median(edge_lengths)) * 2.0)
    tangents = np.empty_like(points)
    count = len(points)
    for index in range(count):
        previous = index
        travelled = 0.0
        while travelled < target and previous - 1 != index - count:
            edge_index = (previous - 1) % count
            travelled += float(edge_lengths[edge_index])
            previous -= 1
        following = index
        travelled = 0.0
        while travelled < target and following + 1 != index + count:
            edge_index = following % count
            travelled += float(edge_lengths[edge_index])
            following += 1
        tangents[index] = points[following % count] - points[previous % count]

    tangents -= plane_normal * (tangents @ plane_normal)[:, None]
    tangent_lengths = np.linalg.norm(tangents, axis=1)
    invalid = tangent_lengths <= 1e-12
    if np.any(invalid):
        fallback = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
        fallback -= plane_normal * (fallback @ plane_normal)[:, None]
        tangents[invalid] = fallback[invalid]
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1)[:, None], 1e-12)

    smooth = np.cross(plane_normal, tangents)
    if float(np.sum(np.einsum("ij,ij->i", smooth, raw))) < 0.0:
        smooth *= -1.0
    smooth /= np.maximum(np.linalg.norm(smooth, axis=1)[:, None], 1e-12)
    # A strongly concave corner can legitimately disagree with the wide chord.
    # Keep the incident-face interior direction only at those exceptional
    # vertices; ordinary triangle-scale wobble remains aligned with the stable
    # curve field and therefore cannot grow into long taper spikes.
    agreement = np.einsum("ij,ij->i", smooth, raw)
    concave = agreement <= 0.05
    if np.any(concave):
        fallback = raw[concave].copy()
        fallback -= plane_normal * (fallback @ plane_normal)[:, None]
        fallback_lengths = np.linalg.norm(fallback, axis=1)
        valid = fallback_lengths > 1e-12
        fallback[valid] /= fallback_lengths[valid, None]
        smooth_indices = np.flatnonzero(concave)
        smooth[smooth_indices[valid]] = fallback[valid]
    return smooth


def offset_points_along_conormals(
    points: np.ndarray,
    interior_conormals: np.ndarray,
    inward_offset_mm: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    conormals = np.asarray(interior_conormals, dtype=np.float64)
    if points.shape != conormals.shape:
        raise ValueError("boundary points and interior conormals must match")
    return points + conormals * float(inward_offset_mm)


def smooth_closed_inward_direction_field(
    directions: np.ndarray,
    smoothing_passes: int = 24,
) -> tuple[np.ndarray, dict]:
    """Remove triangle-scale normal noise from one ordered closed loop.

    If every safety-corrected direction lies in the same hemisphere, a single
    insertion axis is both smoother and mechanically meaningful.  Curved
    interfaces that genuinely need a changing direction retain that variation,
    but only after a strong cyclic low-pass filter removes tooth-scale peaks.
    """
    field = np.asarray(directions, dtype=np.float64)
    if field.ndim != 2 or field.shape[1] != 3 or not len(field):
        raise ValueError("closed inward direction field must be a non-empty Nx3 array")
    lengths = np.linalg.norm(field, axis=1)
    if np.any(lengths <= 1e-12) or not np.all(np.isfinite(lengths)):
        raise ValueError("closed inward direction field contains invalid vectors")
    field = field / lengths[:, None]

    mean = field.mean(axis=0)
    resultant = float(np.linalg.norm(mean))
    axis = field[0].copy() if resultant <= 1e-12 else mean / resultant
    minimum_axis_dot = float(np.min(field @ axis))
    if resultant > 1e-12 and minimum_axis_dot >= 0.05:
        smooth = np.tile(axis, (len(field), 1))
        mode = "coherent_loop_axis"
        passes = 0
    else:
        smooth = field.copy()
        passes = max(int(smoothing_passes), 1)
        for _ in range(passes):
            smooth = (
                np.roll(smooth, 2, axis=0)
                + 4.0 * np.roll(smooth, 1, axis=0)
                + 6.0 * smooth
                + 4.0 * np.roll(smooth, -1, axis=0)
                + np.roll(smooth, -2, axis=0)
            ) / 16.0
            smooth_lengths = np.linalg.norm(smooth, axis=1)
            invalid = smooth_lengths <= 1e-12
            smooth[invalid] = field[invalid]
            smooth /= np.maximum(np.linalg.norm(smooth, axis=1)[:, None], 1e-12)
        mode = "strong_cyclic_low_pass"

    adjacent_dots = np.clip(
        np.einsum("ij,ij->i", smooth, np.roll(smooth, -1, axis=0)),
        -1.0,
        1.0,
    )
    return smooth, {
        "direction_mode": mode,
        "direction_mean_resultant": resultant,
        "direction_minimum_axis_dot": minimum_axis_dot,
        "direction_smoothing_passes": passes,
        "direction_adjacent_angle_max_degrees": float(
            np.degrees(np.arccos(adjacent_dots)).max()
        ),
    }


def tapered_profile_inset_limit(
    visible_top_points: np.ndarray,
    fit_clearance_mm: float,
    maximum_taper_depth_mm: float,
    maximum_total_depth_mm: float,
    target_slope_degrees: float = DEFAULT_LEAD_IN_SLOPE_DEGREES,
) -> tuple[float, dict]:
    """Return a visible but collapse-safe lateral inset for one closed loop."""
    visible = np.asarray(visible_top_points, dtype=np.float64)
    if visible.ndim != 2 or visible.shape[1] != 3 or len(visible) < 3:
        raise ValueError("taper profile requires at least three ordered loop points")
    centered = visible - visible.mean(axis=0)
    _u_singular, _singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    plane_normal = vh[-1]
    u, v = orthonormal_basis(plane_normal)
    projected = np.column_stack((centered @ u, centered @ v))
    projected_next = np.roll(projected, -1, axis=0)
    projected_area = 0.5 * abs(
        float(
            np.sum(
                projected[:, 0] * projected_next[:, 1]
                - projected_next[:, 0] * projected[:, 1]
            )
        )
    )
    perimeter = float(
        np.linalg.norm(np.roll(visible, -1, axis=0) - visible, axis=1).sum()
    )
    hydraulic_radius = (
        2.0 * projected_area / perimeter if perimeter > 1e-12 else 0.0
    )
    slope_degrees = float(np.clip(float(target_slope_degrees), 1.0, 89.0))
    slope_tangent = float(np.tan(np.radians(slope_degrees)))
    clearance = max(float(fit_clearance_mm), 0.0)
    lateral_inset = min(
        max(clearance, 0.55 * hydraulic_radius),
        max(float(maximum_taper_depth_mm), 0.0) / max(slope_tangent, 1e-12),
        max(float(maximum_total_depth_mm), 0.0) / max(slope_tangent, 1e-12),
    )
    lateral_inset = max(lateral_inset, min(clearance, 0.55 * hydraulic_radius))
    return lateral_inset, {
        "projected_area_mm2": projected_area,
        "loop_perimeter_mm": perimeter,
        "hydraulic_radius_mm": hydraulic_radius,
        "lateral_inset_mm": lateral_inset,
        "target_slope_degrees": slope_degrees,
    }


def tapered_profile_total_depth_budget(
    fixed_depth_mm: float,
    planar_extra_limit_mm: float | None,
) -> float:
    """Use the full allowed inward travel when sizing the 45-degree shoulder.

    ``fixed_depth_mm`` is the preferred baseline travel, while planar travel may
    legally extend the same interface farther.  Treating only the baseline as
    the profile budget silently capped a requested 5 mm taper at 0.4 mm, making
    the generated wall look cylindrical even though the final cap reached 5 mm.
    """
    return max(float(fixed_depth_mm), 0.0) + max(
        float(planar_extra_limit_mm or 0.0), 0.0
    )


def tapered_profile_inset_candidates(
    fit_clearance_mm: float,
    preferred_profile_inset_mm: float,
) -> list[dict]:
    """Back off a visual taper without abandoning insertion depth.

    Curved or tiny interfaces can intersect the parent when the hydraulic-radius
    shoulder is applied at full size.  Evaluate decreasing shoulder fractions,
    then retain the legacy thin-boundary inward-search candidates as a final
    compatibility fallback.
    """
    clearance = max(float(fit_clearance_mm), 0.0)
    preferred = max(float(preferred_profile_inset_mm), clearance)
    candidates: list[dict] = []
    seen: set[float] = set()

    def append(value: float, profile_factor: float | None, source: str) -> None:
        rounded = round(max(float(value), clearance), 9)
        if rounded in seen:
            return
        seen.add(rounded)
        candidates.append(
            {
                "effective_insert_shrink_mm": rounded,
                "profile_factor": profile_factor,
                "candidate_source": source,
            }
        )

    for factor in (1.0, 0.75, 0.50, 0.25, 0.0):
        append(
            clearance + (preferred - clearance) * factor,
            factor,
            "taper_profile_backoff",
        )
    for additional in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60):
        append(clearance + additional, None, "thin_boundary_inset_fallback")
    # ``build_reserved_cap_decision`` stops at the first candidate which keeps
    # the preferred insertion depth.  That is correct only when the complete
    # candidate stream is ordered from the largest shoulder to the smallest.
    # The legacy fallback values above may be larger than the profile-derived
    # values, so appending them after the profile backoff silently violated that
    # contract and could both select the wrong shoulder and waste every later
    # thickness probe on thin interfaces.
    candidates.sort(
        key=lambda candidate: float(candidate["effective_insert_shrink_mm"]),
        reverse=True,
    )
    return candidates


def select_taper_profile_candidate(
    candidates: list[dict],
    preferred_minimum_depth_mm: float,
) -> tuple[dict, str]:
    """Choose the largest taper that preserves useful insertion depth."""
    if not candidates:
        raise ValueError("at least one valid taper profile candidate is required")
    preferred_depth = max(float(preferred_minimum_depth_mm), 0.0)
    satisfactory = [
        candidate
        for candidate in candidates
        if float(candidate["minimum_depth_mm"]) >= preferred_depth - 1e-9
    ]
    if satisfactory:
        return max(
            satisfactory,
            key=lambda candidate: (
                float(candidate["effective_insert_shrink_mm"]),
                float(candidate["minimum_depth_mm"]),
            ),
        ), "largest_taper_meeting_preferred_minimum_depth"
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate["minimum_depth_mm"]),
            float(candidate["effective_insert_shrink_mm"]),
        ),
    ), "deepest_safe_taper_when_preferred_minimum_unavailable"


def smooth_tapered_sweep_profile(
    visible_top_points: np.ndarray,
    interior_conormals: np.ndarray,
    safe_inward_directions: np.ndarray,
    safe_total_depths: np.ndarray,
    fit_clearance_mm: float,
    maximum_taper_depth_mm: float,
    target_slope_degrees: float = DEFAULT_LEAD_IN_SLOPE_DEGREES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build a visible 45-degree shoulder followed by a smooth deep stem.

    The former lead-in used only the fit allowance (often 0.1--0.2 mm), so a
    multi-millimetre extrusion still looked cylindrical.  This profile reserves
    a bounded fraction of the loop's hydraulic radius for the tapered shoulder;
    the remaining safe travel becomes the narrower stem.  The same smoothed
    ordered-loop direction field drives every generated ring.
    """
    visible = np.asarray(visible_top_points, dtype=np.float64)
    conormals = np.asarray(interior_conormals, dtype=np.float64)
    directions = np.asarray(safe_inward_directions, dtype=np.float64)
    depths = np.asarray(safe_total_depths, dtype=np.float64)
    if not (
        visible.shape == conormals.shape == directions.shape
        and depths.shape == (len(visible),)
        and len(visible) >= 3
    ):
        raise ValueError("smooth tapered sweep inputs must have matching loop shapes")
    conormal_lengths = np.linalg.norm(conormals, axis=1)
    if np.any(conormal_lengths <= 1e-12):
        raise ValueError("smooth tapered sweep conormals must be non-zero")
    conormals = conormals / conormal_lengths[:, None]
    smooth_directions, direction_record = smooth_closed_inward_direction_field(
        directions
    )

    total_depth = max(float(np.min(depths)), 0.0)
    slope_degrees = float(np.clip(float(target_slope_degrees), 1.0, 89.0))
    slope_tangent = float(np.tan(np.radians(slope_degrees)))

    clearance = max(float(fit_clearance_mm), 0.0)
    maximum_taper_depth = max(float(maximum_taper_depth_mm), 0.0)
    lateral_inset, inset_record = tapered_profile_inset_limit(
        visible,
        clearance,
        maximum_taper_depth,
        total_depth,
        slope_degrees,
    )
    shoulder_depth = min(lateral_inset * slope_tangent, total_depth)

    shoulder = (
        visible
        + conormals * lateral_inset
        + smooth_directions * shoulder_depth
    )
    stem_depth = max(total_depth - shoulder_depth, 0.0)
    bottom = shoulder + smooth_directions * stem_depth
    return shoulder, bottom, smooth_directions, {
        "profile": "smooth_45deg_shoulder_with_deep_stem",
        "target_slope_degrees": slope_degrees,
        "fit_clearance_mm": clearance,
        **inset_record,
        "shoulder_depth_mm": shoulder_depth,
        "total_depth_mm": total_depth,
        "stem_depth_mm": stem_depth,
        **direction_record,
    }


def normalized_circular_convolution(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Apply a normalized wraparound convolution without Python roll loops.

    The operation is the same closed-loop weighted sum used by the connector
    direction solver.  ``scipy.ndimage`` performs the physical-radius window
    in compiled code and avoids allocating one full rolled array per offset.
    """
    vectors = np.asarray(values, dtype=np.float64)
    kernel = np.asarray(weights, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("circular convolution values must be Nx3 vectors")
    if kernel.ndim != 1 or not len(kernel) or not np.all(np.isfinite(kernel)):
        raise ValueError("circular convolution weights must be one finite vector")
    result = convolve1d(vectors, kernel, axis=0, mode="wrap")
    result /= np.maximum(np.linalg.norm(result, axis=1)[:, None], 1e-12)
    return result


def safe_boundary_inward_directions(
    vertex_inward_normals: np.ndarray,
    loop: list[int],
    fallback_inward: np.ndarray,
    loop_points: np.ndarray | None = None,
    full_local_below_dot: float = 0.0,
    keep_global_above_dot: float = 0.50,
    flat_safe_dot: float = 0.05,
    smoothing_iterations: int = 32,
    smoothing_radius_mm: float = 1.20,
) -> tuple[np.ndarray, dict]:
    """Prefer one flat direction, otherwise build a continuous safe direction field."""
    fallback = np.asarray(fallback_inward, dtype=np.float64)
    fallback /= max(float(np.linalg.norm(fallback)), 1e-12)
    local = np.asarray(vertex_inward_normals[np.asarray(loop, dtype=np.int64)], dtype=np.float64)
    local /= np.maximum(np.linalg.norm(local, axis=1)[:, None], 1e-12)
    initial_dot = local @ fallback
    flat_direction_safe = bool(len(loop) == 0 or float(initial_dot.min()) >= float(flat_safe_dot))
    if flat_direction_safe:
        directions = np.tile(fallback, (len(loop), 1))
        adjacent_angles = np.zeros(len(loop), dtype=np.float64)
        return directions, {
            "vertices": int(len(loop)),
            "global_outward_vertices_before": int(np.count_nonzero(initial_dot < 0.0)),
            "global_outward_fraction_before": float(np.mean(initial_dot < 0.0)) if len(loop) else 0.0,
            "outward_vertices_after": 0,
            "minimum_global_dot_local_inward_before": float(initial_dot.min()) if len(loop) else 1.0,
            "minimum_safe_dot_local_inward_after": float(initial_dot.min()) if len(loop) else 1.0,
            "median_safe_dot_local_inward_after": float(np.median(initial_dot)) if len(loop) else 1.0,
            "corrected_vertices": 0,
            "maximum_direction_correction_degrees": 0.0,
            "maximum_adjacent_direction_angle_degrees": float(adjacent_angles.max()) if len(adjacent_angles) else 0.0,
            "p95_adjacent_direction_angle_degrees": 0.0,
            "flat_direction_safe": True,
            "flat_safe_dot_threshold": float(flat_safe_dot),
            "direction_mode": "single_global_flat_direction",
        }

    smoothing_half_window = 0
    smoothing_edge_median_mm = 0.0
    smooth_local = local.copy()
    if len(local) >= 4 and loop_points is not None:
        points = np.asarray(loop_points, dtype=np.float64)
        edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        positive_edges = edge_lengths[edge_lengths > 1e-8]
        if len(positive_edges):
            smoothing_edge_median_mm = float(np.median(positive_edges))
            maximum_window = max(1, min((len(local) - 1) // 4, 128))
            smoothing_half_window = min(
                maximum_window,
                max(2, int(math.ceil(max(float(smoothing_radius_mm), 0.0) / smoothing_edge_median_mm))),
            )
            offsets = np.arange(-smoothing_half_window, smoothing_half_window + 1, dtype=np.int64)
            sigma = max(float(smoothing_half_window) * 0.45, 1.0)
            weights = np.exp(-0.5 * (offsets.astype(np.float64) / sigma) ** 2)
            weights /= weights.sum()

            def circular_smooth(values: np.ndarray) -> np.ndarray:
                return normalized_circular_convolution(values, weights)

            smooth_local = circular_smooth(local)
        else:
            circular_smooth = None
    else:
        circular_smooth = None

    smooth_initial_dot = smooth_local @ fallback
    lower = float(full_local_below_dot)
    upper = max(float(keep_global_above_dot), lower + 1e-6)
    blend = np.clip((smooth_initial_dot - lower) / (upper - lower), 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    directions = smooth_local * (1.0 - blend[:, None]) + fallback[None, :] * blend[:, None]
    directions /= np.maximum(np.linalg.norm(directions, axis=1)[:, None], 1e-12)
    if circular_smooth is not None:
        directions = circular_smooth(directions)

    # Smooth on the closed boundary, then project minimally back into each
    # vertex's local inward hemisphere.  Repeating both operations prevents
    # the abrupt per-vertex normal changes which twist the bottom ring into
    # radial folds while retaining a strictly inward first-order direction.
    minimum_safe_dot = 0.02
    relaxation_kernel = np.asarray([0.25, 0.50, 0.25], dtype=np.float64)
    for _ in range(max(int(smoothing_iterations), 0)):
        directions = normalized_circular_convolution(
            directions,
            relaxation_kernel,
        )
        current_dot = np.einsum("ij,ij->i", smooth_local, directions)
        unsafe = current_dot < minimum_safe_dot
        if np.any(unsafe):
            unsafe_directions = directions[unsafe]
            unsafe_local = smooth_local[unsafe]
            unsafe_dot = current_dot[unsafe]
            tangent = unsafe_directions - unsafe_dot[:, None] * unsafe_local
            tangent_length = np.linalg.norm(tangent, axis=1)
            valid_tangent = tangent_length > 1e-12
            projected = np.empty_like(unsafe_directions)
            projected[valid_tangent] = (
                tangent[valid_tangent]
                / tangent_length[valid_tangent, None]
                * math.sqrt(max(0.0, 1.0 - minimum_safe_dot * minimum_safe_dot))
                + unsafe_local[valid_tangent] * minimum_safe_dot
            )
            projected[~valid_tangent] = unsafe_local[~valid_tangent]
            directions[unsafe] = projected
        directions /= np.maximum(np.linalg.norm(directions, axis=1)[:, None], 1e-12)

    final_dot = np.einsum("ij,ij->i", smooth_local, directions)
    unsafe = final_dot <= 1e-6
    if np.any(unsafe):
        directions[unsafe] = local[unsafe]
        final_dot[unsafe] = 1.0
    angular_change = np.degrees(
        np.arccos(np.clip(directions @ fallback, -1.0, 1.0))
    )
    adjacent_angles = np.degrees(
        np.arccos(np.clip(np.einsum("ij,ij->i", directions, np.roll(directions, -1, axis=0)), -1.0, 1.0))
    )
    corrected = initial_dot < upper
    return directions, {
        "vertices": int(len(loop)),
        "global_outward_vertices_before": int(np.count_nonzero(initial_dot < 0.0)),
        "global_outward_fraction_before": float(np.mean(initial_dot < 0.0)) if len(loop) else 0.0,
        "outward_vertices_after": int(np.count_nonzero(final_dot <= 0.0)),
        "minimum_global_dot_local_inward_before": float(initial_dot.min()) if len(loop) else 1.0,
        "minimum_safe_dot_local_inward_after": float(final_dot.min()) if len(loop) else 1.0,
        "median_safe_dot_local_inward_after": float(np.median(final_dot)) if len(loop) else 1.0,
        "corrected_vertices": int(np.count_nonzero(corrected)),
        "maximum_direction_correction_degrees": float(angular_change.max()) if len(loop) else 0.0,
        "maximum_adjacent_direction_angle_degrees": float(adjacent_angles.max()) if len(adjacent_angles) else 0.0,
        "p95_adjacent_direction_angle_degrees": float(np.percentile(adjacent_angles, 95.0)) if len(adjacent_angles) else 0.0,
        "flat_direction_safe": False,
        "flat_safe_dot_threshold": float(flat_safe_dot),
        "direction_smoothing_iterations": int(max(int(smoothing_iterations), 0)),
        "direction_smoothing_radius_mm": float(max(float(smoothing_radius_mm), 0.0)),
        "direction_smoothing_half_window_vertices": int(smoothing_half_window),
        "direction_smoothing_edge_median_mm": float(smoothing_edge_median_mm),
        "raw_local_outward_vertices_after": int(
            np.count_nonzero(np.einsum("ij,ij->i", local, directions) <= 0.0)
        ),
        "direction_mode": "smoothed_boundary_local_hemisphere",
    }


def reference_loop_inward_directions(
    ref: dict | None,
    global_loop: list[int],
    fallback_inward: np.ndarray,
) -> np.ndarray:
    mapping = (ref or {}).get("inward_by_global", {})
    fallback = np.asarray(fallback_inward, dtype=np.float64)
    fallback /= max(float(np.linalg.norm(fallback)), 1e-12)
    return np.asarray(
        [mapping.get(int(global_id), fallback) for global_id in global_loop],
        dtype=np.float64,
    )


def reference_loop_interior_conormals(
    ref: dict | None,
    global_loop: list[int],
) -> np.ndarray | None:
    mapping = (ref or {}).get("interior_conormal_by_global", {})
    if not mapping or any(int(global_id) not in mapping for global_id in global_loop):
        return None
    return np.asarray(
        [mapping[int(global_id)] for global_id in global_loop],
        dtype=np.float64,
    )


def effective_feature_clearance(
    component: Component,
    requested_mm: float,
    profile: str = "feature-adaptive",
    feature_ratio: float = 0.08,
    minimum_mm: float = 0.10,
) -> tuple[float, dict]:
    """Clamp clearance for small painted details without changing the visible top surface."""
    requested = max(float(requested_mm), 0.0)
    extents = np.asarray(component.bbox_max, dtype=np.float64) - np.asarray(
        component.bbox_min, dtype=np.float64
    )
    positive_extents = extents[extents > 1e-6]
    feature_width = float(positive_extents.min()) if len(positive_extents) else 0.0
    if profile == "fixed" or requested <= 0.0 or feature_width <= 0.0:
        effective = requested
        reason = "fixed_profile" if profile == "fixed" else "no_positive_feature_extent"
    else:
        feature_limit = max(float(minimum_mm), feature_width * max(float(feature_ratio), 1e-9))
        effective = min(requested, feature_limit)
        reason = "feature_scale_clamp" if effective < requested - 1e-12 else "requested_within_feature_limit"
    return effective, {
        "profile": str(profile),
        "requested_fit_clearance_mm": requested,
        "effective_fit_clearance_mm": float(effective),
        "feature_width_mm": feature_width,
        "feature_ratio": float(feature_ratio),
        "minimum_target_mm": float(minimum_mm),
        "reason": reason,
    }


def tapered_lead_depths(
    visible_top_points: np.ndarray,
    internal_fit_points: np.ndarray,
    lead_directions: np.ndarray,
    remaining_distances: np.ndarray,
    maximum_lead_depth_mm: float,
    target_slope_degrees: float = DEFAULT_LEAD_IN_SLOPE_DEGREES,
) -> tuple[np.ndarray, dict]:
    """Choose local lead depths from the lateral shrink for a stable taper."""
    visible = np.asarray(visible_top_points, dtype=np.float64)
    fit = np.asarray(internal_fit_points, dtype=np.float64)
    directions = np.asarray(lead_directions, dtype=np.float64)
    remaining = np.asarray(remaining_distances, dtype=np.float64)
    count = int(len(visible))
    if not (
        visible.shape == fit.shape == directions.shape
        and remaining.shape == (count,)
    ):
        raise ValueError("lead taper inputs must have matching shapes")
    direction_lengths = np.linalg.norm(directions, axis=1)
    if np.any(direction_lengths <= 1e-12) or not np.all(
        np.isfinite(direction_lengths)
    ):
        raise ValueError("lead taper directions must be finite non-zero vectors")
    directions = directions / direction_lengths[:, None]
    offsets = fit - visible
    axial_offsets = np.einsum("ij,ij->i", offsets, directions)
    lateral_vectors = offsets - directions * axial_offsets[:, None]
    lateral_shifts = np.linalg.norm(lateral_vectors, axis=1)
    slope_degrees = float(np.clip(target_slope_degrees, 1.0, 89.0))
    target_axial_depths = lateral_shifts * np.tan(np.deg2rad(slope_degrees))
    desired_added_depths = np.maximum(target_axial_depths - axial_offsets, 0.0)
    depth_caps = tapered_lead_depth_caps(remaining, maximum_lead_depth_mm)
    # ``maximum_lead_depth_mm`` is an exterior-to-shoulder axial limit, not
    # merely an allowance measured from the already faired internal ring.
    # Planar-arc retopology may move that ring slightly along the insertion axis;
    # adding the full nominal lead after such a shift can otherwise turn a
    # requested 5.00 mm shoulder into 5.2 mm in the emitted mesh.
    absolute_depth_caps = np.maximum(
        max(float(maximum_lead_depth_mm), 0.0) - axial_offsets,
        0.0,
    )
    lead_depths = np.minimum.reduce(
        (desired_added_depths, depth_caps, absolute_depth_caps)
    )
    actual_axial_depths = np.maximum(axial_offsets + lead_depths, 0.0)
    measurable = lateral_shifts > 1e-9
    measured_slopes = np.degrees(
        np.arctan2(actual_axial_depths[measurable], lateral_shifts[measurable])
    )
    record = {
        "lead_in_shape": "outer-large_inner-small_taper",
        "lead_in_slope_target_degrees": slope_degrees,
        "lead_in_slope_measured_min_degrees": (
            float(measured_slopes.min()) if len(measured_slopes) else None
        ),
        "lead_in_slope_measured_max_degrees": (
            float(measured_slopes.max()) if len(measured_slopes) else None
        ),
        "lead_in_lateral_shift_min_mm": (
            float(lateral_shifts.min()) if len(lateral_shifts) else 0.0
        ),
        "lead_in_lateral_shift_max_mm": (
            float(lateral_shifts.max()) if len(lateral_shifts) else 0.0
        ),
        "lead_in_depth_limit_mm": max(float(maximum_lead_depth_mm), 0.0),
    }
    return lead_depths, record


def tapered_lead_depth_caps(
    remaining_distances: np.ndarray,
    maximum_lead_depth_mm: float,
    minimum_stem_mm: float = 0.05,
) -> np.ndarray:
    """Reserve a printable short stem without halving the requested taper.

    The legacy ``remaining * 0.5`` cap forced a nominal 45-degree, 5 mm taper
    to use only 2.5 mm of axial shoulder, producing an actual angle near 26
    degrees.  Leave 0.05 mm (or ten percent for very shallow relief) so the lead
    and bottom rings stay distinct while nearly all useful depth can form the
    requested shoulder.
    """
    remaining = np.maximum(np.asarray(remaining_distances, dtype=np.float64), 0.0)
    stem_reserve = np.minimum(
        max(float(minimum_stem_mm), 0.0),
        remaining * 0.10,
    )
    return np.minimum(
        max(float(maximum_lead_depth_mm), 0.0),
        np.maximum(remaining - stem_reserve, 0.0),
    )


def coherent_tapered_lead_ring(
    visible_top_points: np.ndarray,
    internal_fit_points: np.ndarray,
    safe_inward_directions: np.ndarray,
    remaining_distances: np.ndarray,
    maximum_lead_depth_mm: float,
    target_slope_degrees: float = DEFAULT_LEAD_IN_SLOPE_DEGREES,
    reference_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build one smooth lead ring instead of one axial height per triangle.

    A boundary direction field can be locally safe yet still inherit small
    normal changes from every source triangle.  Extruding each point by its own
    direction and depth turns those changes into visible hills.  Prefer one
    coherent loop axis when the safety-corrected field shares a hemisphere and
    use one conservative loop depth in either mode.
    """
    visible = np.asarray(visible_top_points, dtype=np.float64)
    fit = np.asarray(internal_fit_points, dtype=np.float64)
    directions = np.asarray(safe_inward_directions, dtype=np.float64)
    remaining = np.asarray(remaining_distances, dtype=np.float64)
    if not (
        visible.shape == fit.shape == directions.shape
        and remaining.shape == (len(visible),)
    ):
        raise ValueError("coherent lead ring inputs must have matching shapes")
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths <= 1e-12) or not np.all(np.isfinite(lengths)):
        raise ValueError("coherent lead directions must be finite non-zero vectors")
    directions = directions / lengths[:, None]

    if reference_axis is None:
        effective_directions, direction_record = smooth_closed_inward_direction_field(
            directions
        )
    else:
        axis = np.asarray(reference_axis, dtype=np.float64).copy()
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("lead reference axis must be one finite 3-vector")
        axis_length = float(np.linalg.norm(axis))
        if axis_length <= 1e-12:
            raise ValueError("lead reference axis must be non-zero")
        axis /= axis_length
        if float(np.dot(axis, directions.mean(axis=0))) < 0.0:
            axis *= -1.0
        effective_directions = np.tile(axis, (len(directions), 1))
        direction_record = {
            "direction_mode": "reference_insertion_axis",
            "direction_mean_resultant": 1.0,
            "direction_minimum_axis_dot": 1.0,
            "direction_smoothing_passes": 0,
            "direction_adjacent_angle_max_degrees": 0.0,
        }
    coherent_axis = effective_directions.mean(axis=0)
    coherent_axis /= max(float(np.linalg.norm(coherent_axis)), 1e-12)

    raw_depths, record = tapered_lead_depths(
        visible,
        fit,
        effective_directions,
        remaining,
        maximum_lead_depth_mm,
        target_slope_degrees,
    )
    depth_caps = tapered_lead_depth_caps(remaining, maximum_lead_depth_mm)
    if reference_axis is None:
        loop_depth = min(
            float(np.median(raw_depths)) if len(raw_depths) else 0.0,
            float(depth_caps.min()) if len(depth_caps) else 0.0,
        )
        loop_depth = max(loop_depth, 0.0)
        lead_depths = np.full(len(visible), loop_depth, dtype=np.float64)
    else:
        # The generated shoulder stays smooth because every point shares one
        # insertion axis.  Per-vertex amounts only cancel axial offsets already
        # introduced by retopology and apply local wall-thickness ceilings.
        lead_depths = np.maximum(raw_depths, 0.0)
        loop_depth = float(np.median(lead_depths)) if len(lead_depths) else 0.0
    lead_points = fit + effective_directions * lead_depths[:, None]

    offsets = fit - visible
    axial_offsets = np.einsum("ij,ij->i", offsets, effective_directions)
    lateral_vectors = offsets - effective_directions * axial_offsets[:, None]
    lateral_shifts = np.linalg.norm(lateral_vectors, axis=1)
    actual_axial_depths = np.maximum(axial_offsets + lead_depths, 0.0)
    measurable = lateral_shifts > 1e-9
    measured_slopes = np.degrees(
        np.arctan2(actual_axial_depths[measurable], lateral_shifts[measurable])
    )
    record.update(
        {
            "lead_in_direction_mode": direction_record["direction_mode"],
            "lead_in_coherent_axis": coherent_axis.round(9).tolist(),
            "lead_in_direction_mean_resultant": direction_record[
                "direction_mean_resultant"
            ],
            "lead_in_direction_adjacent_angle_max_degrees": direction_record[
                "direction_adjacent_angle_max_degrees"
            ],
            "lead_in_loop_depth_mm": loop_depth,
            "lead_in_raw_depth_range_mm": (
                float(np.ptp(raw_depths)) if len(raw_depths) else 0.0
            ),
            "lead_in_final_depth_range_mm": (
                float(np.ptp(lead_depths)) if len(lead_depths) else 0.0
            ),
            "lead_in_slope_measured_min_degrees": (
                float(measured_slopes.min()) if len(measured_slopes) else None
            ),
            "lead_in_slope_measured_max_degrees": (
                float(measured_slopes.max()) if len(measured_slopes) else None
            ),
        }
    )
    return lead_points, lead_depths, effective_directions, record


def build_component_cut_references(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    model_center: np.ndarray,
    inward_overrides: dict[int, np.ndarray] | None = None,
    fit_clearance_by_part: dict[int, float] | None = None,
    clearance_mode: str = "insert-shrink",
) -> list[dict]:
    inward_overrides = inward_overrides or {}
    fit_clearance_by_part = fit_clearance_by_part or {}
    refs = []
    for index, component in enumerate(components, start=1):
        local_vertices, local_faces, _, global_vertex_ids = build_local_mesh(vertices, faces, component)
        loops = boundary_loops(local_faces)
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
) -> tuple[list[dict], dict[int, Component], dict[int, list[int]]]:
    context_started_at = time.perf_counter()
    from .layer_seam_planning import prepare_layer_seams, layer_child_boundary_topology
    vertices, faces, interface_retopology = prepare_layer_seams(
        vertices, faces, components, parent_index, direct_child_indices,
        assembly_children, boundary_neighbor_lookup, interface_retopology)
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
            faces, components, subtree, interface_retopology
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
                                # Screening and final selection must use one
                                # thickness horizon.  A shorter ray can observe
                                # an entry without its matching exit (or neither)
                                # and therefore cannot classify whether the
                                # origin is inside the parent.  Let the shared
                                # cap planner apply its ordinary authoritative
                                # ceiling here as it does in the replay below.
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
                        selected_fit_points = offset_points_along_conormals(
                            candidate_points,
                            candidate_conormals,
                            effective_insert_shrink_mm,
                        )
                        # Candidate evaluation above already used the complete
                        # authoritative thickness horizon.  Replaying the same
                        # deterministic plan here performed a third identical
                        # multi-million-candidate ray pass without adding a
                        # stronger safety check; retain the selected evaluated
                        # result instead.
                        candidate_decision = selected["decision"]
                        authoritative_reserved_distances = np.asarray(
                            selected["reserved_distances"], dtype=np.float64
                        ).copy()
                        reserved_record = dict(selected["reserved_record"])
                        screening_minimum_depth_mm = float(
                            selected["minimum_depth_mm"]
                        )
                        authoritative_minimum_depth_mm = float(
                            np.min(authoritative_reserved_distances)
                        )
                        if (
                            screening_minimum_depth_mm
                            >= preferred_minimum_depth_mm - 1e-9
                            and authoritative_minimum_depth_mm
                            < preferred_minimum_depth_mm - 1e-9
                        ):
                            raise RuntimeError(
                                "target-depth screening disagrees with the authoritative "
                                "parent-thickness audit"
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
                                "taper_profile_authoritative_replan_applied": True,
                                "taper_profile_authoritative_minimum_depth_mm": (
                                    authoritative_minimum_depth_mm
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
                                authoritative_reserved_distances,
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
                                    "safe_depth_mm": (
                                        hidden_candidate.safe_depth_mm
                                    ),
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
                                "safe_depth_mm": hidden_candidate.safe_depth_mm,
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
                                    selected_hidden.safe_depth_mm
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
                                selected_hidden.safe_depth_mm
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
                    local_connector_safety = local_connector_safe_depth_at_footprint(
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
                        visible_interface_simplification_tolerance=float(
                            interface_retopology.config.visible_interface_simplification_tolerance
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
                        "局部连接器已使用排除子树后的内部足迹预计算安全深度",
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


def fit_plane_normal(points: np.ndarray, fallback_normal: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        normal = np.asarray(vh[-1], dtype=np.float64)
    except np.linalg.LinAlgError:
        normal = np.asarray(fallback_normal, dtype=np.float64)
    if float(np.linalg.norm(normal)) <= 1e-12:
        normal = np.asarray(fallback_normal, dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    if float(np.dot(normal, fallback_normal)) < 0.0:
        normal = -normal
    return normal


class ParentThicknessProbe:
    """Measure the first opposite-surface hit along inward boundary rays."""

    # A merely positive dot accepts an almost tangent side wall as an opposing
    # shell.  Prefer a materially opposing face whenever one exists later on
    # the same finite ray, while retaining weak-facing hits as a compatibility
    # fallback for genuinely oblique shells.
    minimum_preferred_opposing_face_dot = 0.15

    @dataclass(frozen=True)
    class BroadPhaseCandidateCache:
        """Direction-independent conservative candidates for repeated rays.

        Candidate groups contain global triangle ids.  A triangle intersecting
        any finite ray of ``search_limit_mm`` from an origin has its centroid
        within that limit plus its bounding radius, irrespective of direction.
        The normal per-direction capsule filter and exact ray test still run
        for every measurement.
        """

        probe_identity: int
        points: np.ndarray
        search_limit_mm: float
        candidate_groups: tuple[tuple[np.ndarray, ...], ...]
        candidate_count: int

    def prepare_safety_limit_candidates(
        self,
        points: np.ndarray,
        global_ceiling_mm: float,
    ) -> "ParentThicknessProbe.BroadPhaseCandidateCache":
        """Build a reusable superset for repeated ``safety_limit`` calls."""
        origins = np.asarray(points, dtype=np.float64)
        global_ceiling = min(
            max(float(global_ceiling_mm), 0.0),
            MAXIMUM_SAFE_INWARD_DEPTH_MM,
        )
        search_limit = global_ceiling + PARENT_THICKNESS_CLEARANCE_MM
        bucket_groups: list[tuple[np.ndarray, ...]] = []
        candidate_count = 0
        for triangle_ids, tree, maximum_radius in self.radius_buckets:
            local_groups = tree.query_ball_point(
                origins,
                search_limit + float(maximum_radius) + 1e-8,
                workers=-1 if len(origins) >= 64 else 1,
            )
            global_groups = tuple(
                triangle_ids[np.asarray(group, dtype=np.int64)]
                for group in local_groups
            )
            candidate_count += sum(len(group) for group in global_groups)
            bucket_groups.append(global_groups)
        return self.BroadPhaseCandidateCache(
            probe_identity=id(self),
            points=origins.copy(),
            search_limit_mm=search_limit,
            candidate_groups=tuple(bucket_groups),
            candidate_count=int(candidate_count),
        )

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        triangle_source_face_indices: np.ndarray | None = None,
        triangle_owner_indices: np.ndarray | None = None,
    ) -> None:
        self.mesh_vertices = np.asarray(vertices, dtype=np.float64)
        self.mesh_faces = np.asarray(faces, dtype=np.int64)
        self.triangles = self.mesh_vertices[self.mesh_faces]
        self._surface_topology_cache: dict = {}
        triangle_count = int(len(self.triangles))
        self.triangle_source_face_indices = (
            np.arange(triangle_count, dtype=np.int64)
            if triangle_source_face_indices is None
            else np.asarray(
                triangle_source_face_indices,
                dtype=np.int64,
            ).copy()
        )
        self.triangle_owner_indices = (
            np.zeros(triangle_count, dtype=np.int32)
            if triangle_owner_indices is None
            else np.asarray(
                triangle_owner_indices,
                dtype=np.int32,
            ).copy()
        )
        if self.triangle_source_face_indices.shape != (triangle_count,):
            raise ValueError(
                "triangle_source_face_indices must match the probe triangle count"
            )
        if self.triangle_owner_indices.shape != (triangle_count,):
            raise ValueError(
                "triangle_owner_indices must match the probe triangle count"
            )
        self.active_triangle_mask = np.ones(triangle_count, dtype=bool)
        self.active_triangle_count = triangle_count
        self.filter_context: dict = {
            "excluded_triangle_count": 0,
            "included_triangle_count": triangle_count,
        }
        self.centroids = self.triangles.mean(axis=1)
        self.radii = np.linalg.norm(
            self.triangles - self.centroids[:, None, :],
            axis=2,
        ).max(axis=1)
        self.normals = np.cross(
            self.triangles[:, 1] - self.triangles[:, 0],
            self.triangles[:, 2] - self.triangles[:, 0],
        )
        self.normals /= np.maximum(
            np.linalg.norm(self.normals, axis=1)[:, None],
            1e-12,
        )
        self.maximum_radius = float(self.radii.max()) if len(self.radii) else 0.0
        self.radius_buckets: list[tuple[np.ndarray, cKDTree, float]] = []
        if len(self.centroids):
            radius_scale_keys = np.ceil(
                np.log2(np.maximum(self.radii, 1e-9))
            ).astype(np.int16)
            for key in np.unique(radius_scale_keys):
                triangle_ids = np.flatnonzero(
                    radius_scale_keys == key
                ).astype(np.int64)
                bucket_centroids = self.centroids[triangle_ids]
                bucket_maximum_radius = float(
                    self.radii[triangle_ids].max()
                )
                self.radius_buckets.append(
                    (
                        triangle_ids,
                        cKDTree(bucket_centroids),
                        bucket_maximum_radius,
                    )
                )

    def excluding_triangles(
        self,
        triangle_indices: np.ndarray,
        filter_context: dict | None = None,
    ) -> "ParentThicknessProbe":
        """Return a lightweight probe view that shares acceleration data."""
        excluded = np.unique(
            np.asarray(triangle_indices, dtype=np.int64)
        )
        if len(excluded) and (
            int(excluded.min()) < 0
            or int(excluded.max()) >= len(self.triangles)
        ):
            raise ValueError("excluded triangle index is outside the probe")
        active_mask = np.asarray(
            self.active_triangle_mask,
            dtype=bool,
        ).copy()
        active_mask[excluded] = False
        view = object.__new__(type(self))
        for attribute in (
            "mesh_vertices",
            "mesh_faces",
            "_surface_topology_cache",
            "triangles",
            "triangle_source_face_indices",
            "triangle_owner_indices",
            "centroids",
            "radii",
            "normals",
            "maximum_radius",
        ):
            setattr(view, attribute, getattr(self, attribute))
        view.active_triangle_mask = active_mask
        view.active_triangle_count = int(np.count_nonzero(active_mask))
        # Rebuild only the inexpensive centroid trees for the active parent
        # shell.  Sharing the original bucket trees made every later query
        # return child-subtree triangles which were immediately discarded.
        # Recursive cap planning probes the same filtered parent many times, so
        # paying this construction cost once removes that noise from every ray.
        view.radius_buckets = []
        for bucket_triangle_ids, _bucket_tree, _bucket_maximum_radius in self.radius_buckets:
            active_triangle_ids = bucket_triangle_ids[
                active_mask[bucket_triangle_ids]
            ]
            if not len(active_triangle_ids):
                continue
            active_centroids = self.centroids[active_triangle_ids]
            active_maximum_radius = float(
                self.radii[active_triangle_ids].max()
            )
            view.radius_buckets.append(
                (
                    active_triangle_ids,
                    cKDTree(active_centroids),
                    active_maximum_radius,
                )
            )
        view.maximum_radius = max(
            (float(bucket[2]) for bucket in view.radius_buckets),
            default=0.0,
        )
        excluded_owner_indices = sorted(
            int(value)
            for value in np.unique(
                self.triangle_owner_indices[excluded]
            )
            if int(value) > 0
        )
        view.filter_context = {
            **dict(filter_context or {}),
            "excluded_triangle_count": int(
                len(active_mask) - view.active_triangle_count
            ),
            "included_triangle_count": int(view.active_triangle_count),
            "excluded_owner_indices": excluded_owner_indices,
        }
        return view

    def _same_local_surface_hits(
        self,
        points: np.ndarray,
        hit_triangle_ids: np.ndarray,
        hit_distances: np.ndarray,
    ) -> np.ndarray:
        """Recognize short ray hits reachable along the sampled shell.

        Euclidean distance alone cannot distinguish a nearby fold of the
        source surface from an opposing wall.  A fold is also nearby along the
        mesh graph, while an actual inner/opposite shell requires a much longer
        route around the closed solid.  Only the small near-origin candidate
        band is audited, keeping this exact topological test inexpensive.
        """
        result = np.zeros(len(points), dtype=bool)
        # Audit the whole preferred printable-depth range.  Stopping at a
        # small Euclidean band merely moved the minimum to the next triangle
        # on a continuous curved surface (0.13 -> 0.26 mm on the Yoshi seam).
        # Topology, rather than this horizon, decides whether a hit is local;
        # the horizon only avoids work on hits that already provide the normal
        # requested insertion depth.
        horizon = DEFAULT_EFFECTIVE_MINIMUM_INWARD_DEPTH_MM
        candidates = np.flatnonzero(
            (hit_triangle_ids >= 0)
            & np.isfinite(hit_distances)
            & (hit_distances <= horizon + 1e-9)
        )
        if not len(candidates) or not len(self.mesh_vertices):
            return result
        cache = self._surface_topology_cache
        if "vertex_tree" not in cache:
            cache["vertex_tree"] = cKDTree(self.mesh_vertices)
        if "vertex_graph" not in cache:
            edges = np.vstack(
                (
                    self.mesh_faces[:, [0, 1]],
                    self.mesh_faces[:, [1, 2]],
                    self.mesh_faces[:, [2, 0]],
                )
            )
            lengths = np.linalg.norm(
                self.mesh_vertices[edges[:, 0]]
                - self.mesh_vertices[edges[:, 1]],
                axis=1,
            )
            rows = np.concatenate((edges[:, 0], edges[:, 1]))
            cols = np.concatenate((edges[:, 1], edges[:, 0]))
            weights = np.concatenate((lengths, lengths))
            cache["vertex_graph"] = coo_matrix(
                (weights, (rows, cols)),
                shape=(len(self.mesh_vertices), len(self.mesh_vertices)),
            ).tocsr()
        _nearest_distances, origin_vertices = cache["vertex_tree"].query(
            np.asarray(points)[candidates], k=1
        )
        for candidate, origin_vertex in zip(candidates, origin_vertices):
            triangle_id = int(hit_triangle_ids[candidate])
            if triangle_id >= len(self.mesh_faces):
                continue
            chord = float(hit_distances[candidate])
            # Allow curved/folded paths to be longer than their chord.  The
            # search remains local and cannot walk around a hollow shell to an
            # actual opposite wall.
            geodesic_limit = max(4.0 * chord, horizon)
            distances = dijkstra(
                cache["vertex_graph"],
                directed=False,
                indices=int(origin_vertex),
                limit=geodesic_limit,
            )
            if np.any(np.isfinite(distances[self.mesh_faces[triangle_id]])):
                result[candidate] = True
        return result

    def _owner_counts(self, triangle_ids: np.ndarray) -> dict[str, int]:
        triangle_ids = np.asarray(triangle_ids, dtype=np.int64)
        owner_indices = np.asarray(
            getattr(
                self,
                "triangle_owner_indices",
                np.empty(0, dtype=np.int32),
            ),
            dtype=np.int32,
        )
        triangle_ids = triangle_ids[
            (triangle_ids >= 0)
            & (triangle_ids < len(owner_indices))
        ]
        if not len(triangle_ids):
            return {}
        owners, counts = np.unique(
            owner_indices[triangle_ids],
            return_counts=True,
        )
        return {
            str(int(owner)): int(count)
            for owner, count in zip(owners, counts)
            if int(owner) > 0
        }

    def first_hit_distances(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        search_limit_mm: float,
        broad_phase_cache: "ParentThicknessProbe.BroadPhaseCandidateCache | None" = None,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1)[:, None],
            1e-12,
        )
        search_limit = max(float(search_limit_mm), 0.0)
        misses = search_limit + PARENT_THICKNESS_CLEARANCE_MM
        hits = np.full(len(points), misses, dtype=np.float64)
        self.last_hit_had_preceding_entry = np.zeros(
            len(points),
            dtype=bool,
        )
        if not self.radius_buckets or search_limit <= 0.0:
            return hits
        if broad_phase_cache is not None:
            if broad_phase_cache.probe_identity != id(self):
                raise ValueError("parent-thickness candidate cache belongs to another probe")
            if (
                broad_phase_cache.points.shape != points.shape
                or not np.array_equal(broad_phase_cache.points, points)
                or search_limit > broad_phase_cache.search_limit_mm + 1e-9
            ):
                raise ValueError("parent-thickness candidate cache does not cover this query")
            if len(broad_phase_cache.candidate_groups) != len(self.radius_buckets):
                raise ValueError("parent-thickness candidate cache bucket mismatch")

        query_started_at = time.perf_counter()
        runtime_log(
            "厚度测量",
            "thickness_candidates_start",
            "开始查询父体厚度射线候选三角形",
            probe_vertex_count=int(len(points)),
            source_triangle_count=int(len(self.triangles)),
            search_limit_mm=round(float(search_limit), 6),
            maximum_triangle_radius_mm=round(float(self.maximum_radius), 6),
            radius_bucket_count=int(len(self.radius_buckets)),
        )
        midpoints = points + directions * (search_limit * 0.5)
        candidate_count = 0
        query_elapsed_seconds = 0.0
        intersection_elapsed_seconds = 0.0
        broad_phase_segment_counts: collections.Counter = collections.Counter()
        exit_hits: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [
            [] for _ in range(len(points))
        ]
        entry_hits: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [
            [] for _ in range(len(points))
        ]
        selected_exit_distances = np.full(len(points), np.nan, dtype=np.float64)
        selected_entry_distances = np.full(len(points), np.nan, dtype=np.float64)
        selected_exit_triangle_ids = np.full(len(points), -1, dtype=np.int64)
        selected_entry_triangle_ids = np.full(len(points), -1, dtype=np.int64)
        for bucket_index, (
            bucket_triangle_ids,
            bucket_tree,
            bucket_maximum_radius,
        ) in enumerate(self.radius_buckets):
            bucket_query_started_at = time.perf_counter()
            # A single sphere around the complete segment has radius L/2+r and
            # is extremely loose for dense small triangles.  Cover the ray by
            # up to eight conservative capsule cells instead.  If a triangle
            # intersects a cell, its centroid is within triangle_radius plus
            # half the cell length of that cell's midpoint, so this cannot drop
            # a true hit while dramatically shrinking the broad phase.
            target_cell_length = max(
                2.0 * float(bucket_maximum_radius),
                search_limit / 8.0,
                1e-6,
            )
            segment_count = int(
                np.clip(
                    math.ceil(search_limit / target_cell_length),
                    1,
                    8,
                )
            )
            broad_phase_segment_counts[int(segment_count)] += 1
            if broad_phase_cache is not None:
                # The cached groups are a direction-independent superset.
                # Keep them as global ids; the segment-sphere rejection below
                # restores the tight per-direction capsule before exact tests.
                candidate_groups = broad_phase_cache.candidate_groups[bucket_index]
            elif segment_count == 1:
                candidate_groups = bucket_tree.query_ball_point(
                    midpoints,
                    search_limit * 0.5 + bucket_maximum_radius + 1e-8,
                    workers=-1 if len(points) >= 64 else 1,
                )
            else:
                cell_length = search_limit / float(segment_count)
                fractions = (
                    np.arange(segment_count, dtype=np.float64) + 0.5
                ) / float(segment_count)
                sample_points = (
                    points[:, None, :]
                    + directions[:, None, :]
                    * (search_limit * fractions[None, :, None])
                )
                sampled_groups = bucket_tree.query_ball_point(
                    sample_points.reshape(-1, 3),
                    bucket_maximum_radius + cell_length * 0.5 + 1e-8,
                    workers=-1 if len(points) >= 64 else 1,
                )
                candidate_groups = []
                for point_index in range(len(points)):
                    start = int(point_index * segment_count)
                    stop = int(start + segment_count)
                    groups = sampled_groups[start:stop]
                    nonempty = [
                        np.asarray(group, dtype=np.int64)
                        for group in groups
                        if len(group)
                    ]
                    if not nonempty:
                        candidate_groups.append([])
                    elif len(nonempty) == 1:
                        candidate_groups.append(nonempty[0].tolist())
                    else:
                        candidate_groups.append(
                            np.unique(np.concatenate(nonempty)).tolist()
                        )
            query_elapsed_seconds += float(
                time.perf_counter() - bucket_query_started_at
            )
            candidate_count += int(
                sum(len(group) for group in candidate_groups)
            )

            bucket_intersection_started_at = time.perf_counter()
            for index, (origin, direction, candidates) in enumerate(
                zip(points, directions, candidate_groups)
            ):
                if not len(candidates):
                    continue
                candidate_ids = (
                    np.asarray(candidates, dtype=np.int64)
                    if broad_phase_cache is not None
                    else bucket_triangle_ids[np.asarray(candidates, dtype=np.int64)]
                )
                candidate_ids = candidate_ids[
                    self.active_triangle_mask[candidate_ids]
                ]
                if not len(candidate_ids):
                    continue

                centroid_delta = self.centroids[candidate_ids] - origin
                ray_projection = centroid_delta @ direction
                clamped_projection = np.clip(
                    ray_projection,
                    0.0,
                    search_limit,
                )
                closest = origin + clamped_projection[:, None] * direction
                sphere_distance = np.linalg.norm(
                    self.centroids[candidate_ids] - closest,
                    axis=1,
                )
                candidate_ids = candidate_ids[
                    sphere_distance <= self.radii[candidate_ids] + 1e-7
                ]
                if not len(candidate_ids):
                    continue

                triangles = self.triangles[candidate_ids]
                edge_1 = triangles[:, 1] - triangles[:, 0]
                edge_2 = triangles[:, 2] - triangles[:, 0]
                h = np.cross(
                    np.broadcast_to(direction, edge_2.shape),
                    edge_2,
                )
                determinant = np.einsum("ij,ij->i", edge_1, h)
                active = np.abs(determinant) > 1e-12
                inverse = np.zeros_like(determinant)
                inverse[active] = 1.0 / determinant[active]
                s = origin - triangles[:, 0]
                u = inverse * np.einsum("ij,ij->i", s, h)
                q = np.cross(s, edge_1)
                v = inverse * (q @ direction)
                distance = inverse * np.einsum("ij,ij->i", edge_2, q)
                geometric_hit = (
                    active
                    & (u >= -1e-9)
                    & (v >= -1e-9)
                    & (u + v <= 1.0 + 1e-9)
                    & (distance > 1e-4)
                    & (distance <= search_limit + 1e-9)
                )
                if not np.any(geometric_hit):
                    continue
                hit_distances = distance[geometric_hit]
                hit_triangle_ids = candidate_ids[geometric_hit]
                hit_facing = (
                    self.normals[candidate_ids][geometric_hit] @ direction
                )
                exiting_mask = hit_facing > 1e-6
                entering_mask = hit_facing < -1e-6
                exiting = hit_distances[exiting_mask]
                entering = hit_distances[entering_mask]
                if len(exiting):
                    exit_hits[index].append(
                        (
                            exiting,
                            hit_triangle_ids[exiting_mask],
                            hit_facing[exiting_mask],
                        )
                    )
                if len(entering):
                    entry_hits[index].append(
                        (
                            entering,
                            hit_triangle_ids[entering_mask],
                            hit_facing[entering_mask],
                        )
                    )
            intersection_elapsed_seconds += float(
                time.perf_counter() - bucket_intersection_started_at
            )

        runtime_log(
            "厚度测量",
            "thickness_candidates_ready",
            "父体厚度射线候选三角形查询完成",
            probe_vertex_count=int(len(points)),
            source_triangle_count=int(len(self.triangles)),
            candidate_count=candidate_count,
            average_candidates_per_vertex=round(
                float(candidate_count / max(len(points), 1)),
                3,
            ),
            maximum_triangle_radius_mm=round(float(self.maximum_radius), 6),
            radius_bucket_count=int(len(self.radius_buckets)),
            broad_phase_segment_counts={
                str(int(key)): int(value)
                for key, value in sorted(broad_phase_segment_counts.items())
            },
            query_elapsed_seconds=round(float(query_elapsed_seconds), 3),
        )
        weak_exit_fallback_vertices = 0
        rejected_weak_exit_hits = 0
        preferred_dot = float(self.minimum_preferred_opposing_face_dot)
        for index, exiting_groups in enumerate(exit_hits):
            if not exiting_groups:
                continue
            has_preferred_exit = any(
                bool(np.any(group_facing >= preferred_dot))
                for _group_distances, _group_triangle_ids, group_facing in exiting_groups
            )
            if not has_preferred_exit:
                weak_exit_fallback_vertices += 1
            exit_distance = math.inf
            exit_triangle_id = -1
            for group_distances, group_triangle_ids, group_facing in exiting_groups:
                eligible_indices = np.flatnonzero(
                    group_facing >= preferred_dot
                ) if has_preferred_exit else np.arange(len(group_distances))
                if has_preferred_exit:
                    rejected_weak_exit_hits += int(
                        np.count_nonzero(group_facing < preferred_dot)
                    )
                if not len(eligible_indices):
                    continue
                group_index = int(
                    eligible_indices[
                        np.argmin(group_distances[eligible_indices])
                    ]
                )
                candidate_distance = float(group_distances[int(group_index)])
                if candidate_distance < exit_distance:
                    exit_distance = candidate_distance
                    exit_triangle_id = int(group_triangle_ids[int(group_index)])
            selected_exit_distances[index] = exit_distance
            selected_exit_triangle_ids[index] = exit_triangle_id
            entry_distance = 0.0
            entry_triangle_id = -1
            entering_groups = entry_hits[index]
            has_preferred_entry = any(
                bool(np.any(group_facing <= -preferred_dot))
                for _group_distances, _group_triangle_ids, group_facing in entering_groups
            )
            for group_distances, group_triangle_ids, group_facing in entering_groups:
                eligible_indices = np.flatnonzero(
                    group_distances < exit_distance - 1e-5
                )
                if has_preferred_entry:
                    eligible_indices = eligible_indices[
                        group_facing[eligible_indices] <= -preferred_dot
                    ]
                if len(eligible_indices):
                    group_index = int(
                        eligible_indices[
                            np.argmax(group_distances[eligible_indices])
                        ]
                    )
                    candidate_distance = float(
                        group_distances[group_index]
                    )
                    if candidate_distance > entry_distance:
                        entry_distance = candidate_distance
                        entry_triangle_id = int(
                            group_triangle_ids[group_index]
                        )
            if entry_distance > 0.0:
                selected_entry_distances[index] = entry_distance
                selected_entry_triangle_ids[index] = entry_triangle_id
            hits[index] = exit_distance - entry_distance
        near_hit_mask = hits <= PARENT_THICKNESS_CLEARANCE_MM + 1e-9
        paired_hit_mask = np.isfinite(selected_entry_distances)
        self.last_hit_had_preceding_entry = paired_hit_mask.copy()
        self.last_selected_exit_triangle_ids = (
            selected_exit_triangle_ids.copy()
        )
        self.last_selected_entry_triangle_ids = (
            selected_entry_triangle_ids.copy()
        )
        # Preserve the actual ray distances as well as the interval thickness.
        # When a ray starts in free space and later crosses a very thin opposing
        # shell, the usable insertion depth is limited by the shell entry, not by
        # the tiny entry-to-exit thickness of that remote shell.
        self.last_selected_exit_distances = selected_exit_distances.copy()
        self.last_selected_entry_distances = selected_entry_distances.copy()
        self.last_hit_diagnostics = {
            "near_hit_vertices": int(np.count_nonzero(near_hit_mask)),
            "near_hit_with_preceding_entry_vertices": int(
                np.count_nonzero(near_hit_mask & paired_hit_mask)
            ),
            "near_hit_without_preceding_entry_vertices": int(
                np.count_nonzero(near_hit_mask & ~paired_hit_mask)
            ),
            "selected_exit_distance_min_mm": (
                float(np.nanmin(selected_exit_distances))
                if np.any(np.isfinite(selected_exit_distances))
                else None
            ),
            "selected_preceding_entry_distance_max_mm": (
                float(np.nanmax(selected_entry_distances))
                if np.any(np.isfinite(selected_entry_distances))
                else None
            ),
            "selected_exit_owner_counts": self._owner_counts(
                selected_exit_triangle_ids
            ),
            "selected_preceding_entry_owner_counts": self._owner_counts(
                selected_entry_triangle_ids
            ),
            "preferred_opposing_face_dot_threshold": preferred_dot,
            "weak_exit_fallback_vertices": int(weak_exit_fallback_vertices),
            "rejected_weak_exit_hits": int(rejected_weak_exit_hits),
            "probe_filter": dict(
                getattr(self, "filter_context", {})
            ),
        }
        runtime_log(
            "厚度测量",
            "thickness_intersections_done",
            "父体厚度射线相交计算完成",
            probe_vertex_count=int(len(points)),
            candidate_count=candidate_count,
            measured_hit_count=int(np.count_nonzero(hits < misses)),
            intersection_elapsed_seconds=round(
                float(intersection_elapsed_seconds),
                3,
            ),
            total_elapsed_seconds=round(
                float(time.perf_counter() - query_started_at),
                3,
            ),
            **self.last_hit_diagnostics,
        )
        return hits

    def safety_limit(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        global_ceiling_mm: float,
        broad_phase_cache: "ParentThicknessProbe.BroadPhaseCandidateCache | None" = None,
    ) -> tuple[float, dict]:
        global_ceiling = min(
            max(float(global_ceiling_mm), 0.0),
            MAXIMUM_SAFE_INWARD_DEPTH_MM,
        )
        search_limit = global_ceiling + PARENT_THICKNESS_CLEARANCE_MM
        if broad_phase_cache is None:
            # Preserve the overridable three-argument hook used by lightweight
            # policy probes and downstream integrations.
            thicknesses = self.first_hit_distances(
                points,
                directions,
                search_limit,
            )
        else:
            thicknesses = self.first_hit_distances(
                points,
                directions,
                search_limit,
                broad_phase_cache=broad_phase_cache,
            )
        original_thicknesses = thicknesses.copy()
        preceding_entry_distances = getattr(
            self,
            "last_selected_entry_distances",
            None,
        )
        selected_entry_triangle_ids = getattr(
            self,
            "last_selected_entry_triangle_ids",
            np.full(len(thicknesses), -1, dtype=np.int64),
        )
        parent_part_index = int(
            getattr(self, "filter_context", {}).get("parent_part_index", 0)
        )
        triangle_owner_indices = np.asarray(
            getattr(self, "triangle_owner_indices", np.empty(0, dtype=np.int32))
        )
        entry_owner_indices = np.full(len(thicknesses), 0, dtype=np.int32)
        valid_entry_triangle_ids = (
            (selected_entry_triangle_ids >= 0)
            & (selected_entry_triangle_ids < len(triangle_owner_indices))
        )
        entry_owner_indices[valid_entry_triangle_ids] = triangle_owner_indices[
            selected_entry_triangle_ids[valid_entry_triangle_ids]
        ]
        # Re-entering the shell currently being inset is a local surface fold,
        # even when curvature puts the re-entry farther from the ray origin.
        # It is not a remote obstacle.  Owner identity supplies the topological
        # distinction that a distance threshold cannot: genuinely separate
        # shells retain a different owner and continue to limit travel at
        # their entry point.
        paired_parent_reentry = (
            parent_part_index > 0
        ) & (entry_owner_indices == parent_part_index)
        if (
            hasattr(self, "mesh_vertices")
            and isinstance(preceding_entry_distances, np.ndarray)
            and preceding_entry_distances.shape == thicknesses.shape
        ):
            # Paint ownership partitions one physical shell into many parts;
            # it is not a shell-connectivity label.  A differently painted
            # entry that is reachable locally along the same mesh is still a
            # folded/re-entered source surface, not a remote obstacle.
            paired_parent_reentry |= self._same_local_surface_hits(
                points,
                selected_entry_triangle_ids,
                preceding_entry_distances,
            )
        paired_remote_shell_repair = np.zeros_like(thicknesses, dtype=bool)
        if (
            isinstance(preceding_entry_distances, np.ndarray)
            and preceding_entry_distances.shape == thicknesses.shape
        ):
            # ``first_hit_distances`` returns the material interval between a
            # preceding entry and its exit.  That interval is a valid local
            # wall thickness only when the ray starts on (within numerical
            # reserve of) that wall.  If the entry is remote, the generated
            # cap travels through free space first and is limited by the entry
            # location, regardless of whether the remote shell itself is
            # hair-thin or substantial.  Restricting this repair to intervals
            # below the clearance made a bounded screening ray report a safe
            # target while the longer authoritative ray incorrectly treated a
            # 0.218 mm remote shell as the available depth.
            paired_remote_shell_repair = (
                np.isfinite(preceding_entry_distances)
                & ~paired_parent_reentry
                & (
                    preceding_entry_distances
                    > PARENT_THICKNESS_CLEARANCE_MM + 1e-9
                )
            )
            thicknesses[paired_remote_shell_repair] = (
                preceding_entry_distances[paired_remote_shell_repair]
            )
        raw_minimum = float(thicknesses.min()) if len(thicknesses) else search_limit
        selected_exit_distances = getattr(
            self,
            "last_selected_exit_distances",
            None,
        )
        has_exit_classification = bool(
            isinstance(selected_exit_distances, np.ndarray)
            and selected_exit_distances.shape == thicknesses.shape
        )
        selected_exit_triangle_ids = getattr(
            self,
            "last_selected_exit_triangle_ids",
            np.full(len(thicknesses), -1, dtype=np.int64),
        )
        topologically_local_surface = np.zeros_like(thicknesses, dtype=bool)
        if (
            parent_part_index > 0
            and hasattr(self, "mesh_vertices")
            and selected_exit_triangle_ids.shape == thicknesses.shape
        ):
            exit_owner_indices = np.zeros(len(thicknesses), dtype=np.int32)
            valid_exit_ids = (
                (selected_exit_triangle_ids >= 0)
                & (selected_exit_triangle_ids < len(triangle_owner_indices))
            )
            exit_owner_indices[valid_exit_ids] = triangle_owner_indices[
                selected_exit_triangle_ids[valid_exit_ids]
            ]
            topologically_local_surface = self._same_local_surface_hits(
                points,
                selected_exit_triangle_ids,
                np.where(
                    exit_owner_indices == parent_part_index,
                    selected_exit_distances,
                    np.inf,
                ),
            )
        # A hit belongs to the sampled surface neighbourhood when either its
        # complete exit or its still-local material interval is within the
        # origin tolerance.  A paired entry inside the manufacturing clearance
        # is also unconditionally local: it means that the ray started in the
        # seam uncertainty band, briefly crossed out of the tessellated parent,
        # then entered it again.  The following exit can be arbitrarily far
        # from that entry along a folded surface, so classifying the pair by
        # interval length creates a brittle threshold chase.  Remote paired
        # shells were replaced by their entry distance above and are excluded
        # from this local-pair classification.
        local_paired_surface = (
            np.isfinite(preceding_entry_distances)
            & (~paired_remote_shell_repair | paired_parent_reentry)
            if isinstance(preceding_entry_distances, np.ndarray)
            and preceding_entry_distances.shape == thicknesses.shape
            else np.zeros_like(thicknesses, dtype=bool)
        )
        surface_near = (
            (thicknesses <= PARENT_SURFACE_HIT_TOLERANCE_MM + 1e-9)
            | local_paired_surface
            | topologically_local_surface
            | (
                (
                    selected_exit_distances
                    if has_exit_classification
                    else np.full_like(thicknesses, np.inf)
                )
                <= PARENT_SURFACE_HIT_TOLERANCE_MM + 1e-9
            )
        )
        coincident = thicknesses <= PARENT_THICKNESS_CLEARANCE_MM + 1e-9
        coincident_count = int(np.count_nonzero(coincident))
        coincident_ratio = float(coincident_count / max(len(thicknesses), 1))
        preceding_entry_mask = getattr(
            self,
            "last_hit_had_preceding_entry",
            None,
        )
        has_entry_classification = bool(
            isinstance(preceding_entry_mask, np.ndarray)
            and preceding_entry_mask.shape == thicknesses.shape
        )
        near_origin_surface_hits = (
            surface_near
            if has_entry_classification
            else np.zeros_like(coincident)
        )
        near_origin_surface_hit_count = int(
            np.count_nonzero(near_origin_surface_hits)
        )
        usable_mask = ~near_origin_surface_hits
        after_unpaired = thicknesses[usable_mask]
        after_unpaired_coincident = (
            after_unpaired
            <= PARENT_THICKNESS_CLEARANCE_MM + 1e-9
        )
        after_unpaired_coincident_count = int(
            np.count_nonzero(after_unpaired_coincident)
        )
        after_unpaired_coincident_ratio = float(
            after_unpaired_coincident_count
            / max(len(thicknesses), 1)
        )
        discard_isolated_coincident = bool(
            after_unpaired_coincident_count
            and after_unpaired_coincident_ratio < 0.01
            and np.any(~after_unpaired_coincident)
        )
        usable_thicknesses = (
            after_unpaired[~after_unpaired_coincident]
            if discard_isolated_coincident
            else after_unpaired
        )
        if discard_isolated_coincident:
            usable_mask &= ~coincident
        measured_minimum = (
            float(usable_thicknesses.min())
            if len(usable_thicknesses)
            else search_limit
        )
        thickness_quantiles = (
            np.quantile(usable_thicknesses, [0.0, 0.01, 0.05, 0.50, 0.95, 1.0])
            if len(usable_thicknesses)
            else np.full(6, search_limit, dtype=np.float64)
        )
        safe_maximum = min(
            global_ceiling,
            max(0.0, measured_minimum - PARENT_THICKNESS_CLEARANCE_MM),
        )
        measured_hits = thicknesses <= search_limit + 1e-9
        limiting_vertex_indices = np.flatnonzero(
            usable_mask
            & np.isclose(
                thicknesses,
                measured_minimum,
                rtol=0.0,
                atol=1e-9,
            )
        )
        selected_entry_triangle_ids = getattr(
            self,
            "last_selected_entry_triangle_ids",
            np.full(len(thicknesses), -1, dtype=np.int64),
        )
        limiting_exit_triangle_ids = selected_exit_triangle_ids[
            limiting_vertex_indices
        ]
        limiting_entry_triangle_ids = selected_entry_triangle_ids[
            limiting_vertex_indices
        ]
        triangle_source_face_indices = np.asarray(
            getattr(
                self,
                "triangle_source_face_indices",
                np.empty(0, dtype=np.int64),
            ),
            dtype=np.int64,
        )
        valid_limiting_exit_triangle_ids = limiting_exit_triangle_ids[
            (limiting_exit_triangle_ids >= 0)
            & (
                limiting_exit_triangle_ids
                < len(triangle_source_face_indices)
            )
        ]
        valid_limiting_entry_triangle_ids = limiting_entry_triangle_ids[
            (limiting_entry_triangle_ids >= 0)
            & (
                limiting_entry_triangle_ids
                < len(triangle_source_face_indices)
            )
        ]
        limiting_exit_source_faces_all = sorted(
            {
            int(value)
            for value in triangle_source_face_indices[
                valid_limiting_exit_triangle_ids
            ]
            }
        )
        limiting_entry_source_faces_all = sorted(
            {
            int(value)
            for value in triangle_source_face_indices[
                valid_limiting_entry_triangle_ids
            ]
            }
        )
        limiting_vertex_sample = [
            int(value) for value in limiting_vertex_indices[:16]
        ]
        limiting_probe_points = np.asarray(points, dtype=np.float64)[
            limiting_vertex_indices[:16]
        ].round(9).tolist()
        limiting_probe_directions = np.asarray(directions, dtype=np.float64)[
            limiting_vertex_indices[:16]
        ].round(9).tolist()
        limiting_exit_source_faces = limiting_exit_source_faces_all[:16]
        limiting_entry_source_faces = limiting_entry_source_faces_all[:16]
        result = {
            "parent_thickness_min_mm": measured_minimum,
            "parent_thickness_raw_min_mm": raw_minimum,
            "parent_thickness_interval_raw_min_mm": (
                float(original_thicknesses.min())
                if len(original_thicknesses)
                else search_limit
            ),
            "parent_thickness_remote_shell_interval_hits_repaired": int(
                np.count_nonzero(paired_remote_shell_repair)
            ),
            "parent_thickness_parent_reentry_hits_discarded": int(
                np.count_nonzero(paired_parent_reentry)
            ),
            "parent_thickness_topologically_local_hits_discarded": int(
                np.count_nonzero(topologically_local_surface)
            ),
            "parent_thickness_coincident_hit_vertices": coincident_count,
            "parent_thickness_coincident_hit_ratio": coincident_ratio,
            "parent_thickness_unpaired_surface_hit_vertices_discarded": (
                near_origin_surface_hit_count
            ),
            "parent_thickness_near_origin_surface_hit_vertices_discarded": (
                near_origin_surface_hit_count
            ),
            "parent_thickness_remaining_coincident_hit_vertices": (
                after_unpaired_coincident_count
            ),
            "parent_thickness_remaining_coincident_hit_ratio": (
                after_unpaired_coincident_ratio
            ),
            "parent_thickness_isolated_coincident_hits_discarded": discard_isolated_coincident,
            "parent_thickness_hit_vertices": int(np.count_nonzero(measured_hits)),
            "parent_thickness_probe_vertices": int(len(points)),
            "parent_thickness_is_lower_bound": bool(not np.all(measured_hits)),
            "parent_thickness_quantiles_mm": {
                key: float(value)
                for key, value in zip(
                    ("minimum", "p01", "p05", "median", "p95", "maximum"),
                    thickness_quantiles,
                )
            },
            "parent_thickness_clearance_mm": PARENT_THICKNESS_CLEARANCE_MM,
            "parent_surface_hit_tolerance_mm": (
                PARENT_SURFACE_HIT_TOLERANCE_MM
            ),
            "safe_maximum_inward_depth_mm": safe_maximum,
            "parent_thickness_limiting_probe_vertex_indices": [
                int(value) for value in limiting_vertex_sample
            ],
            "parent_thickness_limiting_probe_vertex_count": int(
                len(limiting_vertex_indices)
            ),
            "parent_thickness_limiting_probe_points": limiting_probe_points,
            "parent_thickness_limiting_probe_directions": limiting_probe_directions,
            "parent_thickness_limiting_exit_face_indices": (
                limiting_exit_source_faces
            ),
            "parent_thickness_limiting_exit_face_count": int(
                len(limiting_exit_source_faces_all)
            ),
            "parent_thickness_limiting_entry_face_indices": (
                limiting_entry_source_faces
            ),
            "parent_thickness_limiting_entry_face_count": int(
                len(limiting_entry_source_faces_all)
            ),
            "parent_thickness_limiting_exit_owner_counts": (
                self._owner_counts(limiting_exit_triangle_ids)
            ),
            "parent_thickness_limiting_entry_owner_counts": (
                self._owner_counts(limiting_entry_triangle_ids)
            ),
            "parent_thickness_probe_filter": dict(
                getattr(self, "filter_context", {})
            ),
            "parent_thickness_hit_diagnostics": dict(
                getattr(self, "last_hit_diagnostics", {})
            ),
        }
        runtime_log(
            "厚度测量",
            "parent_thickness_limit_selected",
            "父体厚度安全上限及限制命中面已确定",
            measured_parent_thickness_mm=float(measured_minimum),
            safe_maximum_inward_depth_mm=float(safe_maximum),
            limiting_probe_vertex_indices=[
                int(value) for value in limiting_vertex_sample
            ],
            limiting_probe_vertex_count=int(len(limiting_vertex_indices)),
            limiting_probe_points=limiting_probe_points,
            limiting_probe_directions=limiting_probe_directions,
            limiting_exit_face_indices=limiting_exit_source_faces,
            limiting_exit_face_count=int(
                len(limiting_exit_source_faces_all)
            ),
            limiting_entry_face_indices=limiting_entry_source_faces,
            limiting_entry_face_count=int(
                len(limiting_entry_source_faces_all)
            ),
            limiting_exit_owner_counts=self._owner_counts(
                limiting_exit_triangle_ids
            ),
            limiting_entry_owner_counts=self._owner_counts(
                limiting_entry_triangle_ids
            ),
            probe_filter=dict(
                getattr(self, "filter_context", {})
            ),
        )
        return safe_maximum, result


def boundary_cap_distances(
    points: np.ndarray,
    fallback_inward: np.ndarray,
    inward_directions: np.ndarray,
    fixed_depth_mm: float,
    flat_clearance_mm: float,
    cap_mode: str,
    planar_extra_limit_mm: float,
    parent_thickness_probe: ParentThicknessProbe | None = None,
    safety_ceiling_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Use a common bottom plane when it fits; otherwise use safe local offset."""
    points = np.asarray(points, dtype=np.float64)
    fallback = np.asarray(fallback_inward, dtype=np.float64)
    fallback /= max(float(np.linalg.norm(fallback)), 1e-12)
    local_plane_rays = np.asarray(inward_directions, dtype=np.float64)
    local_plane_rays /= np.maximum(
        np.linalg.norm(local_plane_rays, axis=1)[:, None],
        1e-12,
    )
    directions, smooth_direction_record = smooth_closed_inward_direction_field(
        local_plane_rays
    )
    preferred_minimum = max(float(fixed_depth_mm), 0.4)
    global_ceiling = min(
        preferred_minimum + max(float(planar_extra_limit_mm), 0.0),
        MAXIMUM_SAFE_INWARD_DEPTH_MM,
    )
    if safety_ceiling_mm is not None:
        global_ceiling = min(
            global_ceiling,
            max(float(safety_ceiling_mm), 0.0),
        )

    def measured_limit(candidate_directions: np.ndarray) -> tuple[float, dict]:
        if parent_thickness_probe is None:
            return global_ceiling, {
                "parent_thickness_min_mm": global_ceiling
                + PARENT_THICKNESS_CLEARANCE_MM,
                "parent_thickness_hit_vertices": 0,
                "parent_thickness_probe_vertices": int(len(points)),
                "parent_thickness_is_lower_bound": True,
                "parent_thickness_clearance_mm": PARENT_THICKNESS_CLEARANCE_MM,
                "safe_maximum_inward_depth_mm": global_ceiling,
            }
        return parent_thickness_probe.safety_limit(
            points,
            candidate_directions,
            global_ceiling,
        )

    candidate_specs: list[tuple[str, np.ndarray]] = [("global_inward", fallback)]
    best_fit_normal = fit_plane_normal(points, fallback)
    best_fit_dot_global = float(np.dot(best_fit_normal, fallback))
    if best_fit_dot_global >= 0.15:
        candidate_specs.append(("loop_best_fit_normal", best_fit_normal))

    # Every candidate plane uses the same smoothed local ray field.  Thickness
    # depends on those rays, not on the normal used to place the common bottom
    # plane, so measuring once is both exact and substantially cheaper on dense
    # painted seams.  Previously the global and best-fit plane candidates each
    # repeated the identical broad phase and ray/triangle intersection pass.
    shared_safe_maximum, shared_thickness_record = measured_limit(
        local_plane_rays
    )
    plane_candidates = []
    for orientation, plane_direction in candidate_specs:
        # The cap plane and the travel rays solve different problems.  The
        # former must be one coherent printable plane; the latter should retain
        # the already-smoothed local inward field so thickness is measured
        # through the parent instead of along an arbitrary average axis.  Each
        # local ray is intersected with the common plane below, so varying rays
        # cannot reintroduce the old wavy/folded bottom surface.
        plane_directions = local_plane_rays.copy()
        ray_dot_plane = plane_directions @ plane_direction
        if len(ray_dot_plane) and float(ray_dot_plane.min()) <= 0.05:
            continue
        safe_maximum = shared_safe_maximum
        thickness_record = dict(shared_thickness_record)
        if safe_maximum <= 1e-6:
            effective_minimum = 0.0
        else:
            effective_minimum = min(preferred_minimum, safe_maximum)
        values = points @ plane_direction
        plane_s = float(np.min(values + safe_maximum * ray_dot_plane))
        distances = (plane_s - values) / ray_dot_plane
        required_maximum = (
            float(distances.max()) if len(distances) else effective_minimum
        )
        minimum_travel = (
            float(distances.min()) if len(distances) else effective_minimum
        )
        regular_feasible = (
            effective_minimum > 0.0
            and minimum_travel >= effective_minimum - 1e-9
            and required_maximum <= safe_maximum + 1e-9
        )
        thin_parent_limited = bool(
            safe_maximum < preferred_minimum - 1e-9
        )
        thin_parent_planar_minimum = min(
            safe_maximum,
            MINIMUM_COHERENT_PLANAR_FLOOR_DEPTH_MM,
        )
        thin_parent_planar_feasible = bool(
            thin_parent_limited
            and thin_parent_planar_minimum > 0.0
            and minimum_travel >= thin_parent_planar_minimum - 1e-9
            and required_maximum <= safe_maximum + 1e-9
        )
        feasible = bool(regular_feasible or thin_parent_planar_feasible)
        forced_planar_feasible = feasible or (
            preferred_minimum <= 0.4 + 1e-9
            and safe_maximum > 0.0
            and minimum_travel >= PARENT_THICKNESS_CLEARANCE_MM - 1e-9
            and required_maximum <= safe_maximum + 1e-9
        )
        plane_candidates.append(
            {
                "orientation": orientation,
                "direction": plane_direction,
                "directions": plane_directions,
                "distances": distances,
                "plane_s": plane_s,
                "span_mm": float(values.max() - values.min()) if len(values) else 0.0,
                "required_maximum_mm": required_maximum,
                "minimum_travel_mm": minimum_travel,
                "effective_minimum_mm": effective_minimum,
                "safe_maximum_mm": safe_maximum,
                "minimum_ray_dot_plane": (
                    float(ray_dot_plane.min()) if len(ray_dot_plane) else 1.0
                ),
                "regular_feasible": regular_feasible,
                "thin_parent_limited": thin_parent_limited,
                "thin_parent_planar_minimum_mm": thin_parent_planar_minimum,
                "thin_parent_planar_feasible": thin_parent_planar_feasible,
                "feasible": feasible,
                "forced_planar_feasible": forced_planar_feasible,
                "thickness_record": thickness_record,
            }
        )

    if cap_mode == "tilted":
        selectable = [
            candidate
            for candidate in plane_candidates
            if candidate["orientation"] == "loop_best_fit_normal"
            and candidate["forced_planar_feasible"]
        ]
    elif cap_mode == "flat":
        selectable = [
            candidate
            for candidate in plane_candidates
            if candidate["orientation"] == "global_inward"
            and candidate["forced_planar_feasible"]
        ]
    elif cap_mode == "offset":
        selectable = []
    else:
        selectable = [candidate for candidate in plane_candidates if candidate["feasible"]]

    if selectable:
        selected = min(
            selectable,
            key=lambda candidate: (
                -candidate["minimum_travel_mm"],
                0 if candidate["orientation"] == "global_inward" else 1,
            ),
        )
        distances = np.asarray(selected["distances"], dtype=np.float64)
        record = {
            "requested_cap_mode": cap_mode,
            "cap_mode": "flat",
            "flat_orientation": selected["orientation"],
            "cap_depth_reference": "common_plane_with_variable_point_depth",
            "inward_depth_policy": "deepest_safe_common_plane",
            "preferred_minimum_inward_depth_mm": preferred_minimum,
            "effective_minimum_inward_depth_mm": min(
                selected["effective_minimum_mm"],
                selected["minimum_travel_mm"],
            ),
            "safe_maximum_inward_depth_mm": selected["safe_maximum_mm"],
            "flat_span_mm": selected["span_mm"],
            "flat_bottom_required_max_extension_mm": selected["required_maximum_mm"],
            "flat_bottom_plane_s": selected["plane_s"],
            "flat_plane_extension_max_mm": selected["required_maximum_mm"],
            "minimum_generated_inward_travel_mm": float(distances.min()),
            "maximum_generated_inward_travel_mm": float(distances.max()),
            "thin_parent_planar_floor_applied": bool(
                selected["thin_parent_planar_feasible"]
                and not selected["regular_feasible"]
            ),
            "thin_parent_planar_minimum_depth_mm": float(
                selected["thin_parent_planar_minimum_mm"]
            ),
            "thin_parent_planar_achieved_minimum_depth_mm": float(
                selected["minimum_travel_mm"]
            ),
            "flat_priority_applied": True,
            "flat_feasibility_basis": (
                "thin_parent_printable_planar_floor"
                if selected["thin_parent_planar_feasible"]
                and not selected["regular_feasible"]
                else "required_plane_depth_within_parent_thickness_safety_limit"
            ),
            "flat_travel_within_limit": True,
            "target_exceeded": bool(
                selected["required_maximum_mm"] > preferred_minimum + 1e-9
            ),
            "target_exceeded_by_mm": float(
                max(0.0, selected["required_maximum_mm"] - preferred_minimum)
            ),
            "fixed_inward_depth_mm": preferred_minimum,
            "fixed_inward_depth_applied": bool(
                selected["minimum_travel_mm"] >= preferred_minimum - 1e-9
            ),
            "local_direction_correction_applied": False,
            "local_direction_corrected_vertices": 0,
            "minimum_direction_dot_global": 1.0,
            "best_fit_flat_normal": best_fit_normal.round(6).tolist(),
            "best_fit_flat_normal_dot_global": best_fit_dot_global,
            "plane_candidate_required_maxima_mm": {
                candidate["orientation"]: candidate["required_maximum_mm"]
                for candidate in plane_candidates
            },
            **smooth_direction_record,
            **selected["thickness_record"],
        }
        return distances, np.asarray(selected["directions"]), record

    if cap_mode in {"flat", "tilted"}:
        if not plane_candidates:
            raise ValueError(
                f"{cap_mode} cap cannot fit: no usable plane candidate"
            )
        attempted = (
            plane_candidates[0]
            if cap_mode == "flat"
            else next(
                (
                    candidate
                    for candidate in plane_candidates
                    if candidate["orientation"] == "loop_best_fit_normal"
                ),
                plane_candidates[0],
            )
        )
        if (
            preferred_minimum <= 0.4 + 1e-9
            and parent_thickness_probe is not None
            and len(points) >= 3
        ):
            uniform_directions = np.asarray(
                attempted["directions"],
                dtype=np.float64,
            )
            raw_thicknesses = parent_thickness_probe.first_hit_distances(
                points,
                uniform_directions,
                global_ceiling + PARENT_THICKNESS_CLEARANCE_MM,
            )
            # Keep the normal absolute reserve wherever the source shell is
            # thick enough.  For user-authorized source walls thinner than the
            # reserve itself, preserve half of the measured thickness so both
            # sides remain positive; the closed-loop Lipschitz pass then spreads
            # that exceptional shallow point with a <=45 degree transition.
            per_vertex_reserve = np.where(
                raw_thicknesses < PARENT_THICKNESS_CLEARANCE_MM,
                raw_thicknesses * 0.5,
                PARENT_THICKNESS_CLEARANCE_MM,
            )
            per_vertex_safe = np.minimum(
                global_ceiling,
                np.maximum(0.0, raw_thicknesses - per_vertex_reserve),
            )
            # ``per_vertex_safe`` has already had the full parent-thickness
            # clearance subtracted above.  Requiring another clearance-sized
            # extrusion here double-counts that reserve and rejects valid,
            # very shallow local relief points.  A strictly positive travel
            # is sufficient to keep the generated wall non-degenerate while
            # preserving the requested clearance from every measured hit.
            if float(per_vertex_safe.min()) > 1e-6:
                distances = per_vertex_safe.copy()
                edge_lengths = np.linalg.norm(
                    np.roll(points, -1, axis=0) - points,
                    axis=1,
                )
                for _ in range(3):
                    for index in range(1, len(distances)):
                        distances[index] = min(
                            distances[index],
                            distances[index - 1] + edge_lengths[index - 1],
                        )
                    distances[0] = min(
                        distances[0],
                        distances[-1] + edge_lengths[-1],
                    )
                    for index in range(len(distances) - 2, -1, -1):
                        distances[index] = min(
                            distances[index],
                            distances[index + 1] + edge_lengths[index],
                        )
                    distances[-1] = min(
                        distances[-1],
                        distances[0] + edge_lengths[-1],
                    )
                depth_deltas = np.abs(
                    np.roll(distances, -1) - distances
                )
                slope_degrees = np.degrees(
                    np.arctan2(
                        depth_deltas,
                        np.maximum(edge_lengths, 1e-12),
                    )
                )
                record = {
                    "requested_cap_mode": cap_mode,
                    "cap_mode": "uniform-direction-lipschitz",
                    "flat_orientation": attempted["orientation"],
                    "cap_depth_reference": "per_vertex_measured_ceiling_with_uniform_direction",
                    "inward_depth_policy": "maximum_safe_depth_with_planar_arc_interface",
                    "preferred_minimum_inward_depth_mm": preferred_minimum,
                    "effective_minimum_inward_depth_mm": float(distances.min()),
                    "safe_maximum_inward_depth_mm": global_ceiling,
                    "minimum_generated_inward_travel_mm": float(distances.min()),
                    "maximum_generated_inward_travel_mm": float(distances.max()),
                    "per_vertex_raw_safe_minimum_mm": float(per_vertex_safe.min()),
                    "per_vertex_raw_safe_maximum_mm": float(per_vertex_safe.max()),
                    "per_vertex_reserve_minimum_mm": float(per_vertex_reserve.min()),
                    "per_vertex_half_thickness_reserve_count": int(
                        np.count_nonzero(
                            raw_thicknesses
                            < PARENT_THICKNESS_CLEARANCE_MM - 1e-9
                        )
                    ),
                    "per_vertex_limited_count": int(
                        np.count_nonzero(per_vertex_safe < global_ceiling - 1e-9)
                    ),
                    "per_vertex_faired_count": int(
                        np.count_nonzero(distances < per_vertex_safe - 1e-9)
                    ),
                    "depth_field_slope_max_degrees": float(slope_degrees.max()),
                    "flat_span_mm": attempted["span_mm"],
                    "flat_bottom_required_max_extension_mm": float(distances.max()),
                    "flat_bottom_plane_s": None,
                    "flat_plane_extension_max_mm": float(distances.max()),
                    "flat_priority_applied": False,
                    "flat_feasibility_basis": "common_plane_blocked_by_local_thickness",
                    "flat_travel_within_limit": True,
                    "target_exceeded": bool(
                        float(distances.max()) > preferred_minimum + 1e-9
                    ),
                    "target_exceeded_by_mm": float(
                        max(0.0, float(distances.max()) - preferred_minimum)
                    ),
                    "fixed_inward_depth_mm": preferred_minimum,
                    "fixed_inward_depth_applied": False,
                    "local_direction_correction_applied": False,
                    "local_direction_corrected_vertices": 0,
                    "minimum_direction_dot_global": float(
                        np.min(uniform_directions @ fallback)
                    ),
                    "best_fit_flat_normal": best_fit_normal.round(6).tolist(),
                    "best_fit_flat_normal_dot_global": best_fit_dot_global,
                    "plane_candidate_required_maxima_mm": {
                        candidate["orientation"]: candidate["required_maximum_mm"]
                        for candidate in plane_candidates
                    },
                    **smooth_direction_record,
                }
                return distances, uniform_directions, record
            zero_safe_indices = np.flatnonzero(per_vertex_safe <= 1e-6)
            raise ValueError(
                f"Forced {cap_mode} uniform-direction fallback has no positive "
                "travel after parent-thickness clearance: "
                f"raw_thickness_min={float(raw_thicknesses.min()):.9f}mm, "
                f"zero_safe_count={int(len(zero_safe_indices))}, "
                f"zero_safe_indices={zero_safe_indices[:16].tolist()}, "
                f"clearance={PARENT_THICKNESS_CLEARANCE_MM:.6f}mm"
            )
        raise ValueError(
            f"Forced {cap_mode} cap cannot fit inside the parent-thickness safety limit: "
            f"required={attempted['required_maximum_mm']:.6f}mm, "
            f"safe_maximum={attempted['safe_maximum_mm']:.6f}mm, "
            f"minimum_travel={attempted['minimum_travel_mm']:.6f}mm"
        )

    local_safe_maximum, local_thickness_record = measured_limit(directions)
    if local_safe_maximum <= 1e-6:
        raise ValueError(
            "No positive inward depth remains after the 0.05 mm parent-thickness clearance: "
            + json.dumps(local_thickness_record, ensure_ascii=False, sort_keys=True)
        )
    local_target_depth = min(preferred_minimum, local_safe_maximum)
    distances = np.full(len(points), local_target_depth, dtype=np.float64)
    required_plane_maximum = min(
        (
            float(candidate["required_maximum_mm"])
            for candidate in plane_candidates
        ),
        default=preferred_minimum,
    )
    direction_dot = np.clip(directions @ fallback, -1.0, 1.0)
    record = {
        "requested_cap_mode": cap_mode,
        "cap_mode": "local-offset",
        "cap_depth_reference": "per_boundary_vertex_safe_local_inward_direction",
        "preferred_minimum_inward_depth_mm": preferred_minimum,
        "effective_minimum_inward_depth_mm": min(
            preferred_minimum,
            local_safe_maximum,
        ),
        "safe_maximum_inward_depth_mm": local_safe_maximum,
        "local_offset_depth_policy": "requested_depth_clamped_to_measured_ceiling",
        "flat_span_mm": float(
            min(
                (candidate["span_mm"] for candidate in plane_candidates),
                default=0.0,
            )
        ),
        "flat_bottom_required_max_extension_mm": required_plane_maximum,
        "flat_bottom_plane_s": None,
        "flat_plane_extension_max_mm": required_plane_maximum,
        "minimum_generated_inward_travel_mm": local_target_depth,
        "maximum_generated_inward_travel_mm": local_target_depth,
        "flat_priority_applied": False,
        "flat_feasibility_basis": "required_plane_depth_exceeds_parent_thickness_safety_limit",
        "flat_travel_within_limit": False,
        "local_fallback_reason": "required_plane_depth_exceeds_safe_maximum",
        "target_exceeded": bool(required_plane_maximum > preferred_minimum + 1e-9),
        "target_exceeded_by_mm": float(
            max(0.0, required_plane_maximum - preferred_minimum)
        ),
        "fixed_inward_depth_mm": preferred_minimum,
        "fixed_inward_depth_applied": bool(
            abs(local_target_depth - preferred_minimum) <= 1e-9
        ),
        "local_direction_correction_applied": bool(
            np.any(direction_dot < 1.0 - 1e-8)
        ),
        "local_direction_corrected_vertices": int(
            np.count_nonzero(direction_dot < 1.0 - 1e-8)
        ),
        "minimum_direction_dot_global": (
            float(direction_dot.min()) if len(direction_dot) else 1.0
        ),
        "best_fit_flat_normal": best_fit_normal.round(6).tolist(),
        "best_fit_flat_normal_dot_global": best_fit_dot_global,
        "plane_candidate_required_maxima_mm": {
            candidate["orientation"]: candidate["required_maximum_mm"]
            for candidate in plane_candidates
        },
        **smooth_direction_record,
        **local_thickness_record,
    }
    return distances, directions, record


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
    record["backing_taper_target_degrees"] = 45.0
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
        "Printable 45-degree backing profile is ready",
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
        if layer_index == len(backing_profile.rings) - 1:
            tolerance = float(
                plan.get("visible_interface_simplification_tolerance_mm", 0.0)
            )
            original_count = len(layer_points)
            retained = np.arange(original_count, dtype=np.int64)
            simplification_skip_reason = None
            # Source-indexed transition faces are the only safe way to join a
            # heavily reduced backing contour to a dense visible rim.  If an
            # earlier profile stage has already changed the ring cardinality,
            # there is no longer a one-to-one source-index correspondence.
            # Keep that ring intact instead of feeding a three-point RDP result
            # into the generic annulus solver, where it would be expanded with
            # straight chords that erase real concavities.
            source_index_correspondence = original_count == len(previous_ids)
            if tolerance > 0.0 and original_count > 3 and source_index_correspondence:
                from .contour_simplification import simplify_closed_contour

                projected = _project_connector_points(layer_points, plan)
                retained = simplify_closed_contour(projected, tolerance)
                candidate = layer_points[retained]
                candidate_2d = projected[retained]
                compact_ring = _project_connector_points(
                    np.asarray(plan["peg_top"], dtype=np.float64), plan
                )
                # RDP chords can cut across a deep concavity.  Keep the dense
                # ring rather than changing annulus/component topology when a
                # compact connector would cease to be enclosed.
                if all(
                    point_in_poly(point, candidate_2d)
                    or point_on_poly_boundary(point, candidate_2d)
                    for point in compact_ring
                ):
                    layer_points = candidate
                else:
                    retained = np.arange(original_count, dtype=np.int64)
                    simplification_skip_reason = "compact_connector_not_enclosed"
            elif tolerance > 0.0 and original_count > 3:
                simplification_skip_reason = "source_index_correspondence_unavailable"
            plan["visible_interface_simplification_source_indices"] = [
                int(value) for value in retained
            ]
            plan["visible_interface_outer_vertices_before_simplification"] = int(
                original_count
            )
            plan["visible_interface_outer_vertices_after_simplification"] = int(
                len(layer_points)
            )
            plan["visible_interface_simplification_applied"] = bool(
                len(layer_points) < original_count
            )
            plan["visible_interface_simplification_skip_reason"] = (
                simplification_skip_reason
            )
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
        simplification_applied = bool(
            float(plan.get("visible_interface_simplification_tolerance_mm", 0.0)) > 0.0
            and len(layer_ids) < len(previous_ids)
        )
        retained = plan.get("visible_interface_simplification_source_indices")
        original_outer_count = int(
            plan.get("visible_interface_outer_vertices_before_simplification", -1)
        )
        if (
            simplification_applied
            and isinstance(retained, list)
            and len(retained) == len(layer_ids)
            and len(previous_ids) == original_outer_count
        ):
            count = len(previous_ids)
            faces: list[tuple[int, int, int]] = []
            maximum_fanout_seen = 0
            for inner_index, start_value in enumerate(retained):
                start = int(start_value)
                end = int(retained[(inner_index + 1) % len(retained)])
                run = (end - start) % count
                maximum_fanout_seen = max(maximum_fanout_seen, run + 1)
                for step in range(run):
                    current = (start + step) % count
                    following = (current + 1) % count
                    faces.append((
                        int(previous_ids[current]),
                        int(previous_ids[following]),
                        int(layer_ids[inner_index]),
                    ))
                faces.append((
                    int(previous_ids[end]),
                    int(layer_ids[(inner_index + 1) % len(layer_ids)]),
                    int(layer_ids[inner_index]),
                ))
            point_lookup = {
                **{int(i): np.asarray(p, dtype=np.float64) for i, p in zip(previous_ids, previous_points)},
                **{int(i): np.asarray(p, dtype=np.float64) for i, p in zip(layer_ids, layer_points)},
            }
            faces = orient_face_patch_consistently(
                faces, point_lookup, np.asarray(plan["inward"], dtype=np.float64)
            )
            triangles = np.asarray(
                [[point_lookup[value] for value in face] for face in faces],
                dtype=np.float64,
            )
            double_areas = np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                axis=1,
            )
            if np.any(double_areas <= 1e-12):
                raise ValueError("simplified backing transition contains a degenerate face")
            output_faces.extend([list(face) for face in faces])
            edge_lengths = np.linalg.norm(
                triangles[:, [1, 2, 0]] - triangles[:, [0, 1, 2]], axis=2
            )
            added = len(faces)
            strip_audit = {
                "valid": True,
                "reason": "",
                "strategy": "source_indexed_simplified_profile_transition",
                "maximum_fanout": int(maximum_fanout_seen),
                "maximum_cross_edge_mm": float(edge_lengths.max(initial=0.0)),
            }
            returned_points = layer_points
        else:
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
                else local_connector_safe_depth_at_footprint(
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
                else local_connector_safe_depth_at_footprint(
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


class InwardDirectionPlanner:
    """Build a smooth, locally inward-safe direction field for one boundary."""

    plan = staticmethod(safe_boundary_inward_directions)


class AdaptiveCapPlanner:
    """Choose coherent flat or smooth local-offset cap geometry."""

    choose = staticmethod(boundary_cap_distances)


class BoundaryTriangulator:
    """Triangulate simple, spatial, and holed boundary caps without center fans."""

    polygon = staticmethod(triangulate_polygon_ear_clip)
    spatial_loop = staticmethod(triangulate_ordered_loop_3d)
    cap = staticmethod(triangulate_cap)


class PartMeshBuilder:
    """Build final insert, body-cut, and recursive subassembly meshes."""

    build_part = staticmethod(make_part_mesh)
    build_body_cut = staticmethod(make_body_cut_mesh)
    build_layer_subassembly = staticmethod(make_layer_child_subassembly_mesh)
