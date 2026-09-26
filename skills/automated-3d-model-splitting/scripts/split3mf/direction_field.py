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

def visible_top_edge_clearance(insert_shrink_mm: float) -> float:
    """Keep the exterior cut ring coincident; start fit clearance below it.

    ``insert_shrink_mm`` remains part of the signature so every caller routes
    the requested internal fit through this explicit visible-surface policy.
    The lead-in ring still transitions to the full shrink below the source
    surface, while the original/fared top ring is never radially offset.
    """
    _ = max(float(insert_shrink_mm), 0.0)
    return 0.0

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
