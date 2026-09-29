from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import trimesh

from .annulus_projection import audit_projection
from .local_ray_probe import LocalRayProbe
from .mesh import (
    homothetic_loop_points,
    orthonormal_basis,
    point_in_poly,
    project_points,
    radial_offset_points,
    signed_area,
    triangulate_polygon_ear_clip,
)
from .spatial_intersections import (
    first_nonincident_triangle_intersection_3d,
)

MAX_INTERFACE_EXTENSION_MM = 5.0
MIN_INTERFACE_EXTENSION_MM = 0.2
# Disjoint distance bands, each measured against every probe ray.
# Contact at or below 0.1 mm remains ignored; equality at a ratio is accepted.
MORTISE_COLLISION_DISTANCE_THRESHOLD_MM = 0.1
MORTISE_COLLISION_RATIO_THRESHOLD = 0.40
MORTISE_FAR_COLLISION_DISTANCE_MM = 1.0
MORTISE_FAR_COLLISION_RATIO_THRESHOLD = 0.05
MORTISE_RAY_NUMERICAL_EPSILON_MM = 1e-7


@dataclass(frozen=True)
class InterfaceSurface:
    vertices: np.ndarray
    faces: np.ndarray
    record: dict


@dataclass(frozen=True)
class PairedInterfaceSurfaces:
    tenon: InterfaceSurface
    mortise: InterfaceSurface


def _mortise_collision_statistics(hit_distances: np.ndarray) -> dict:
    """Summarize blocked construction probes using the configured thresholds."""
    distances = np.asarray(hit_distances, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(distances)
    blocking_mask = finite_mask & (
        distances > MORTISE_COLLISION_DISTANCE_THRESHOLD_MM
    )
    probe_count = int(len(distances))
    blocking_count = int(np.count_nonzero(blocking_mask))
    raw_hit_count = int(np.count_nonzero(finite_mask))
    ratio = blocking_count / probe_count if probe_count else 0.0
    hit_ratio = blocking_count / raw_hit_count if raw_hit_count else 0.0
    blocking_distances = distances[blocking_mask]
    bands = {}
    for name, mask, threshold in (
        ("short", blocking_mask & (distances <= MORTISE_FAR_COLLISION_DISTANCE_MM),
         MORTISE_COLLISION_RATIO_THRESHOLD),
        ("far", blocking_mask & (distances > MORTISE_FAR_COLLISION_DISTANCE_MM),
         MORTISE_FAR_COLLISION_RATIO_THRESHOLD),
    ):
        count = int(np.count_nonzero(mask))
        band_ratio = count / probe_count if probe_count else 0.0
        bands[name] = {
            "hit_count": count, "ratio": float(band_ratio),
            "ratio_threshold": threshold, "collision": band_ratio > threshold,
        }
    return {
        "probe_count": probe_count,
        "raw_hit_count": raw_hit_count,
        "near_surface_hit_count": int(
            np.count_nonzero(finite_mask & ~blocking_mask)
        ),
        "blocking_hit_count": blocking_count,
        "blocking_ratio": float(ratio),
        "blocking_ratio_of_hit_rays": float(hit_ratio),
        "nearest_blocking_hit_mm": (
            float(np.min(blocking_distances)) if blocking_count else None
        ),
        "distance_bands": bands,
        "collision": any(band["collision"] for band in bands.values()),
    }


def build_mortise_shell_probe(mortise_shell_triangles: np.ndarray) -> LocalRayProbe:
    triangles = np.asarray(mortise_shell_triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("mortise shell triangles must have shape (M, 3, 3)")
    if not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError("mortise shell has no usable triangles")
    # LocalRayProbe.exits deliberately accepts only outward-facing crossings.
    # Duplicating the reverse winding turns a normal entry into a false exit:
    # an insert is expected to enter the parent before reaching its far wall.
    return LocalRayProbe(SimpleNamespace(triangles=triangles))


def _polygon_centroid(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    cross = (
        points[:, 0] * np.roll(points[:, 1], -1)
        - np.roll(points[:, 0], -1) * points[:, 1]
    )
    twice_area = float(cross.sum())
    if abs(twice_area) <= 1e-12:
        # Degenerate contours still get a deterministic best-effort center;
        # the zero-area condition is retained in the quality diagnostics.
        return points.mean(axis=0)
    return np.array(
        [
            np.sum((points[:, 0] + np.roll(points[:, 0], -1)) * cross),
            np.sum((points[:, 1] + np.roll(points[:, 1], -1)) * cross),
        ],
        dtype=np.float64,
    ) / (3.0 * twice_area)


def _simple_planar_inner_contour(
    desired_xy: np.ndarray, outer_xyz: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Build an ellipse in the plane normal to the tenon insertion axis."""
    desired = np.asarray(desired_xy, dtype=np.float64)
    projected_ring = np.column_stack((desired, np.zeros(len(desired))))
    center = desired.mean(axis=0)
    covariance = np.cov((desired - center).T, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues[::-1], 1e-8)
    axes = eigenvectors[:, ::-1]
    if np.linalg.det(axes) < 0.0:
        axes[:, 1] *= -1.0
    radii = np.sqrt(2.0 * eigenvalues)
    relative_first = (desired[0] - center) @ axes
    phase = float(np.arctan2(
        relative_first[1] / radii[1], relative_first[0] / radii[0]
    ))
    edges = np.maximum(np.linalg.norm(
        np.roll(outer_xyz, -1, axis=0) - outer_xyz, axis=1
    ), 1e-9)
    progress = np.concatenate(([0.0], np.cumsum(edges[:-1]))) / edges.sum()
    orientation = 1.0 if signed_area(desired) >= 0.0 else -1.0
    base_angles = phase + orientation * (2.0 * np.pi * progress)
    # A single first-point match can twist a long concave boundary around the
    # ellipse. Pick one cyclic phase for the whole ring by minimizing total
    # outer-to-inner travel; the plane normal and insertion axis stay fixed.
    phase_candidates = np.linspace(-np.pi, np.pi, 65)[:-1]
    errors = []
    for offset in phase_candidates:
        trial = center + np.column_stack((
            radii[0] * np.cos(base_angles + offset),
            radii[1] * np.sin(base_angles + offset),
        )) @ axes.T
        errors.append(float(np.mean(np.sum((trial - desired) ** 2, axis=1))))
    phase_offset = float(phase_candidates[int(np.argmin(errors))])
    angles = base_angles + phase_offset
    repaired = center + np.column_stack((
        radii[0] * np.cos(angles), radii[1] * np.sin(angles)
    )) @ axes.T
    return repaired, {
        "strategy": "arclength_parameterized_planar_ellipse",
        "original_projection_crossings": None,
        "projection_intersection_scan_performed": False,
        "ellipse_phase_offset_rad": phase_offset,
        "ellipse_mean_squared_xy_adjustment_mm2": min(errors),
        "max_xy_adjustment_mm": float(np.max(np.linalg.norm(repaired - desired, axis=1))),
    }


def densify_closed_contour(points: np.ndarray, subdivisions: int = 3) -> np.ndarray:
    """Add smooth periodic Catmull–Rom samples while retaining each Stage 04 anchor."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        raise ValueError("closed contour needs at least three XYZ points")
    steps = max(1, int(subdivisions))
    result = []
    count = len(values)
    for index in range(count):
        p0 = values[(index - 1) % count]
        p1 = values[index]
        p2 = values[(index + 1) % count]
        p3 = values[(index + 2) % count]
        for sample in range(steps):
            t = sample / steps
            t2, t3 = t * t, t * t * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            result.append(point)
    return np.asarray(result, dtype=np.float64)


def _curved_cap_triangulation(
    points: np.ndarray,
    preferred_normal: np.ndarray,
) -> tuple[list[tuple[int, int, int]], dict, np.ndarray | None]:
    """Try one mating-axis projection for a curved cap, without projection search."""
    values = np.asarray(points, dtype=np.float64)
    center = values.mean(axis=0)
    normal = np.asarray(preferred_normal, dtype=np.float64)
    length = float(np.linalg.norm(normal))
    if length > 1e-10:
        normal /= length
        u, v = orthonormal_basis(normal)
        projected = project_points(values, center, u, v)
        triangles = triangulate_polygon_ear_clip(projected)
        if len(triangles) == len(values) - 2:
            return triangles, {
                "geometry": "curved_xyz_boundary",
                "triangulation_projection_normal": normal.round(8).tolist(),
                "projection_crossing_count": None,
                "projection_search_performed": False,
            }, None
    # A projection-independent curved fan always preserves every boundary
    # edge. Its final local index denotes the added centre vertex.
    center_vertex = values.mean(axis=0)
    fan = [
        (index, (index + 1) % len(values), len(values))
        for index in range(len(values))
    ]
    return fan, {
        "geometry": "curved_xyz_boundary",
        "triangulation_projection_normal": None,
        "projection_crossing_count": None,
        "projection_search_performed": False,
        "reason": "single_axis_projection_did_not_triangulate_full_contour",
        "strategy": "projection_independent_curved_center_fan",
    }, center_vertex


def build_pairwise_interface_surfaces(
    *,
    boundary_points_mm: np.ndarray,
    insertion_direction: np.ndarray,
    socket_inward_direction: np.ndarray,
    scale_ratio: float,
    clearance_mm: float,
    mortise_shell_probe: LocalRayProbe,
    interface_id: str,
    extension_depth_limit_mm: float = MAX_INTERFACE_EXTENSION_MM,
) -> PairedInterfaceSurfaces:
    """Build a capped annular interface with collision-limited socket depth."""
    outer = np.asarray(boundary_points_mm, dtype=np.float64)
    tenon_axis = np.asarray(insertion_direction, dtype=np.float64)
    mortise_axis = np.asarray(socket_inward_direction, dtype=np.float64)
    if outer.ndim != 2 or outer.shape[1] != 3 or len(outer) < 3:
        raise ValueError(f"{interface_id}: interface loop needs at least three 3D points")
    if (
        not np.isfinite(outer).all()
        or tenon_axis.shape != (3,)
        or mortise_axis.shape != (3,)
        or not np.isfinite(tenon_axis).all()
        or not np.isfinite(mortise_axis).all()
    ):
        raise ValueError(f"{interface_id}: interface points and direction must be finite")
    tenon_axis_length = float(np.linalg.norm(tenon_axis))
    mortise_axis_length = float(np.linalg.norm(mortise_axis))
    if tenon_axis_length <= 1e-10 or mortise_axis_length <= 1e-10:
        raise ValueError(f"{interface_id}: Stage 04 mating direction is degenerate")
    tenon_axis = tenon_axis / tenon_axis_length
    mortise_axis = mortise_axis / mortise_axis_length
    # Stage 04 identifies the tenon side and provides the approved mating axis.
    # The contour normal is only diagnostic: replacing the mating axis with it
    # can turn the tenon sideways when the painted boundary is strongly curved.
    centered_outer = outer - outer.mean(axis=0)
    boundary_normal = np.cross(
        centered_outer, np.roll(centered_outer, -1, axis=0)
    ).sum(axis=0)
    boundary_normal_length = float(np.linalg.norm(boundary_normal))
    if boundary_normal_length <= 1e-10:
        raise ValueError(f"{interface_id}: simplified boundary has no stable area normal")
    boundary_normal /= boundary_normal_length
    if float(np.dot(boundary_normal, tenon_axis)) < 0.0:
        boundary_normal = -boundary_normal
    stage04_axis = tenon_axis.copy()
    stage04_socket_axis = mortise_axis.copy()
    tenon_axis = stage04_axis.copy()
    mortise_axis = stage04_axis.copy()
    ratio = float(scale_ratio)
    clearance = float(clearance_mm)
    if not np.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError(f"{interface_id}: interface scale ratio must be between 0 and 1 inclusive")
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError(f"{interface_id}: interface clearance must be finite and non-negative")

    origin = outer.mean(axis=0)
    u, v = orthonormal_basis(tenon_axis)
    outer_2d = project_points(outer, origin, u, v)
    outer_area = signed_area(outer_2d)
    outer_area_valid = abs(outer_area) > 1e-12
    center_2d = _polygon_centroid(outer_2d)
    # The Stage 04 boundary may be curved in XYZ. Build the inset in its
    # interface plane before extending it, so the inner wall follows the
    # Stage 04 mating axis and every terminal cap is planar.
    # The inner profile is elliptical; no percentage contour or automatic
    # fallback can silently replace this selected geometry.
    contour_attempts = []
    candidate_specs = [ratio]
    selected = None
    for candidate_ratio in candidate_specs:
        candidate = homothetic_loop_points(outer, tenon_axis, candidate_ratio)
        desired_xy = project_points(candidate, origin, u, v)
        try:
            repaired_xy, contour_record = _simple_planar_inner_contour(desired_xy, outer)
        except ValueError as exc:
            contour_attempts.append({
                "scale_ratio": candidate_ratio, "ellipse": True,
                "accepted": False, "reason": str(exc),
            })
            continue
        candidate += (
            (repaired_xy[:, 0] - desired_xy[:, 0])[:, None] * u
            + (repaired_xy[:, 1] - desired_xy[:, 1])[:, None] * v
        )
        candidate_axial = candidate @ mortise_axis
        # A curved source boundary may reach farther into the host than its
        # mean plane. Anchor the ellipse beyond that deepest boundary point,
        # so the connection never folds backward along the insertion axis.
        base_plane = float(np.max(outer @ mortise_axis)) + MIN_INTERFACE_EXTENSION_MM
        candidate += (base_plane - candidate_axial)[:, None] * mortise_axis
        candidate_mortise = radial_offset_points(
            candidate, origin, u, v, tenon_axis, clearance
        )
        try:
            if contour_record["strategy"] == "arclength_parameterized_planar_ellipse":
                candidate_cap = [
                    (index, (index + 1) % len(candidate), len(candidate))
                    for index in range(len(candidate))
                ]
                candidate_center = candidate.mean(axis=0)
            else:
                candidate_cap, _, candidate_center = _curved_cap_triangulation(
                    candidate, tenon_axis
                )
        except ValueError as exc:
            contour_attempts.append({
                "scale_ratio": candidate_ratio, "ellipse": True,
                "accepted": False, "reason": str(exc),
            })
            continue
        candidate_cap_points = (
            np.vstack((candidate_mortise, candidate_mortise.mean(axis=0)))
            if candidate_center is not None else candidate_mortise
        )
        samples = np.vstack((
            candidate,
            *(
                candidate_mortise * (1.0 - fraction)
                + np.roll(candidate_mortise, -1, axis=0) * fraction
                for fraction in (0.25, 0.5, 0.75)
            ),
            np.asarray([
                candidate_cap_points[np.asarray(face, dtype=np.int64)].mean(axis=0)
                for face in candidate_cap
            ]),
            np.asarray([
                (candidate_cap_points[int(left)] + candidate_cap_points[int(right)]) * 0.5
                for face in candidate_cap
                for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
            ]),
        ))
        min_hits, _ = mortise_shell_probe.exits(
            samples, np.broadcast_to(mortise_axis, samples.shape).copy(),
            MIN_INTERFACE_EXTENSION_MM + clearance,
            epsilon=MORTISE_RAY_NUMERICAL_EPSILON_MM,
        )
        thickness = _mortise_collision_statistics(min_hits)
        accepted = not thickness["collision"]
        contour_attempts.append({
            "scale_ratio": candidate_ratio, "ellipse": True,
            "accepted": accepted, "strategy": contour_record["strategy"],
            "minimum_depth_thickness_probe": thickness,
        })
        if accepted:
            selected = (candidate_ratio, candidate, contour_record)
            break
    if selected is None:
        raise ValueError(f"{interface_id}: elliptical inner contour lacks adequate host thickness: {contour_attempts}")
    ratio, inner, inner_contour_record = selected
    inner_2d = project_points(inner, origin, u, v)
    inner_orientation_consistent = (
        np.sign(signed_area(inner_2d)) == np.sign(outer_area)
    )
    inner_inside_outer = bool(point_in_poly(inner_2d[0], outer_2d))

    mortise_inner = radial_offset_points(
        inner, origin, u, v, tenon_axis, clearance
    )
    mortise_inner_2d = project_points(mortise_inner, origin, u, v)
    intersection_diagnostics_3d = {
        "coordinate_space": "world_xyz_mm",
        "self_intersection_scans_performed": False,
        "policy": "skipped_for_ellipse_candidate_by_user_request",
    }
    mortise_orientation_consistent = (
        np.sign(signed_area(mortise_inner_2d)) == np.sign(outer_area)
    )
    if inner_contour_record["strategy"] == "arclength_parameterized_planar_ellipse":
        cap_triangles = [
            (index, (index + 1) % len(inner), len(inner))
            for index in range(len(inner))
        ]
        cap_center = inner.mean(axis=0)
        cap_triangulation = {
            "geometry": "planar_convex_inner_ring",
            "strategy": "convex_center_fan",
            "projection_crossing_count": 0,
        }
    else:
        cap_triangles, cap_triangulation, cap_center = _curved_cap_triangulation(
            inner, tenon_axis
        )
    outer_cap_triangles, outer_cap_triangulation, outer_cap_center = (
        _curved_cap_triangulation(outer, tenon_axis)
    )
    mortise_cap_points = (
        np.vstack((mortise_inner, mortise_inner.mean(axis=0)))
        if cap_center is not None
        else mortise_inner
    )
    cap_points = np.asarray(
        [
            mortise_cap_points[np.asarray(face, dtype=np.int64)].mean(axis=0)
            for face in cap_triangles
        ],
        dtype=np.float64,
    ).reshape((-1, 3))
    cap_edge_points = np.asarray(
        [
                (mortise_cap_points[int(left)] + mortise_cap_points[int(right)]) * 0.5
            for triangle in cap_triangles
        for left, right in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            )
        ],
        dtype=np.float64,
    ).reshape((-1, 3))
    edge_samples = np.concatenate(
        [
            mortise_inner * (1.0 - fraction)
            + np.roll(mortise_inner, -1, axis=0) * fraction
            for fraction in (0.25, 0.5, 0.75)
        ],
        axis=0,
    )
    probe_points = np.vstack((inner, edge_samples, cap_points, cap_edge_points))
    probe_directions = np.broadcast_to(mortise_axis, probe_points.shape).copy()
    depth_limit = min(MAX_INTERFACE_EXTENSION_MM, float(extension_depth_limit_mm))
    if not np.isfinite(depth_limit) or depth_limit < MIN_INTERFACE_EXTENSION_MM:
        raise ValueError(f"{interface_id}: mortise shell has less than {MIN_INTERFACE_EXTENSION_MM} mm available depth")
    extension_depth = depth_limit
    depth_attempts: list[dict] = []
    while True:
        hit_distances, hit_faces = mortise_shell_probe.exits(
            probe_points,
            probe_directions,
            extension_depth + clearance,
            epsilon=MORTISE_RAY_NUMERICAL_EPSILON_MM,
        )
        collision_stats = _mortise_collision_statistics(hit_distances)
        hit_mask = np.isfinite(hit_distances)
        blocking_mask = hit_mask & (
            hit_distances > MORTISE_COLLISION_DISTANCE_THRESHOLD_MM
        )
        hit_source_face_ids = np.unique(hit_faces[hit_mask])
        depth_attempts.append({
            "candidate_depth_mm": float(extension_depth),
            "corresponding_mortise_depth_mm": float(extension_depth + clearance),
            **collision_stats,
            "hit_face_ids": [int(value) for value in hit_source_face_ids[:16]],
        })
        if not collision_stats["collision"]:
            break
        if extension_depth <= MIN_INTERFACE_EXTENSION_MM + 1e-9:
            nearest_hit = float(np.min(hit_distances[blocking_mask]))
            nearest_hit_index = int(
                np.flatnonzero(blocking_mask)[np.argmin(hit_distances[blocking_mask])]
            )
            hit_records = []
            for probe_index in np.flatnonzero(blocking_mask)[:8]:
                face_index = int(hit_faces[probe_index])
                triangle = mortise_shell_probe.triangles[face_index]
                normal = np.cross(
                    triangle[1] - triangle[0], triangle[2] - triangle[0]
                )
                normal_length = float(np.linalg.norm(normal))
                if normal_length > 1e-12:
                    normal /= normal_length
                hit_records.append({
                    "probe_index": int(probe_index),
                    "probe_point_mm": probe_points[probe_index].round(6).tolist(),
                    "exit_mm": round(float(hit_distances[probe_index]), 6),
                    "host_face": face_index,
                    "host_normal_dot_axis": round(float(normal @ mortise_axis), 6),
                })
            raise ValueError(
                f"{interface_id}: mortise shell intersects the inner ring at the "
                f"minimum {MIN_INTERFACE_EXTENSION_MM} mm tenon depth "
                f"(blocking_hits={collision_stats['blocking_hit_count']}/"
                f"{collision_stats['probe_count']} total probe rays "
                f"({collision_stats['raw_hit_count']} rays hit any shell surface), "
                f"blocking_ratio={collision_stats['blocking_ratio']:.1%}, "
                f"distance_bands={collision_stats['distance_bands']}, "
                f"distance_threshold_mm={MORTISE_COLLISION_DISTANCE_THRESHOLD_MM:.3f}, "
                f"nearest_exit_mm={nearest_hit:.6f}, "
                f"probe_index={nearest_hit_index}, hit_samples={hit_records})"
            )
        extension_depth = max(extension_depth * 0.5, MIN_INTERFACE_EXTENSION_MM)

    count = len(outer)
    connection_faces: list[tuple[int, int, int]] = []
    for index in range(count):
        following = (index + 1) % count
        connection_faces.extend((
            (index, following, count + index),
            (following, count + following, count + index),
        ))
    faces = np.asarray(connection_faces, dtype=np.int64).reshape((-1, 3))
    mortise_projection_audit = audit_projection(
        faces,
        np.vstack((outer_2d, mortise_inner_2d)),
        list(range(count)),
        list(range(count, 2 * count)),
    )
    def make_surface(
        side: str,
        side_axis: np.ndarray,
        inner_base: np.ndarray,
        surface_depth: float,
        cap_profile_triangles: list[tuple[int, int, int]],
        sign: float,
    ) -> InterfaceSurface:
        inner_base_2d = project_points(inner_base, origin, u, v)
        inner_end = inner_base + mortise_axis * surface_depth
        # One tapered side joins the frozen outer ring directly to the
        # extended inner ring.  There is no intermediate annulus or wall.
        placed_vertices = np.vstack((outer, inner_end))
        cap_center_index = 2 * count
        if cap_center is not None:
            placed_vertices = np.vstack((placed_vertices, inner_end.mean(axis=0)))
        outer_cap_center_index = len(placed_vertices)
        if outer_cap_center is not None:
            placed_vertices = np.vstack((placed_vertices, outer_cap_center))
        projected_vertices = np.vstack((outer_2d, inner_base_2d))
        projection_audit = audit_projection(
            faces,
            projected_vertices,
            list(range(count)),
            list(range(count, 2 * count)),
        )
        side_mesh = trimesh.Trimesh(
            vertices=placed_vertices, faces=faces, process=False
        )
        side_edge_counts = np.bincount(
            np.asarray(side_mesh.edges_unique_inverse, dtype=np.int64),
            minlength=len(side_mesh.edges_unique),
        )
        triangles = placed_vertices[faces]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        double_areas = np.linalg.norm(face_normals, axis=1)
        audit = SimpleNamespace(
            valid=bool(
                len(faces) == count * 2
                and np.count_nonzero(side_edge_counts == 1) == count * 2
                and not np.any(side_edge_counts > 2)
            ),
            face_count=int(len(faces)),
            boundary_edge_count=int(np.count_nonzero(side_edge_counts == 1)),
            degenerate_face_count=int(np.count_nonzero(double_areas <= 1e-12)),
            strategy="xyz_direct_outer_to_extended_inner_strip",
        )
        oriented_areas = face_normals @ tenon_axis
        area_epsilon = max(float(np.ptp(outer, axis=0).max()) ** 2 * 1e-14, 1e-14)
        degenerate_band_face_count = int(np.count_nonzero(np.abs(oriented_areas) <= area_epsilon))
        reversed_band_face_count = int(np.count_nonzero(oriented_areas < -area_epsilon))
        cap_faces = np.asarray([
            tuple(
                cap_center_index if cap_center is not None and int(value) == count
                else count + int(value)
                for value in triangle
            )
            for triangle in cap_profile_triangles
        ], dtype=np.int64).reshape((-1, 3))
        outer_cap_faces = np.asarray([
            tuple(
                outer_cap_center_index if outer_cap_center is not None and int(value) == count
                else int(value)
                for value in triangle
            )
            for triangle in outer_cap_triangles
        ], dtype=np.int64).reshape((-1, 3))
        complete_faces = np.vstack((
            faces,
            cap_faces,
            outer_cap_faces,
        ))
        collision_mesh = trimesh.Trimesh(
            vertices=placed_vertices,
            faces=complete_faces,
            process=False,
        )
        trimesh.repair.fix_winding(collision_mesh)
        trimesh.repair.fix_inversion(collision_mesh, multibody=True)
        edge_counts = np.bincount(
            np.asarray(collision_mesh.edges_unique_inverse, dtype=np.int64),
            minlength=len(collision_mesh.edges_unique),
        )
        unique_edges = np.asarray(collision_mesh.edges_unique, dtype=np.int64)
        open_edges = {
            tuple(sorted(map(int, edge)))
            for edge in unique_edges[edge_counts == 1]
        }
        over_shared_edge_count = int(np.count_nonzero(edge_counts > 2))
        closure_valid = (
            not open_edges
            and over_shared_edge_count == 0
            and bool(collision_mesh.is_winding_consistent)
        )
        complete_triangles = placed_vertices[np.asarray(collision_mesh.faces, dtype=np.int64)]
        complete_double_areas = np.linalg.norm(
            np.cross(
                complete_triangles[:, 1] - complete_triangles[:, 0],
                complete_triangles[:, 2] - complete_triangles[:, 0],
            ),
            axis=1,
        )
        degenerate_face_count = int(np.count_nonzero(complete_double_areas <= 1e-12))
        # Self-intersection scanning of the ellipse candidate is intentionally
        # skipped; the user requested direct acceptance after ellipse creation.
        # Closure, winding, degeneracy, thickness and package checks still run.
        if inner_contour_record.get("strategy") == "arclength_parameterized_planar_ellipse":
            xyz_audit = {
                "valid": None,
                "performed": False,
                "reason": "skipped_for_ellipse_candidate_by_user_request",
            }
        else:
            xyz_audit = first_nonincident_triangle_intersection_3d(
                np.vstack((faces, cap_faces)), placed_vertices
            )
        ellipse_self_intersection_allowed = (
            inner_contour_record.get("strategy")
            == "arclength_parameterized_planar_ellipse"
            and xyz_audit.get("performed") is False
        )
        if (
            not closure_valid
            or degenerate_face_count
            or (not xyz_audit["valid"] and not ellipse_self_intersection_allowed)
        ):
            raise ValueError(
                f"{interface_id} {side}: 3D interface validation failed; "
                f"projection_diagnostic={projection_audit.reason or mortise_projection_audit.reason}, "
                f"closure={closure_valid}, degenerate_faces={degenerate_face_count}, "
                f"xyz_audit={xyz_audit}, strip_face_count={len(faces)}, "
                f"inner_cap_face_count={len(cap_faces)}, "
                f"outer_axis_range_mm={[float(np.min(outer @ mortise_axis)), float(np.max(outer @ mortise_axis))]}, "
                f"inner_axis_mm={float(np.mean(inner_end @ mortise_axis))}, "
                f"surface_depth_mm={surface_depth}"
            )
        return InterfaceSurface(
            vertices=placed_vertices,
            faces=np.asarray(collision_mesh.faces, dtype=np.int64),
            record={
                "interface_id": interface_id,
                "side": side,
                "geometry_role": (
                    "additive_tenon_surface"
                    if side == "tenon"
                    else "subtractive_mortise_surface"
                ),
                "boundary_vertex_count": count,
                "scale_ratio": ratio,
                "inner_planar_contour": inner_contour_record,
                "ellipse_plane_normal": tenon_axis.round(8).tolist(),
                "ellipse_extrusion_axis": mortise_axis.round(8).tolist(),
                "inner_contour_attempts": contour_attempts,
                "clearance_mm": clearance,
                "inner_ring_offset_mm": float(sign * clearance),
                "quality_gates_blocking": False,
                "quality_diagnostics": {
                    "outer_projected_area_valid": bool(outer_area_valid),
                    "inner_orientation_consistent": bool(inner_orientation_consistent),
                    "inner_first_vertex_inside_outer": bool(inner_inside_outer),
                    "mortise_offset_orientation_consistent": bool(mortise_orientation_consistent),
                    "depth_collision_at_minimum": bool(
                        depth_attempts
                        and depth_attempts[-1]["collision"]
                        and extension_depth <= MIN_INTERFACE_EXTENSION_MM + 1e-9
                    ),
                    "self_intersections_3d": intersection_diagnostics_3d,
                    "curved_cap_triangulation": cap_triangulation,
                },
                "outer_boundary_shift_mm": 0.0,
                "clearance_direction": "mortise_only_outward_profile_and_depth",
                "extension_direction": mortise_axis.round(8).tolist(),
                "stage04_mating_direction": stage04_axis.round(8).tolist(),
                "stage04_socket_inward_direction": stage04_socket_axis.round(8).tolist(),
                "stage04_to_boundary_normal_angle_deg": float(
                    np.degrees(np.arccos(np.clip(
                        np.dot(stage04_axis, boundary_normal), -1.0, 1.0
                    )))
                ),
                "stage04_to_extension_angle_deg": float(np.degrees(np.arccos(np.clip(
                    np.dot(stage04_axis, mortise_axis), -1.0, 1.0
                )))),
                "extension_depth_mm": float(surface_depth),
                "tenon_extension_depth_mm": float(extension_depth),
                "mortise_recess_depth_mm": float(extension_depth + clearance),
                "additional_mortise_depth_mm": float(clearance),
                "extension_depth_limit_mm": float(depth_limit),
                "extension_depth_minimum_mm": float(MIN_INTERFACE_EXTENSION_MM),
                "extension_depth_attempts": depth_attempts,
                "minimum_depth_collision_remaining": bool(
                    depth_attempts
                    and depth_attempts[-1]["collision"]
                    and extension_depth <= MIN_INTERFACE_EXTENSION_MM + 1e-9
                ),
                "inner_cap_face_count": int(len(cap_faces)),
                "inner_cap_geometry": "plane",
                "inner_cap_max_plane_error_mm": float(np.ptp(inner_end @ mortise_axis)),
                "inner_ring_max_travel_mm": float(np.max((inner_end - inner_base) @ mortise_axis)),
                "outer_boundary_cap_face_count": int(len(outer_cap_faces)),
                "outer_boundary_cap_triangulation": outer_cap_triangulation,
                "outer_boundary_closed_from_frozen_simplified_loop": True,
                "outer_to_extended_inner_face_count": int(len(connection_faces)),
                "stage04_side_direction": side_axis.round(8).tolist(),
                "center_mm": (origin + center_2d[0] * u + center_2d[1] * v).round(8).tolist(),
                "outer_projected_area_mm2": abs(float(outer_area)),
                "inner_projected_area_mm2": abs(float(signed_area(inner_base_2d))),
                "topology_audit": {
                    "valid": bool(audit.valid),
                    "face_count": int(audit.face_count),
                    "boundary_edge_count": int(audit.boundary_edge_count),
                    "degenerate_face_count": int(audit.degenerate_face_count),
                    "quality_gates_blocking": False,
                    "mortise_offset_orientation_consistent": bool(mortise_orientation_consistent),
                    "band_degenerate_face_count": degenerate_band_face_count,
                    "band_reversed_face_count": reversed_band_face_count,
                    "strategy": str(audit.strategy),
                    "clearance_projection": {
                        "valid": bool(projection_audit.valid),
                        "crossing_edges": projection_audit.crossing_edges,
                        "reason": projection_audit.reason,
                    },
                    "projection_intersection_scan_performed": False,
                    "three_dimensional_self_intersection": xyz_audit,
                    "three_dimensional_self_intersection_allowed": bool(
                        ellipse_self_intersection_allowed
                    ),
                    "self_intersection_scan_policy": (
                        "skipped_for_ellipse_candidate_by_user_request"
                        if ellipse_self_intersection_allowed
                        else "performed_and_blocking"
                    ),
                    "three_dimensional_self_intersection_scan_performed": bool(
                        xyz_audit.get("performed", True)
                    ),
                    "mortise_projection": {
                        "valid": bool(mortise_projection_audit.valid),
                        "crossing_edges": mortise_projection_audit.crossing_edges,
                        "reason": mortise_projection_audit.reason,
                    },
                    "inner_cap_closed": bool(len(cap_faces) > 0),
                    "curved_cap_geometry": cap_triangulation["geometry"],
                    "direct_connection_cap_closure_valid": bool(closure_valid),
                    "closure_diagnostics_only": True,
                    "open_outer_interface_edges": int(len(open_edges)),
                    "over_shared_edges": over_shared_edge_count,
                    "winding_consistent": bool(collision_mesh.is_winding_consistent),
                    "degenerate_face_count": degenerate_face_count,
                    "outer_interface_boundary_edge_count": int(count),
                    "outer_boundary_capped": True,
                },
            },
        )

    return PairedInterfaceSurfaces(
        tenon=make_surface(
            "tenon", tenon_axis, inner, extension_depth, cap_triangles, 0.0
        ),
        mortise=make_surface(
            "mortise",
            mortise_axis,
            mortise_inner,
            extension_depth + clearance,
            cap_triangles,
            1.0,
        ),
    )
