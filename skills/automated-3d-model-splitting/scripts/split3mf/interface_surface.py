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
from .spatial_intersections import count_polyline_self_intersections_3d

MAX_INTERFACE_EXTENSION_MM = 10.0
MIN_INTERFACE_EXTENSION_MM = 0.2


@dataclass(frozen=True)
class InterfaceSurface:
    vertices: np.ndarray
    faces: np.ndarray
    record: dict


@dataclass(frozen=True)
class PairedInterfaceSurfaces:
    tenon: InterfaceSurface
    mortise: InterfaceSurface


def build_mortise_shell_probe(mortise_shell_triangles: np.ndarray) -> LocalRayProbe:
    triangles = np.asarray(mortise_shell_triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("mortise shell triangles must have shape (M, 3, 3)")
    if not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError("mortise shell has no usable triangles")
    # Both windings make the ray test independent of source-shell face winding
    # and allow open colored patches as collision obstacles.
    return LocalRayProbe(
        SimpleNamespace(
            triangles=np.concatenate((triangles, triangles[:, ::-1, :]))
        )
    )


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


def _curved_cap_triangulation(
    points: np.ndarray,
    preferred_normal: np.ndarray,
) -> tuple[list[tuple[int, int, int]], dict, np.ndarray | None]:
    """Triangulate a curved XYZ cap without treating projection crossings as cuts."""
    values = np.asarray(points, dtype=np.float64)
    center = values.mean(axis=0)
    _left, _singular, right = np.linalg.svd(values - center, full_matrices=False)
    candidates = [
        np.asarray(preferred_normal, dtype=np.float64),
        right[-1], right[0], right[1],
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    tried: set[tuple[float, float, float]] = set()
    best: tuple[int, list[tuple[int, int, int]], np.ndarray, np.ndarray] | None = None
    for candidate in candidates:
        normal = np.asarray(candidate, dtype=np.float64)
        length = float(np.linalg.norm(normal))
        if length <= 1e-10:
            continue
        normal /= length
        key = tuple(np.round(normal, 8).tolist())
        if key in tried:
            continue
        tried.add(key)
        u, v = orthonormal_basis(normal)
        projected = project_points(values, center, u, v)
        crossing_count = int(count_polyline_self_intersections_3d(
            np.column_stack((projected, np.zeros(len(projected))))
        )["segment_intersection_count"])
        if crossing_count:
            if best is None or crossing_count < best[0]:
                best = (crossing_count, [], projected, normal)
            continue
        triangles = triangulate_polygon_ear_clip(projected)
        if len(triangles) == len(values) - 2:
            return triangles, {
                "geometry": "curved_xyz_boundary",
                "triangulation_projection_normal": normal.round(8).tolist(),
                "projection_crossing_count": 0,
                "projection_is_connectivity_only": True,
            }, None
        if best is None:
            best = (0, triangles, projected, normal)
    if best is not None and len(best[1]) == len(values) - 2:
        return best[1], {
            "geometry": "curved_xyz_boundary",
            "triangulation_projection_normal": best[3].round(8).tolist(),
            "projection_crossing_count": int(best[0]),
            "projection_is_connectivity_only": True,
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
        "projection_crossing_count": None if best is None else int(best[0]),
        "projection_is_connectivity_only": True,
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
) -> PairedInterfaceSurfaces:
    """Build a capped annular interface with collision-limited socket depth."""
    outer = np.asarray(boundary_points_mm, dtype=np.float64)
    frozen_outer = outer.copy()
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
    ratio = float(scale_ratio)
    clearance = float(clearance_mm)
    if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError(f"{interface_id}: interface scale ratio must be between 0 and 1")
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError(f"{interface_id}: interface clearance must be finite and non-negative")

    origin = outer.mean(axis=0)
    u, v = orthonormal_basis(tenon_axis)
    outer_2d = project_points(outer, origin, u, v)
    outer_area = signed_area(outer_2d)
    outer_area_valid = abs(outer_area) > 1e-12
    center_2d = _polygon_centroid(outer_2d)
    inner = homothetic_loop_points(outer, tenon_axis, ratio)
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
        "frozen_outer_boundary": count_polyline_self_intersections_3d(frozen_outer),
        "interface_outer_boundary": count_polyline_self_intersections_3d(outer),
        "tenon_inner_ring": count_polyline_self_intersections_3d(inner),
        "mortise_inner_ring": count_polyline_self_intersections_3d(mortise_inner),
        "projection_only": {
            "outer_ring": count_polyline_self_intersections_3d(
                np.column_stack((outer_2d, np.zeros(len(outer_2d))))
            ),
            "tenon_inner_ring": count_polyline_self_intersections_3d(
                np.column_stack((inner_2d, np.zeros(len(inner_2d))))
            ),
            "mortise_inner_ring": count_polyline_self_intersections_3d(
                np.column_stack((mortise_inner_2d, np.zeros(len(mortise_inner_2d))))
            ),
            "policy": "diagnostic_only_never_delete_xyz_separated_arcs",
        },
    }
    mortise_orientation_consistent = (
        np.sign(signed_area(mortise_inner_2d)) == np.sign(outer_area)
    )
    cap_triangles, cap_triangulation, cap_center = _curved_cap_triangulation(
        inner, tenon_axis
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
    extension_depth = MAX_INTERFACE_EXTENSION_MM
    depth_attempts: list[dict] = []
    while True:
        hit_distances, hit_faces = mortise_shell_probe.exits(
            probe_points,
            probe_directions,
            extension_depth + clearance,
            epsilon=1e-7,
        )
        hit_mask = np.isfinite(hit_distances)
        shell_face_count = len(mortise_shell_probe.triangles) // 2
        hit_source_face_ids = np.unique(
            hit_faces[hit_mask] % max(shell_face_count, 1)
        )
        depth_attempts.append({
            "candidate_depth_mm": float(extension_depth),
            "corresponding_mortise_depth_mm": float(extension_depth + clearance),
            "collision": bool(np.any(hit_mask)),
            "hit_count": int(np.count_nonzero(hit_mask)),
            "nearest_hit_mm": (
                float(np.min(hit_distances[hit_mask])) if np.any(hit_mask) else None
            ),
            "hit_face_ids": [int(value) for value in hit_source_face_ids[:16]],
        })
        if not np.any(hit_mask):
            break
        if extension_depth <= MIN_INTERFACE_EXTENSION_MM + 1e-9:
            # Keep the minimum-depth candidate and report its collision. Stage 05
            # is diagnostic-only: geometric quality checks must not cancel the run.
            break
        extension_depth = max(extension_depth * 0.5, MIN_INTERFACE_EXTENSION_MM)

    count = len(outer)
    ring_faces: list[tuple[int, int, int]] = []
    for index in range(count):
        following = (index + 1) % count
        ring_faces.extend((
            (index, following, count + index),
            (following, count + following, count + index),
        ))
    faces = np.asarray(ring_faces, dtype=np.int64).reshape((-1, 3))
    annulus_mesh = trimesh.Trimesh(
        vertices=np.vstack((outer, inner)), faces=faces, process=False
    )
    annulus_edge_counts = np.bincount(
        np.asarray(annulus_mesh.edges_unique_inverse, dtype=np.int64),
        minlength=len(annulus_mesh.edges_unique),
    )
    annulus_double_areas = np.linalg.norm(
        np.cross(
            annulus_mesh.triangles[:, 1] - annulus_mesh.triangles[:, 0],
            annulus_mesh.triangles[:, 2] - annulus_mesh.triangles[:, 0],
        ),
        axis=1,
    )
    audit = SimpleNamespace(
        valid=bool(
            len(faces) == count * 2
            and np.count_nonzero(annulus_edge_counts == 1) == count * 2
            and not np.any(annulus_edge_counts > 2)
        ),
        face_count=int(len(faces)),
        boundary_edge_count=int(np.count_nonzero(annulus_edge_counts == 1)),
        degenerate_face_count=int(np.count_nonzero(annulus_double_areas <= 1e-12)),
        strategy="xyz_corresponding_triangle_strip",
    )
    mortise_projection_audit = audit_projection(
        faces,
        np.vstack((outer_2d, mortise_inner_2d)),
        list(range(count)),
        list(range(count, 2 * count)),
        intersection_points_3d=np.vstack((outer, mortise_inner)),
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
        placed_vertices = np.vstack((outer, inner_base, inner_end))
        cap_center_index = 3 * count
        if cap_center is not None:
            placed_vertices = np.vstack((placed_vertices, inner_end.mean(axis=0)))
        projected_vertices = np.vstack((outer_2d, inner_base_2d))
        projection_audit = audit_projection(
            faces[: len(ring_faces)],
            projected_vertices,
            list(range(count)),
            list(range(count, 2 * count)),
            intersection_points_3d=np.vstack((outer, inner_base)),
        )
        triangles = placed_vertices[faces[: len(ring_faces)]]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        oriented_areas = face_normals @ tenon_axis
        area_epsilon = max(float(np.ptp(outer, axis=0).max()) ** 2 * 1e-14, 1e-14)
        degenerate_band_face_count = int(np.count_nonzero(np.abs(oriented_areas) <= area_epsilon))
        reversed_band_face_count = int(np.count_nonzero(oriented_areas < -area_epsilon))
        side_faces: list[tuple[int, int, int]] = []
        for index in range(count):
            next_index = (index + 1) % count
            first = count + index
            next_first = count + next_index
            first_end = 2 * count + index
            next_end = 2 * count + next_index
            side_faces.extend((
                (first, next_first, next_end),
                (first, next_end, first_end),
            ))
        cap_faces = np.asarray([
            tuple(
                cap_center_index if cap_center is not None and int(value) == count
                else 2 * count + int(value)
                for value in triangle
            )
            for triangle in cap_profile_triangles
        ], dtype=np.int64).reshape((-1, 3))
        complete_faces = np.vstack((faces, np.asarray(side_faces, dtype=np.int64).reshape((-1, 3)), cap_faces))
        collision_mesh = trimesh.Trimesh(
            vertices=placed_vertices,
            faces=complete_faces,
            process=False,
        )
        trimesh.repair.fix_winding(collision_mesh)
        edge_counts = np.bincount(
            np.asarray(collision_mesh.edges_unique_inverse, dtype=np.int64),
            minlength=len(collision_mesh.edges_unique),
        )
        unique_edges = np.asarray(collision_mesh.edges_unique, dtype=np.int64)
        open_edges = {
            tuple(sorted(map(int, edge)))
            for edge in unique_edges[edge_counts == 1]
        }
        expected_outer_edges = {
            tuple(sorted((index, (index + 1) % count)))
            for index in range(count)
        }
        over_shared_edge_count = int(np.count_nonzero(edge_counts > 2))
        closure_valid = (
            open_edges == expected_outer_edges
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
                "extension_depth_mm": float(surface_depth),
                "tenon_extension_depth_mm": float(extension_depth),
                "mortise_recess_depth_mm": float(extension_depth + clearance),
                "additional_mortise_depth_mm": float(clearance),
                "extension_depth_limit_mm": float(MAX_INTERFACE_EXTENSION_MM),
                "extension_depth_minimum_mm": float(MIN_INTERFACE_EXTENSION_MM),
                "extension_depth_attempts": depth_attempts,
                "minimum_depth_collision_remaining": bool(
                    depth_attempts
                    and depth_attempts[-1]["collision"]
                    and extension_depth <= MIN_INTERFACE_EXTENSION_MM + 1e-9
                ),
                "inner_cap_face_count": int(len(cap_faces)),
                "annular_band_face_count": int(len(ring_faces)),
                "inner_wall_face_count": int(len(side_faces)),
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
                    "mortise_projection": {
                        "valid": bool(mortise_projection_audit.valid),
                        "crossing_edges": mortise_projection_audit.crossing_edges,
                        "reason": mortise_projection_audit.reason,
                    },
                    "inner_cap_closed": bool(len(cap_faces) > 0),
                    "curved_cap_geometry": cap_triangulation["geometry"],
                    "inner_band_cap_closure_valid": bool(closure_valid),
                    "closure_diagnostics_only": True,
                    "open_outer_interface_edges": int(len(open_edges)),
                    "over_shared_edges": over_shared_edge_count,
                    "winding_consistent": bool(collision_mesh.is_winding_consistent),
                    "degenerate_face_count": degenerate_face_count,
                    "outer_interface_boundary_edge_count": int(count),
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
