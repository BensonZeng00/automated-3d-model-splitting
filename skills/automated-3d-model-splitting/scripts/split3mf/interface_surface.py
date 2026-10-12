"""Shared elliptical profiles for tapered and curved-shoulder interfaces."""
from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import trimesh

from .interface_shape import classify_interface_shape
from .interface_profile import _simple_planar_inner_contour
from .local_ray_probe import LocalRayProbe
from .mesh import (
    orthonormal_basis,
    homothetic_loop_points,
    project_points,
    radial_offset_points,
)

MAX_INTERFACE_EXTENSION_MM = 5.0
MIN_INTERFACE_EXTENSION_MM = 0.2
INTERFACE_WALL_RESERVE_MM = 0.05
MORTISE_RAY_NUMERICAL_EPSILON_MM = 1e-7


class InterfaceDepthError(ValueError):
    """An unsafe connector must not silently fall back to another shape."""


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
    triangles = np.asarray(mortise_shell_triangles, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("mortise shell triangles must have shape (M, 3, 3)")
    if not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError("mortise shell has no usable triangles")
    return LocalRayProbe(SimpleNamespace(triangles=triangles))


def _surface(vertices, faces, record):
    """Validate only the generated patch; leave source geometry untouched."""
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    trimesh.repair.fix_winding(mesh)
    n = record["boundary_vertex_count"]
    counts = np.bincount(mesh.edges_unique_inverse)
    open_edges = {tuple(edge) for edge in mesh.edges_unique[counts == 1]}
    expected = {tuple(sorted((i, (i+1) % n))) for i in range(n)}
    areas = np.linalg.norm(np.cross(
        mesh.triangles[:, 1]-mesh.triangles[:, 0],
        mesh.triangles[:, 2]-mesh.triangles[:, 0]), axis=1)
    if (open_edges != expected or np.any(counts > 2)
            or not mesh.is_winding_consistent or np.any(areas <= 1e-12)):
        raise ValueError(f"{record['interface_id']}: generated patch topology invalid")
    record["topology_audit"] = {
        "valid": True, "open_outer_interface_edges": n,
        "over_shared_edges": 0, "winding_consistent": True,
        "degenerate_face_count": 0,
    }
    return InterfaceSurface(vertices, np.asarray(mesh.faces), record)


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
    outer = np.asarray(boundary_points_mm, dtype=float)
    axis = np.asarray(insertion_direction, dtype=float)
    socket = np.asarray(socket_inward_direction, dtype=float)
    if (outer.ndim != 2 or outer.shape[1:] != (3,) or len(outer) < 3
            or not np.isfinite(outer).all()):
        raise ValueError(f"{interface_id}: invalid shared boundary")
    if any(x.shape != (3,) or not np.isfinite(x).all()
           or np.linalg.norm(x) <= 1e-10 for x in (axis, socket)):
        raise ValueError(f"{interface_id}: invalid mating direction")
    axis = axis / np.linalg.norm(axis)
    socket = socket / np.linalg.norm(socket)
    if axis @ socket < 1 - 1e-6:
        raise ValueError(f"{interface_id}: both sides require the same validated inward axis")
    ratio, clearance, limit = map(float, (scale_ratio, clearance_mm, extension_depth_limit_mm))
    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("interface scale ratio must be in (0, 1]")
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError("clearance must be finite and nonnegative")

    shape = classify_interface_shape(outer)
    shape["source"] = "frozen_simplified_boundary"
    scaled_contour = shape["classification"] == "planar"
    n = len(outer)
    if not np.isfinite(limit) or limit < MIN_INTERFACE_EXTENSION_MM:
        raise InterfaceDepthError("connector requires a positive depth preference")
    alignment = 0.0
    profile_record = None
    # Both styles reuse the original full-rim ellipse profile. Curved rims
    # retain the image-selected mean-plane shoulder before axial extension.
    origin = outer.mean(axis=0)
    u, v = orthonormal_basis(axis)
    candidate = homothetic_loop_points(outer, axis, ratio)
    xy, profile_record = _simple_planar_inner_contour(
        project_points(candidate, origin, u, v), outer)
    base_plane = (float(np.max(outer @ axis)) + MIN_INTERFACE_EXTENSION_MM
                  if scaled_contour else float(np.mean(outer @ axis)))
    alignment = base_plane - float(np.min(outer @ axis)) if scaled_contour else 0.0
    if scaled_contour:
        # Plane-alignment travel positions the shared tip beyond the highest
        # outer-boundary point; it is not connector depth. Keep the preferred
        # axial extension independent and let the mortise ray audit validate
        # the combined alignment + extension against actual host thickness.
        limit = min(limit, MAX_INTERFACE_EXTENSION_MM)
    if limit < MIN_INTERFACE_EXTENSION_MM:
        raise InterfaceDepthError(f"{interface_id}: plane alignment leaves no connector depth")
    planar_tip = origin + xy[:, :1]*u + xy[:, 1:]*v
    planar_tip += (base_plane-planar_tip @ axis)[:, None]*axis
    socket_tip = radial_offset_points(planar_tip, origin, u, v, axis, clearance)
    m = n
    root_ids = list(range(n)) if scaled_contour else list(range(2*n, 3*n))
    tip_ids = list(range(n, 2*n))
    closure_faces = []
    if not scaled_contour:
        for i in range(n):
            nxt = (i+1) % n
            closure_faces.extend(((i, nxt, 2*n+i), (nxt, 2*n+nxt, 2*n+i)))
    closure_faces = np.asarray(closure_faces, dtype=np.int64).reshape(-1, 3)
    if not scaled_contour:
        # Preserve the existing shoulder exactly; cap its original root ring.
        # Do not emit coincident root/tip rings or zero-height connector walls.
        shoulder = closure_faces.copy()
        shoulder[shoulder >= 2*n] -= n
        cap = np.asarray([(n, n+i, n+i+1) for i in range(1, n-1)])
        faces = np.vstack((shoulder, cap))
        def close_shoulder(side, profile):
            return _surface(np.vstack((outer, profile)), faces, {
                "interface_id": interface_id, "side": side,
                "boundary_vertex_count": n, "outer_boundary_cap_face_count": 0,
                "selected_strategy": "curved_shoulder_without_connector",
                "shape_evidence": shape, "inner_planar_contour": profile_record,
                "extension_direction": axis.tolist(), "stage04_mating_direction": axis.tolist(),
                "extension_depth_mm": 0.0, "tenon_extension_depth_mm": 0.0,
                "mortise_recess_depth_mm": 0.0, "clearance_mm": clearance,
                "bottom_clearance_mm": 0.0, "scale_ratio": ratio,
                "tip_vertex_start": n, "tip_vertex_count": n, "root_vertex_start": n,
                "connector_wall_face_count": 0, "connector_segments": 0,
                "closure_face_count": len(faces), "shoulder_face_count": len(shoulder),
                "closure_boundary_shift_mm": 0.0, "plane_alignment_travel_max_mm": 0.0,
                "depth_probe_query_count": 0, "extension_depth_attempts": [],
            })
        return PairedInterfaceSurfaces(close_shoulder("tenon", planar_tip),
                                       close_shoulder("mortise", socket_tip))
    socket_root = socket_tip - axis*alignment

    # One ray batch for the scaled end profile; shorter attempts reuse
    # distances. No scan of the entire generated assembly is introduced.
    center = socket_root.mean(axis=0)
    samples = np.vstack((socket_root, (socket_root + np.roll(socket_root,-1,axis=0))*.5,
                         (socket_root+center)*.5, center[None, :]))
    sample_count = len(socket_root)
    probe_labels = (
        [f"profile_vertex_{i:03d}" for i in range(sample_count)]
        + [f"profile_edge_midpoint_{i:03d}" for i in range(sample_count)]
        + [f"profile_center_midpoint_{i:03d}" for i in range(sample_count)]
        + ["profile_center"]
    )
    hits, hit_faces = mortise_shell_probe.exits(
        samples, np.broadcast_to(axis, samples.shape),
        alignment + limit + clearance + INTERFACE_WALL_RESERVE_MM,
        epsilon=MORTISE_RAY_NUMERICAL_EPSILON_MM,
    )
    depth = limit
    attempts = []
    while True:
        required_travel = alignment + depth + clearance + INTERFACE_WALL_RESERVE_MM
        blocked = np.isfinite(hits) & (hits <= required_travel)
        blocked_indices = np.flatnonzero(blocked)
        attempts.append({
            "candidate_depth_mm": depth,
            "plane_alignment_mm": alignment,
            "clearance_mm": clearance,
            "wall_reserve_mm": INTERFACE_WALL_RESERVE_MM,
            "required_travel_mm": required_travel,
            "collision": bool(blocked.any()),
            "unsafe_exit_count": int(blocked.sum()),
            "probe_count": len(samples),
            "blocked_probes": [
                {
                    "probe_index": int(index),
                    "probe_type": probe_labels[index],
                    "hit_distance_mm": float(hits[index]),
                    "required_travel_mm": required_travel,
                    "remaining_wall_mm": float(hits[index] - required_travel),
                    "source_triangle_index": int(hit_faces[index]),
                    "probe_origin_mm": samples[index].round(6).tolist(),
                    "ray_direction": axis.round(9).tolist(),
                }
                for index in blocked_indices
            ],
        })
        if not blocked.any():
            break
        if depth <= MIN_INTERFACE_EXTENSION_MM + 1e-9:
            diagnostic = {
                "minimum_depth_mm": MIN_INTERFACE_EXTENSION_MM,
                "plane_alignment_mm": alignment,
                "clearance_mm": clearance,
                "wall_reserve_mm": INTERFACE_WALL_RESERVE_MM,
                "minimum_required_travel_mm": (
                    alignment + MIN_INTERFACE_EXTENSION_MM + clearance
                    + INTERFACE_WALL_RESERVE_MM
                ),
                "probe_count": len(samples),
                "blocked_probe_count": int(blocked.sum()),
                "blocked_probes": attempts[-1]["blocked_probes"],
                "attempts": attempts,
            }
            raise InterfaceDepthError(
                f"{interface_id}: minimum connector depth exceeds host thickness; "
                f"diagnostic={json.dumps(diagnostic, ensure_ascii=False, separators=(',', ':'))}"
            )
        depth = max(depth*.5, MIN_INTERFACE_EXTENSION_MM)

    walls = []
    for i in range(m):
        nxt = (i+1) % m
        walls.extend(((root_ids[i], root_ids[nxt], tip_ids[i]),
                      (root_ids[nxt], tip_ids[nxt], tip_ids[i])))
    # Convex disk triangulation with no extra centre vertex.
    tip_faces = [(tip_ids[0], tip_ids[i], tip_ids[i+1]) for i in range(1,m-1)]
    faces = np.vstack((closure_faces, np.asarray(walls), np.asarray(tip_faces)))

    def build(side):
        extra = clearance if side == "mortise" else 0.0
        tip = socket_tip if side == "mortise" else planar_tip
        vertices = np.vstack((outer, tip+axis*(depth+extra)))
        return _surface(vertices, faces, {
            "interface_id": interface_id, "side": side,
            "boundary_vertex_count": n, "outer_boundary_cap_face_count": 0,
            "selected_strategy": "scaled_inner_contour",
            "inner_planar_contour": profile_record,
            "shape_evidence": shape,
            "extension_direction": axis.tolist(),
            "stage04_mating_direction": axis.tolist(),
            "extension_depth_mm": depth+extra,
            "tenon_extension_depth_mm": depth,
            "mortise_recess_depth_mm": depth+clearance,
            "clearance_mm": clearance,
            "scale_ratio": ratio,
            "connector_segments": m, "tip_vertex_start": n, "tip_vertex_count": m,
            "root_vertex_start": 0, "connector_wall_face_count": 2*m,
            "closure_face_count": len(closure_faces), "closure_boundary_shift_mm": 0.0,
            "depth_reference": "frozen_outer_boundary",
            "plane_alignment_travel_max_mm": alignment,
            "depth_probe_query_count": 1, "extension_depth_attempts": attempts,
            "wall_reserve_mm": INTERFACE_WALL_RESERVE_MM,
        })

    return PairedInterfaceSurfaces(build("tenon"), build("mortise"))
