"""Join preserved exterior triangles to a Stage 04 interface loop geometrically."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .mesh import boundary_loops, triangulate_boundary_cap_without_center


@dataclass(frozen=True)
class InterfaceAttachment:
    interface_id: str
    side: str
    vertices: np.ndarray
    faces: np.ndarray
    outer_vertex_count: int
    outer_cap_face_count: int


def _ordered_bridge(
    source_points: np.ndarray,
    source_ids: np.ndarray,
    target_points: np.ndarray,
    target_ids: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Triangulate a local monotone ribbon by XYZ proximity, without ID matching."""
    def spaced_samples(points: np.ndarray) -> np.ndarray:
        edge_lengths = np.linalg.norm(
            np.roll(points, -1, axis=0) - points, axis=1
        )
        cumulative = np.r_[0.0, np.cumsum(edge_lengths)]
        distances = np.linspace(0.0, cumulative[-1], 64, endpoint=False)
        indices = np.searchsorted(cumulative, distances, side="right") - 1
        fractions = (distances - cumulative[indices]) / np.maximum(
            edge_lengths[indices], 1e-12
        )
        return (
            points[indices] * (1.0 - fractions[:, None])
            + points[(indices + 1) % len(points)] * fractions[:, None]
        )

    start = int(np.argmin(np.linalg.norm(source_points - target_points[0], axis=1)))
    source_points = np.roll(source_points, -start, axis=0)
    source_ids = np.roll(source_ids, -start)
    reversed_points = np.concatenate((target_points[:1], target_points[:0:-1]))
    source_samples = spaced_samples(source_points)
    forward_cost = np.linalg.norm(
        source_samples - spaced_samples(target_points), axis=1
    ).mean()
    reverse_cost = np.linalg.norm(
        source_samples - spaced_samples(reversed_points), axis=1
    ).mean()
    if reverse_cost < forward_cost:
        target_points = reversed_points
        target_ids = np.concatenate((target_ids[:1], target_ids[:0:-1]))
    ns, nt = len(source_ids), len(target_ids)
    faces: list[tuple[int, int, int]] = []
    target_index = 0
    for source_index in range(ns):
        candidates = np.arange(target_index, min(target_index + 17, nt))
        nearest = int(candidates[np.argmin(np.linalg.norm(
            target_points[candidates] - source_points[source_index], axis=1
        ))])
        while target_index < nearest:
            faces.append((
                int(source_ids[source_index % ns]),
                int(target_ids[(target_index + 1) % nt]),
                int(target_ids[target_index % nt]),
            ))
            target_index += 1
        faces.append((
            int(source_ids[source_index]),
            int(source_ids[(source_index + 1) % ns]),
            int(target_ids[target_index]),
        ))
    while target_index < nt:
        faces.append((
            int(source_ids[0]),
            int(target_ids[(target_index + 1) % nt]),
            int(target_ids[target_index % nt]),
        ))
        target_index += 1
    return faces


def compose_part_with_interface(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    interface_vertices: np.ndarray,
    interface_faces: np.ndarray,
    outer_vertex_count: int,
    outer_cap_face_count: int,
) -> tuple[trimesh.Trimesh, dict]:
    """Single-interface adapter for the shared multi-interface composition path."""
    return compose_part_with_interfaces(source_vertices, source_faces, [
        InterfaceAttachment(
            interface_id="single", side="tenon", vertices=interface_vertices,
            faces=interface_faces, outer_vertex_count=outer_vertex_count,
            outer_cap_face_count=outer_cap_face_count,
        )
    ])


def compose_part_with_interfaces(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    attachments: list[InterfaceAttachment],
) -> tuple[trimesh.Trimesh, dict]:
    """Keep exterior triangles and connect every assigned Stage 04 boundary."""
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    loops = boundary_loops(source_faces)
    if not loops:
        raise ValueError("source exterior has no open boundary to receive the interface")
    assignments: list[tuple[InterfaceAttachment, int, np.ndarray]] = []
    selected_loops: set[int] = set()
    for attachment in attachments:
        interface_vertices = np.asarray(attachment.vertices, dtype=np.float64)
        interface_faces = np.asarray(attachment.faces, dtype=np.int64)
        outer_count = int(attachment.outer_vertex_count)
        cap_count = int(attachment.outer_cap_face_count)
        if cap_count <= 0 or cap_count >= len(interface_faces):
            raise ValueError(f"{attachment.interface_id}: interface outer cap is missing")
        if outer_count < 3 or len(interface_vertices) < 2 * outer_count:
            raise ValueError(f"{attachment.interface_id}: interface rings have invalid counts")
        outer_points = interface_vertices[:outer_count]
        candidates = []
        for position, loop in enumerate(loops):
            if position in selected_loops or len(loop) < max(12, outer_count // 3):
                continue
            points = source_vertices[np.asarray(loop, dtype=np.int64)]
            distances = cKDTree(points).query(outer_points)[0]
            candidates.append((float(np.quantile(distances, 0.95)), position, distances))
        if not candidates:
            raise ValueError(f"{attachment.interface_id}: no available source exterior boundary")
        _, selected, distances = min(candidates, key=lambda item: item[0])
        span = float(np.linalg.norm(np.ptp(outer_points, axis=0)))
        if float(distances.max()) > max(0.15, 0.04 * span):
            raise ValueError(f"{attachment.interface_id}: exterior is too far from the simplified interface")
        selected_loops.add(selected)
        assignments.append((attachment, selected, distances))

    vertices = source_vertices.tolist()
    faces = source_faces.tolist()
    source_edges = np.sort(np.vstack((
        source_faces[:, [0, 1]], source_faces[:, [1, 2]], source_faces[:, [2, 0]],
    )), axis=1)
    occupied_edges = set(map(tuple, source_edges.tolist()))
    capped_small_loops = 0
    for index, loop in enumerate(loops):
        if index in selected_loops:
            continue
        points = source_vertices[np.asarray(loop, dtype=np.int64)]
        _, _, vectors = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
        local_occupied = {
            pair
            for left_position, left in enumerate(loop)
            for right in loop[left_position + 1:]
            if (pair := tuple(sorted((int(left), int(right))))) in occupied_edges
        }
        previous_face_count = len(faces)
        added, _record = triangulate_boundary_cap_without_center(
            vertices, faces, list(map(int, loop)), vectors[-1],
            occupied_edges=local_occupied,
            require_planar_quality=False,
        )
        cap_faces = np.asarray(faces[previous_face_count:], dtype=np.int64).reshape((-1, 3))
        cap_triangles = np.asarray(
            [[vertices[int(vertex)] for vertex in triangle] for triangle in cap_faces],
            dtype=np.float64,
        ).reshape((-1, 3, 3))
        cap_areas = np.linalg.norm(np.cross(
            cap_triangles[:, 1] - cap_triangles[:, 0],
            cap_triangles[:, 2] - cap_triangles[:, 0],
        ), axis=1) * 0.5
        if added != len(loop) - 2 or np.any(cap_areas <= 1e-10):
            del faces[previous_face_count:]
            center = len(vertices)
            vertices.append((points.mean(axis=0) + vectors[-1] * 0.001).tolist())
            for position, first in enumerate(loop):
                faces.append((int(first), int(loop[(position + 1) % len(loop)]), center))
        for triangle in faces[previous_face_count:]:
            occupied_edges.update(
                tuple(sorted((int(first), int(second))))
                for first, second in (
                    (triangle[0], triangle[1]),
                    (triangle[1], triangle[2]),
                    (triangle[2], triangle[0]),
                )
            )
        capped_small_loops += 1
    interface_records = []
    for attachment, selected, distances in assignments:
        interface_vertices = np.asarray(attachment.vertices, dtype=np.float64)
        interface_faces = np.asarray(attachment.faces, dtype=np.int64)
        outer_count = int(attachment.outer_vertex_count)
        source_loop = np.asarray(loops[selected], dtype=np.int64)
        offset = len(vertices)
        vertices.extend(interface_vertices.tolist())
        target_ids = np.arange(outer_count, dtype=np.int64) + offset
        bridge_faces = _ordered_bridge(
            source_vertices[source_loop], source_loop,
            interface_vertices[:outer_count], target_ids,
        )
        bridge_triangles = np.asarray(
            [[vertices[int(vertex)] for vertex in triangle] for triangle in bridge_faces],
            dtype=np.float64,
        )
        bridge_lengths = np.linalg.norm(
            bridge_triangles - np.roll(bridge_triangles, -1, axis=1), axis=2
        )
        faces.extend(bridge_faces)
        faces.extend((interface_faces[:-attachment.outer_cap_face_count] + offset).tolist())
        interface_records.append({
            "interface_id": attachment.interface_id,
            "side": attachment.side,
            "source_interface_boundary_points": int(len(source_loop)),
            "simplified_interface_boundary_points": int(outer_count),
            "maximum_boundary_gap_mm": float(distances.max()),
            "bridge_maximum_edge_mm": float(bridge_lengths.max()),
            "outer_cap_faces_removed": int(attachment.outer_cap_face_count),
        })
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64), process=False,
    )
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_inversion(mesh, multibody=True)
    return mesh, {
        "source_exterior_faces_preserved": int(len(source_faces)),
        "small_source_loops_closed": int(capped_small_loops),
        "bridge_strategy": "geometric_monotone_nearest_ribbon",
        "interfaces": interface_records,
    }
