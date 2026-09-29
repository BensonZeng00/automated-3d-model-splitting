"""Join preserved exterior triangles to a Stage 04 interface loop geometrically."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree
import trimesh
from .boundary_matching import distances_to_loop

from .mesh import triangulate_boundary_cap_without_center


@dataclass(frozen=True)
class InterfaceAttachment:
    interface_id: str
    side: str
    vertices: np.ndarray
    faces: np.ndarray
    outer_vertex_count: int
    outer_cap_face_count: int


def _oriented_source_boundary_loops(
    vertices: np.ndarray, faces: np.ndarray
) -> list[list[int]]:
    """Follow exterior half edges without joining opposite winding at a pinch."""
    directed = [
        (int(left), int(right))
        for face in faces
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
    ]
    counts = Counter(tuple(sorted(edge)) for edge in directed)
    boundary = {edge for edge in directed if counts[tuple(sorted(edge))] == 1}
    incoming: dict[int, list[int]] = defaultdict(list)
    outgoing: dict[int, list[int]] = defaultdict(list)
    for left, right in boundary:
        outgoing[left].append(right)
        incoming[right].append(left)
    continuations: dict[tuple[int, int], tuple[int, int]] = {}
    for vertex in incoming.keys() | outgoing.keys():
        sources = incoming[vertex]
        targets = outgoing[vertex]
        if len(sources) != len(targets):
            raise ValueError(f"source boundary is unbalanced at vertex {vertex}")
        choices = sorted(
            (
                float(np.dot(vertices[source] - vertices[vertex],
                             vertices[target] - vertices[vertex]) /
                      max(np.linalg.norm(vertices[source] - vertices[vertex]) *
                          np.linalg.norm(vertices[target] - vertices[vertex]), 1e-12)),
                source, target,
            )
            for source in sources for target in targets
        )
        used_sources: set[int] = set()
        used_targets: set[int] = set()
        for _, source, target in choices:
            if source in used_sources or target in used_targets:
                continue
            continuations[(source, vertex)] = (vertex, target)
            used_sources.add(source)
            used_targets.add(target)
    loops: list[list[int]] = []
    unused = set(boundary)
    while unused:
        first = min(unused)
        current = first
        trail: list[int] = []
        while current in unused:
            unused.remove(current)
            trail.append(current[0])
            current = continuations[current]
        if current != first:
            raise ValueError("source boundary did not close into a directed loop")
        path: list[int] = []
        positions: dict[int, int] = {}
        for vertex in trail + trail[:1]:
            previous = positions.get(vertex)
            if previous is None:
                positions[vertex] = len(path)
                path.append(vertex)
                continue
            cycle = path[previous:]
            if len(cycle) < 3:
                raise ValueError("source boundary contains a cycle shorter than three edges")
            loops.append(cycle)
            for removed in path[previous + 1:]:
                positions.pop(removed)
            path = path[:previous + 1]
    loops.sort(key=lambda loop: (-len(loop), tuple(loop)))
    return loops


def _ordered_bridge(
    source_points: np.ndarray,
    source_ids: np.ndarray,
    target_points: np.ndarray,
    target_ids: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Triangulate a monotone ribbon by matching normalized boundary arclength."""
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

    source_samples = spaced_samples(source_points)
    reversed_points = np.concatenate((target_points[:1], target_points[:0:-1]))
    reversed_ids = np.concatenate((target_ids[:1], target_ids[:0:-1]))
    alignments = []
    for oriented_points, oriented_ids, orientation in (
        (target_points, target_ids, "forward"),
        (reversed_points, reversed_ids, "reverse"),
    ):
        target_samples = spaced_samples(oriented_points)
        costs = [
            float(np.linalg.norm(
                source_samples - np.roll(target_samples, shift, axis=0), axis=1
            ).mean())
            for shift in range(len(source_samples))
        ]
        shift = int(np.argmin(costs))
        approximate_offset = int(round(-shift * len(oriented_ids) / len(source_samples)))
        for adjustment in range(-2, 3):
            offset = approximate_offset + adjustment
            rotated_points = np.roll(oriented_points, offset, axis=0)
            rotated_samples = spaced_samples(rotated_points)
            cost = float(np.linalg.norm(source_samples - rotated_samples, axis=1).mean())
            alignments.append((cost, rotated_points, np.roll(oriented_ids, offset), orientation))
    _alignment_cost, target_points, target_ids, _orientation = min(
        alignments, key=lambda item: item[0]
    )
    ns, nt = len(source_ids), len(target_ids)
    nearest_targets = cKDTree(target_points).query(source_points)[1]
    # Unwrap the nearest-point sequence around the closed contour, then remove
    # local reversals caused by close self-approaching arcs. This tracks actual
    # geometric correspondence instead of assuming both rims have uniform
    # point density along the same normalized arclength.
    nearest_targets = np.asarray(nearest_targets, dtype=np.int64)
    nearest_targets = np.rint(np.unwrap(
        nearest_targets * (2.0 * np.pi / len(target_ids))
    ) * (len(target_ids) / (2.0 * np.pi))).astype(np.int64)
    nearest_targets -= nearest_targets[0]
    nearest_targets = np.maximum.accumulate(nearest_targets)
    nearest_targets = np.clip(nearest_targets, 0, len(target_ids))
    faces: list[tuple[int, int, int]] = []
    source_index = target_index = 0
    while source_index < ns or target_index < nt:
        desired_target = (
            int(nearest_targets[source_index + 1])
            if source_index + 1 < ns else nt
        )
        if target_index < min(desired_target, nt):
            faces.append((
                int(source_ids[source_index % ns]),
                int(target_ids[(target_index + 1) % nt]),
                int(target_ids[target_index % nt]),
            ))
            target_index += 1
        elif source_index < ns:
            faces.append((
                int(source_ids[source_index % ns]),
                int(source_ids[(source_index + 1) % ns]),
                int(target_ids[target_index % nt]),
            ))
            source_index += 1
        else:
            faces.append((
                int(source_ids[source_index % ns]),
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


def _orient_cap_to_source_boundary(
    cap_faces: list[list[int]],
    loop: list[int],
    source_directed_edges: set[tuple[int, int]],
) -> None:
    """Orient connected cap triangles by topology, anchored to source edges."""
    edge_faces: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
    for face_index, face in enumerate(cap_faces):
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            directed = (int(left), int(right))
            edge_faces.setdefault(tuple(sorted(directed)), []).append((face_index, directed))
    if any(len(owners) > 2 for owners in edge_faces.values()):
        raise ValueError("source boundary cap has an edge shared by more than two faces")

    flips: dict[int, bool] = {}
    for position, left in enumerate(loop):
        right = int(loop[(position + 1) % len(loop)])
        edge = (int(left), right)
        owners = edge_faces.get(tuple(sorted(edge)), [])
        if len(owners) != 1:
            raise ValueError("source boundary cap does not cover each loop edge once")
        source_edge = edge if edge in source_directed_edges else edge[::-1]
        if source_edge not in source_directed_edges:
            raise ValueError("source boundary edge has no oriented exterior triangle")
        face_index, cap_edge = owners[0]
        required_flip = cap_edge == source_edge
        previous = flips.setdefault(face_index, required_flip)
        if previous != required_flip:
            raise ValueError("source boundary cap has conflicting boundary winding")

    queue = list(flips)
    queued = set(queue)
    for face_index in queue:
        face = cap_faces[face_index]
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            owners = edge_faces[tuple(sorted((int(left), int(right))))]
            if len(owners) != 2:
                continue
            other_index, other_edge = owners[0] if owners[1][0] == face_index else owners[1]
            this_edge = (int(left), int(right))
            required_flip = flips[face_index] ^ (this_edge == other_edge)
            previous = flips.setdefault(other_index, required_flip)
            if previous != required_flip:
                raise ValueError(
                    f"source boundary cap cannot be oriented consistently across "
                    f"edge {tuple(sorted((int(left), int(right))))}; "
                    f"face {other_index} expected flip {required_flip}, got {previous}"
                )
            if other_index not in queued:
                queue.append(other_index)
                queued.add(other_index)
    if len(flips) != len(cap_faces):
        raise ValueError("source boundary cap contains disconnected triangles")
    for face_index, flip in flips.items():
        if flip:
            cap_faces[face_index][1], cap_faces[face_index][2] = (
                cap_faces[face_index][2], cap_faces[face_index][1]
            )


def _append_planar_convex_cap(
    vertices: list[list[float]],
    faces: list[list[int]],
    loop: list[int],
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    source_area_mm2: float,
) -> dict:
    """Close an almost convex spatial rim without folded cross-body chords."""
    points = source_vertices[np.asarray(loop, dtype=np.int64)]
    origin = points.mean(axis=0)
    _u, _s, basis = np.linalg.svd(points - origin, full_matrices=False)
    projected = (points - origin) @ basis[:2].T
    signed_area = float(np.sum(
        projected[:, 0] * np.roll(projected[:, 1], -1)
        - np.roll(projected[:, 0], -1) * projected[:, 1]
    ) * 0.5)
    if abs(signed_area) <= 1e-8:
        raise ValueError("nonplanar source rim has no stable projected area")
    try:
        hull = ConvexHull(projected)
    except QhullError as exc:
        raise ValueError("nonplanar source rim has no stable planar hull") from exc
    hull_ratio = float(hull.volume / abs(signed_area))
    cap_area_fraction = float(hull.volume / max(source_area_mm2, 1e-12))
    if hull_ratio > 1.05:
        raise ValueError(
            f"nonplanar source rim is too concave for a planar hull cap ({hull_ratio:.3f})"
        )
    if cap_area_fraction < 0.10:
        raise ValueError(
            "nonplanar source rim is a minor fraction of its part surface "
            f"({cap_area_fraction:.3f})"
        )
    plane_normal = basis[2]
    touches = np.isin(source_faces, np.asarray(loop, dtype=np.int64)).sum(axis=1) >= 2
    adjacent_centers = source_vertices[source_faces[touches]].mean(axis=1)
    if len(adjacent_centers) and float(np.dot(
        adjacent_centers.mean(axis=0) - origin, plane_normal
    )) > 0:
        plane_normal = -plane_normal
    height = float(np.max((points - origin) @ plane_normal)) + 0.05
    ring_points = (
        origin + projected[hull.vertices] @ basis[:2]
        + height * plane_normal
    )
    first_id = len(vertices)
    vertices.extend(ring_points.tolist())
    ring_ids = np.arange(first_id, len(vertices), dtype=np.int64)
    faces.extend([list(face) for face in _ordered_bridge(
        points, np.asarray(loop, dtype=np.int64), ring_points, ring_ids
    )])
    cap_count, cap_record = triangulate_boundary_cap_without_center(
        vertices, faces, ring_ids.tolist(), plane_normal,
        require_planar_quality=True,
    )
    if cap_count != len(ring_ids) - 2:
        raise ValueError("planar hull cap did not cover its complete ring")
    return {
        "method": "planar_convex_rim_with_transition_band",
        "projected_hull_area_ratio": hull_ratio,
        "cap_area_fraction_of_source": cap_area_fraction,
        "ring_vertices": len(ring_ids),
        "cap_quality": cap_record["surface_quality"],
    }


def compose_part_with_interfaces(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    attachments: list[InterfaceAttachment],
) -> tuple[trimesh.Trimesh, dict]:
    """Keep exterior triangles and connect every assigned Stage 04 boundary."""
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    source_triangles = source_vertices[source_faces]
    source_area_mm2 = float(np.linalg.norm(np.cross(
        source_triangles[:, 1] - source_triangles[:, 0],
        source_triangles[:, 2] - source_triangles[:, 0],
    ), axis=1).sum() * 0.5)
    loops = _oriented_source_boundary_loops(source_vertices, source_faces)
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
            if position in selected_loops or len(loop) < 3:
                continue
            points = source_vertices[np.asarray(loop, dtype=np.int64)]
            distances = distances_to_loop(outer_points, points)
            candidates.append((float(np.quantile(distances, 0.95)), position, distances))
        if not candidates:
            raise ValueError(f"{attachment.interface_id}: no available source exterior boundary")
        _, selected, distances = min(candidates, key=lambda item: item[0])
        span = float(np.linalg.norm(np.ptp(outer_points, axis=0)))
        maximum_distance = float(distances.max())
        allowed_distance = max(0.15, 0.04 * span)
        if maximum_distance > allowed_distance:
            raise ValueError(
                f"{attachment.interface_id}: exterior is too far from the simplified interface "
                f"({maximum_distance:.4f} mm; limit {allowed_distance:.4f} mm)"
            )
        selected_loops.add(selected)
        assignments.append((attachment, selected, distances))

    vertices = source_vertices.tolist()
    faces = source_faces.tolist()
    source_edges = np.sort(np.vstack((
        source_faces[:, [0, 1]], source_faces[:, [1, 2]], source_faces[:, [2, 0]],
    )), axis=1)
    occupied_edges = set(map(tuple, source_edges.tolist()))
    source_directed_edges = {
        (int(left), int(right))
        for face in source_faces
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
    }
    capped_small_loops = 0
    reconstructed_nonplanar_caps = []
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
        added, cap_record = triangulate_boundary_cap_without_center(
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
        used_planar_fallback = False
        if (not cap_record["surface_quality"]["valid"]
                and float(np.linalg.norm(np.ptp(points, axis=0))) > 10.0):
            original_cap_faces = faces[previous_face_count:]
            original_vertex_count = len(vertices)
            del faces[previous_face_count:]
            try:
                cap_record = _append_planar_convex_cap(
                    vertices, faces, loop, source_vertices, source_faces,
                    source_area_mm2,
                )
            except ValueError as exc:
                if not str(exc).startswith("nonplanar source rim"):
                    raise
                del faces[previous_face_count:]
                del vertices[original_vertex_count:]
                faces.extend(original_cap_faces)
            else:
                reconstructed_nonplanar_caps.append({
                    "source_loop_index": index,
                    "source_boundary_points": len(loop),
                    **cap_record,
                })
                used_planar_fallback = True
        if (not used_planar_fallback
                and (added != len(loop) - 2 or np.any(cap_areas <= 1e-10))):
            del faces[previous_face_count:]
            center = len(vertices)
            vertices.append((points.mean(axis=0) + vectors[-1] * 0.001).tolist())
            for position, first in enumerate(loop):
                faces.append([int(first), int(loop[(position + 1) % len(loop)]), center])
        try:
            _orient_cap_to_source_boundary(faces[previous_face_count:], loop, source_directed_edges)
        except ValueError as exc:
            raise ValueError(f"source boundary loop {index} ({len(loop)} points): {exc}") from exc
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
        "reconstructed_nonplanar_caps": reconstructed_nonplanar_caps,
        "bridge_strategy": "geometric_monotone_nearest_ribbon",
        "interfaces": interface_records,
    }
