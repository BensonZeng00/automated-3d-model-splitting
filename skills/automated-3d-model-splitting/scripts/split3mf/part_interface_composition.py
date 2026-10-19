"""Join preserved exterior triangles to a Stage 04 interface loop geometrically."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from itertools import permutations

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree
import trimesh
from .boundary_matching import distances_to_loop
from .boundary_smoothing import MAX_RELATIVE_SHAPE_CHANGE

from .mesh import triangulate_boundary_cap_without_center

MAX_EXTERIOR_INTERFACE_DISTANCE_MM = 2.0


@dataclass(frozen=True)
class InterfaceAttachment:
    interface_id: str
    side: str
    vertices: np.ndarray
    faces: np.ndarray
    outer_vertex_count: int
    outer_cap_face_count: int
    original_boundary_points: np.ndarray | None = None


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
    """Find a short monotone ribbon between unequal cyclic contours."""
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    source_ids = np.asarray(source_ids, dtype=np.int64)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    ns, nt = len(source_ids), len(target_ids)

    def solve(oriented_points: np.ndarray, oriented_ids: np.ndarray):
        points = np.vstack((oriented_points, oriented_points[:1]))
        source = np.vstack((source_points, source_points[:1]))
        width = nt + 1
        predecessor = np.empty((ns + 1, width), dtype=np.int32)
        previous = np.cumsum(np.linalg.norm(source[0] - points, axis=1))
        predecessor[0] = np.arange(width)
        columns = np.arange(width, dtype=np.int32)
        for row in range(1, ns + 1):
            costs = np.linalg.norm(source[row] - points, axis=1)
            prefix = np.cumsum(costs)
            candidates = previous - np.r_[0.0, prefix[:-1]]
            minima = np.minimum.accumulate(candidates)
            changed = np.r_[True, candidates[1:] < minima[:-1]]
            predecessor[row] = np.maximum.accumulate(np.where(changed, columns, 0))
            previous = prefix + minima
        faces = []
        row, column = ns, nt
        while row:
            entered = int(predecessor[row, column])
            while column > entered:
                faces.append((int(source_ids[row % ns]),
                              int(oriented_ids[column % nt]),
                              int(oriented_ids[(column-1) % nt])))
                column -= 1
            faces.append((int(source_ids[(row-1) % ns]),
                          int(source_ids[row % ns]),
                          int(oriented_ids[column % nt])))
            row -= 1
        while column:
            faces.append((int(source_ids[0]), int(oriented_ids[column % nt]),
                          int(oriented_ids[(column-1) % nt])))
            column -= 1
        return float(previous[-1]), faces

    best = None
    for points, ids in ((target_points, target_ids),
                        (target_points[::-1], target_ids[::-1])):
        closest = int(cKDTree(points).query(source_points[0])[1])
        for offset in ((closest - 1) % nt, closest, (closest + 1) % nt):
            aligned_points = np.roll(points, -offset, axis=0)
            aligned_ids = np.roll(ids, -offset)
            cost, faces = solve(aligned_points, aligned_ids)
            if best is None or cost < best[0]:
                best = cost, faces
    return list(reversed(best[1]))


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
    source_twice_areas = np.linalg.norm(np.cross(
        source_triangles[:, 1] - source_triangles[:, 0],
        source_triangles[:, 2] - source_triangles[:, 0],
    ), axis=1)
    source_area_mm2 = float(source_twice_areas.sum() * 0.5)
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
        if cap_count < 0 or cap_count >= len(interface_faces):
            raise ValueError(f"{attachment.interface_id}: invalid removable cap count")
        if outer_count < 3 or len(interface_vertices) < outer_count:
            raise ValueError(f"{attachment.interface_id}: interface rings have invalid counts")
        if cap_count == 0:
            patch = trimesh.Trimesh(interface_vertices, interface_faces, process=False)
            counts = np.bincount(patch.edges_unique_inverse)
            boundary = {tuple(edge) for edge in patch.edges_unique[counts == 1]}
            expected = {tuple(sorted((i, (i+1) % outer_count))) for i in range(outer_count)}
            if boundary != expected or np.any(counts > 2):
                raise ValueError(f"{attachment.interface_id}: direct closure patch has invalid boundary")
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
        allowed_distance = min(
            MAX_EXTERIOR_INTERFACE_DISTANCE_MM,
            0.15 + MAX_RELATIVE_SHAPE_CHANGE * span,
        )
        if maximum_distance > allowed_distance:
            raise ValueError(
                f"{attachment.interface_id}: exterior is too far from the simplified interface "
                f"({maximum_distance:.4f} mm; limit {allowed_distance:.4f} mm)"
            )
        selected_loops.add(selected)
        assignments.append((attachment, selected, distances))

    # The frozen Stage 03 contours define the cut through source triangles.
    # Clipping at interpolated edge crossings avoids a staircase source rim.
    from .surface_collar_remesh import clip_source_surface
    from .surface_transition_validation import transition_self_intersection_count
    collar_records = []
    for attachment, selected, original_distances in assignments:
        source_loop = np.asarray(loops[selected], dtype=np.int64)
        original = attachment.original_boundary_points
        target = np.asarray(attachment.vertices[:attachment.outer_vertex_count], dtype=np.float64)
        displacement = (
            float(np.linalg.norm(np.asarray(original) - target, axis=1).max())
            if original is not None and np.shape(original) == np.shape(target)
            else float(original_distances.max())
        )
        if displacement < 1e-4:
            displacement = 0.0
        source_points = source_vertices[source_loop]
        edge_lengths = np.linalg.norm(source_points - np.roll(source_points, -1, axis=0), axis=1)
        spacing = float(np.quantile(edge_lengths, 0.95))
        width = original_distances.max() + spacing if displacement else 0.0
        others = [
            source_vertices[np.asarray(loops[other_selected], dtype=np.int64)]
            for other_attachment, other_selected, _ in assignments
            if other_attachment is not attachment
        ]
        if others:
            separation = min(float(cKDTree(other).query(source_points)[0].min()) for other in others)
            width = min(width, 0.3 * separation)
        collar_records.append({
            "interface_id": attachment.interface_id, "side": attachment.side,
            "boundary_displacement_mm": displacement,
            "source_to_frozen_gap_mm": float(original_distances.max()),
            "local_spacing_mm": spacing, "collar_width_mm": float(width),
        })

    original_gaps = {
        id(attachment): float(distances.max()) for attachment, _, distances in assignments
    }
    original_loops = {
        id(attachment): np.asarray(loops[selected], dtype=np.int64)
        for attachment, selected, _ in assignments
    }
    baseline_crossings: dict[int, int] = {}

    def ribbon_crossings(mesh_vertices: np.ndarray, loop: np.ndarray,
                         attachment: InterfaceAttachment) -> int:
        target = np.asarray(attachment.vertices[:attachment.outer_vertex_count], dtype=np.float64)
        target_ids = np.arange(len(target), dtype=np.int64) + len(mesh_vertices)
        ribbon = _ordered_bridge(mesh_vertices[loop], loop, target, target_ids)
        return transition_self_intersection_count(
            np.vstack((mesh_vertices, target)), np.asarray(ribbon, dtype=np.int64)
        )

    attachment_positions = {id(attachment): position for position, attachment in enumerate(attachments)}
    for collar_scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.49, 0.48,
                         0.45, 0.4, 0.3, 0.25, 0.2, 0.125, 0.0625, 0.0):
        widths = [record["collar_width_mm"] * collar_scale for record in collar_records]
        clipped_vertices, retained_source_faces, source_face_origins, clip_record = (
            clip_source_surface(
                source_vertices, source_faces,
                [np.asarray(item.vertices[:item.outer_vertex_count], dtype=np.float64)
                 for item, _, _ in assignments],
                widths,
                [original_loops[id(item)] for item, _, _ in assignments],
            )
        )
        retained_triangles = clipped_vertices[retained_source_faces]
        retained_twice_areas = np.linalg.norm(np.cross(
            retained_triangles[:, 1] - retained_triangles[:, 0],
            retained_triangles[:, 2] - retained_triangles[:, 0],
        ), axis=1)
        degenerate = retained_twice_areas <= 2e-10
        surviving_degenerate_max_displacement_mm = 0.0
        if np.any(degenerate):
            degenerate_origins = source_face_origins[degenerate]
            if (np.any(source_twice_areas[degenerate_origins] > 2e-10)
                    or len(np.unique(degenerate_origins)) != len(degenerate_origins)):
                continue
            retained = retained_triangles[degenerate]
            original = source_triangles[degenerate_origins]
            # Match all three vertices one-to-one, independent of face winding.
            # A surviving source defect may shift by at most 1 mm per vertex.
            best_displacement = np.full(len(retained), np.inf)
            for order in permutations(range(3)):
                displacement = np.linalg.norm(retained - original[:, order], axis=2).max(axis=1)
                best_displacement = np.minimum(best_displacement, displacement)
            if np.any(best_displacement > 1.0):
                continue
            surviving_degenerate_max_displacement_mm = float(best_displacement.max())
        candidate_loops = _oriented_source_boundary_loops(clipped_vertices, retained_source_faces)
        candidate_assignments = []
        selected_loops = set()
        for attachment in attachments:
            target = np.asarray(attachment.vertices[:attachment.outer_vertex_count], dtype=np.float64)
            candidates = []
            for position, loop in enumerate(candidate_loops):
                if position in selected_loops:
                    continue
                distances = distances_to_loop(target, clipped_vertices[np.asarray(loop)])
                candidates.append((float(np.quantile(distances, 0.95)), position, distances))
            if not candidates:
                break
            _, selected, distances = min(candidates, key=lambda item: item[0])
            permitted_gap = min(MAX_EXTERIOR_INTERFACE_DISTANCE_MM, max(
                1.5 * original_gaps[id(attachment)],
                1.5 * widths[attachment_positions[id(attachment)]],
                0.15,
            ))
            if float(distances.max()) > permitted_gap + 1e-6:
                break
            selected_loops.add(selected)
            candidate_assignments.append((attachment, selected, distances))
        if len(candidate_assignments) != len(attachments):
            continue
        candidate_crossings = []
        collision_regression = False
        for attachment, selected, _ in candidate_assignments:
            crossing = ribbon_crossings(
                clipped_vertices, np.asarray(candidate_loops[selected], dtype=np.int64),
                attachment,
            )
            candidate_crossings.append(crossing)
            if crossing:
                key = id(attachment)
                if key not in baseline_crossings:
                    baseline_crossings[key] = ribbon_crossings(
                        source_vertices, original_loops[key], attachment,
                    )
                if crossing > baseline_crossings[key]:
                    collision_regression = True
                    break
        if not collision_regression:
            source_vertices = clipped_vertices
            loops = candidate_loops
            assignments = candidate_assignments
            break
    else:
        raise ValueError("frozen-boundary surface remeshing cannot retain separate valid rims")

    vertices = source_vertices.tolist()
    faces = retained_source_faces.tolist()
    source_edges = np.sort(np.vstack((
        retained_source_faces[:, [0, 1]], retained_source_faces[:, [1, 2]], retained_source_faces[:, [2, 0]],
    )), axis=1)
    occupied_edges = set(map(tuple, source_edges.tolist()))
    source_directed_edges = {
        (int(left), int(right))
        for face in retained_source_faces
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
                    vertices, faces, loop, source_vertices, retained_source_faces,
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
        target_ring = interface_vertices[:outer_count]
        source_ring = source_vertices[source_loop]
        nearest_distance, nearest_index = cKDTree(target_ring).query(source_ring)
        near_target = nearest_distance < 0.5
        if np.any(near_target):
            # A zero-width ribbon cannot be triangulated. Move only the cut
            # rim of the source shell; the confirmed interface stays fixed.
            # Fade the small correction across the complete loop so the
            # 0.5 mm proximity threshold cannot create a displacement kink.
            _, _, axes = np.linalg.svd(target_ring - target_ring.mean(axis=0), full_matrices=False)
            tangents = target_ring[(nearest_index + 1) % outer_count] - target_ring[(nearest_index - 1) % outer_count]
            lateral = np.cross(axes[-1], tangents)
            lateral /= np.maximum(np.linalg.norm(lateral, axis=1, keepdims=True), 1e-12)
            radial = source_ring - target_ring.mean(axis=0)
            lateral *= np.where(np.sum(lateral * radial, axis=1, keepdims=True) < 0, -1.0, 1.0)
            weights = np.clip((0.5 - nearest_distance) / 0.5, 0.0, 1.0)
            corrections = 0.05 * weights[:, None] * lateral
            source_vertices[source_loop] += corrections
            for vertex_id in source_loop:
                vertices[int(vertex_id)] = source_vertices[int(vertex_id)].tolist()
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
        bridge_cross = np.cross(
            bridge_triangles[:, 1] - bridge_triangles[:, 0],
            bridge_triangles[:, 2] - bridge_triangles[:, 0],
        )
        bridge_twice_areas = np.linalg.norm(bridge_cross, axis=1)
        bridge_altitudes = bridge_twice_areas / np.maximum(bridge_lengths.max(axis=1), 1e-12)
        if np.any(bridge_twice_areas <= 1e-10):
            bad = np.flatnonzero(bridge_twice_areas <= 1e-10)
            raise ValueError(
                f"{attachment.interface_id}: surface transition has degenerate triangles "
                f"(count={len(bad)}, first={int(bad[0])}, "
                f"maximum_edge_mm={float(bridge_lengths[bad].max()):.6f}, "
                f"minimum_area_mm2={float(bridge_twice_areas[bad].min() * .5):.9g})"
            )
        faces.extend(bridge_faces)
        kept = (interface_faces[:-attachment.outer_cap_face_count]
                if attachment.outer_cap_face_count else interface_faces)
        faces.extend((kept + offset).tolist())
        interface_records.append({
            "interface_id": attachment.interface_id,
            "side": attachment.side,
            "source_interface_boundary_points": int(len(source_loop)),
            "simplified_interface_boundary_points": int(outer_count),
            "maximum_boundary_gap_mm": float(distances.max()),
            "source_rim_adjusted_points": int(np.count_nonzero(near_target)),
            "source_rim_adjustment_mm": float(np.linalg.norm(corrections, axis=1).max()) if np.any(near_target) else 0.0,
            "bridge_maximum_edge_mm": float(bridge_lengths.max()),
            "bridge_minimum_altitude_mm": float(bridge_altitudes.min()),
            "bridge_faces_below_0_01mm_altitude": int(np.count_nonzero(bridge_altitudes < 0.01)),
            "outer_cap_faces_removed": int(attachment.outer_cap_face_count),
        })
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64), process=False,
    )
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_inversion(mesh, multibody=True)
    return mesh, {
        "source_exterior_faces_preserved": int(len(retained_source_faces)),
        "source_exterior_faces_replaced": int(clip_record["source_faces_removed"]),
        "source_face_origin_indices": source_face_origins.tolist(),
        "retained_source_degenerate_faces": int(np.count_nonzero(degenerate)),
        "maximum_source_degenerate_vertex_displacement_mm": surviving_degenerate_max_displacement_mm,
        "source_surface_remesh": clip_record,
        "interface_collars": collar_records,
        "interface_collar_scale": collar_scale,
        "transition_self_intersections": [
            {"interface_id": item.interface_id, "side": item.side, "crossings": count}
            for (item, _, _), count in zip(assignments, candidate_crossings)
        ],
        "small_source_loops_closed": int(capped_small_loops),
        "reconstructed_nonplanar_caps": reconstructed_nonplanar_caps,
        "bridge_strategy": "geometric_monotone_nearest_ribbon",
        "interfaces": interface_records,
    }
