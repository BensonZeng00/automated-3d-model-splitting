from __future__ import annotations

import numpy as np


def _segment_hits_triangle_batch(starts, ends, targets, shared_points, tolerance, contact_tolerance):
    direction = ends - starts
    edge1 = targets[:, 1] - targets[:, 0]
    edge2 = targets[:, 2] - targets[:, 0]
    p = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, p)
    normal_length = np.linalg.norm(np.cross(edge1, edge2), axis=1)
    direction_length = np.linalg.norm(direction, axis=1)
    usable = np.abs(determinant) > np.maximum(
        tolerance * tolerance, 1e-9 * normal_length * direction_length
    )
    inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=usable)
    offset = starts - targets[:, 0]
    bary_u = np.einsum("ij,ij->i", offset, p) * inverse
    q = np.cross(offset, edge1)
    bary_v = np.einsum("ij,ij->i", direction, q) * inverse
    segment_t = np.einsum("ij,ij->i", edge2, q) * inverse
    hit_points = starts + segment_t[:, None] * direction
    away_from_shared_vertex = (
        ~np.isfinite(shared_points[:, 0])
        | (np.linalg.norm(hit_points - shared_points, axis=1) > contact_tolerance)
    )
    return (usable & away_from_shared_vertex
            & (segment_t > 1e-9) & (segment_t < 1.0 - 1e-9)
            & (bary_u >= -1e-9) & (bary_v >= -1e-9)
            & (bary_u + bary_v <= 1.0 + 1e-9))


def count_local_strip_penetrations_3d(
    faces: np.ndarray, points: np.ndarray, *, maximum_face_gap: int = 8,
    return_pairs: bool = False,
) -> int | list[tuple[int, int]]:
    """Fast phase-screen for nearby nonincident ribbon triangles in XYZ."""
    triangles = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    xyz = np.asarray(points, dtype=np.float64)[triangles]
    low, high = xyz.min(axis=1), xyz.max(axis=1)
    scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    tolerance = max(1e-8, scale * 1e-10)
    contact_tolerance = max(1e-6, scale * 1e-8)
    total = 0
    pairs: list[tuple[int, int]] = []
    for gap in range(2, min(maximum_face_gap + 1, len(triangles))):
        left = np.arange(len(triangles) - gap)
        right = left + gap
        nonincident = ~(
            triangles[left, :, None] == triangles[right, None, :]
        ).any(axis=(1, 2))
        nearby = np.all(low[left] <= high[right] + tolerance, axis=1)
        nearby &= np.all(low[right] <= high[left] + tolerance, axis=1)
        left, right = left[nonincident & nearby], right[nonincident & nearby]
        if not len(left):
            continue
        first, second = xyz[left], xyz[right]
        no_shared = np.full((len(left), 3), np.nan)
        hits = np.zeros(len(left), dtype=bool)
        for edge_index in range(3):
            hits |= _segment_hits_triangle_batch(
                first[:, edge_index], first[:, (edge_index + 1) % 3],
                second, no_shared, tolerance, contact_tolerance,
            )
            hits |= _segment_hits_triangle_batch(
                second[:, edge_index], second[:, (edge_index + 1) % 3],
                first, no_shared, tolerance, contact_tolerance,
            )
        total += int(np.count_nonzero(hits))
        if return_pairs:
            pairs.extend((int(a), int(b)) for a, b in zip(left[hits], right[hits]))
    return pairs if return_pairs else total


def first_nonincident_triangle_intersection_3d(
    faces: np.ndarray, points: np.ndarray
) -> dict:
    """Find a genuine XYZ triangle overlap, ignoring shared topological edges.

    The sweep only compares overlapping 3D bounding boxes. Edge/triangle
    penetration tests cover non-coplanar pairs; a separating-axis test covers
    positive-area coplanar overlap. A 2D projection is never used as proof of
    a 3D intersection.
    """
    triangles = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    vertices = np.asarray(points, dtype=np.float64)
    xyz = vertices[triangles]
    low, high = xyz.min(axis=1), xyz.max(axis=1)
    order = np.argsort(low[:, 0], kind="stable")
    scale = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1.0)
    tolerance = max(1e-8, scale * 1e-10)
    tested = 0

    shared_contact_tolerance = max(1e-6, scale * 1e-8)

    def coplanar_overlap(first, second):
        normal = np.cross(first[1] - first[0], first[2] - first[0])
        coordinate = int(np.argmax(np.abs(normal)))
        a = np.delete(first, coordinate, axis=1)
        b = np.delete(second, coordinate, axis=1)
        for polygon in (a, b):
            for index in range(3):
                edge = polygon[(index + 1) % 3] - polygon[index]
                axis = np.array([-edge[1], edge[0]])
                pa, pb = a @ axis, b @ axis
                if min(pa.max(), pb.max()) - max(pa.min(), pb.min()) <= tolerance:
                    return False
        return True

    for position, first_index in enumerate(order):
        last = int(np.searchsorted(low[order, 0], high[first_index, 0] + tolerance, side="right"))
        candidates = order[position + 1:last]
        if not len(candidates):
            continue
        candidates = candidates[
            np.all(low[candidates] <= high[first_index] + tolerance, axis=1)
            & np.all(high[candidates] >= low[first_index] - tolerance, axis=1)
        ]
        if not len(candidates):
            continue
        # A shared edge is an intentional adjacency; shared vertices alone
        # do not exempt two faces from an interior-overlap check.
        common_vertices = [
            set(triangles[first_index]) & set(triangles[other])
            for other in candidates
        ]
        edge_shared = np.array([len(common) >= 2 for common in common_vertices])
        candidates = candidates[~edge_shared]
        common_vertices = [common for common, omitted in zip(common_vertices, edge_shared) if not omitted]
        if not len(candidates):
            continue
        tested += len(candidates)
        base = xyz[first_index]
        other = xyz[candidates]
        repeats = np.broadcast_to(base, other.shape)
        shared_points = np.full((len(candidates), 3), np.nan)
        for local, common in enumerate(common_vertices):
            if len(common) == 1:
                shared_points[local] = vertices[next(iter(common))]
        hits = np.zeros(len(candidates), dtype=bool)
        for edge_index in range(3):
            hits |= _segment_hits_triangle_batch(
                repeats[:, edge_index], repeats[:, (edge_index + 1) % 3], other,
                shared_points, tolerance, shared_contact_tolerance,
            )
            hits |= _segment_hits_triangle_batch(
                other[:, edge_index], other[:, (edge_index + 1) % 3], repeats,
                shared_points, tolerance, shared_contact_tolerance,
            )
        if np.any(hits):
            other_index = int(candidates[np.flatnonzero(hits)[0]])
            return {"valid": False, "face_pair": [int(first_index), other_index],
                    "shared_vertex_count": int(len(set(triangles[first_index]) & set(triangles[other_index]))),
                    "face_vertices": [triangles[first_index].tolist(), triangles[other_index].tolist()],
                    "face_points_mm": [xyz[first_index].round(8).tolist(), xyz[other_index].round(8).tolist()],
                    "tested_pairs": int(tested), "reason": "triangle_penetration"}
        first_normal = np.cross(base[1] - base[0], base[2] - base[0])
        first_norm = float(np.linalg.norm(first_normal))
        other_normals = np.cross(other[:, 1] - other[:, 0], other[:, 2] - other[:, 0])
        parallel = np.linalg.norm(np.cross(other_normals, first_normal), axis=1) <= (
            tolerance * first_norm * np.maximum(np.linalg.norm(other_normals, axis=1), 1e-30)
        )
        plane_distance = np.max(np.abs((other - base[0]) @ first_normal), axis=1)
        coplanar = parallel & (plane_distance <= tolerance * first_norm)
        for local in np.flatnonzero(coplanar):
            if coplanar_overlap(base, other[local]):
                return {"valid": False, "face_pair": [int(first_index), int(candidates[local])],
                        "tested_pairs": int(tested), "reason": "coplanar_overlap"}
    return {"valid": True, "face_pair": None, "tested_pairs": int(tested), "reason": None}


def count_nonincident_mesh_edge_intersections_3d(
    faces: np.ndarray,
    points: np.ndarray,
    *,
    tolerance_mm: float | None = None,
) -> int:
    """Count nonincident mesh-edge intersections using XYZ segment distance."""
    vertices = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    if not len(triangles):
        return 0
    edges = np.unique(
        np.sort(
            np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])),
            axis=1,
        ),
        axis=0,
    )
    starts, ends = vertices[edges[:, 0]], vertices[edges[:, 1]]
    scale = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1.0)
    tolerance = (
        max(1e-8, scale * 1e-10)
        if tolerance_mm is None
        else max(float(tolerance_mm), 0.0)
    )
    low = np.minimum(starts, ends)
    high = np.maximum(starts, ends)
    order = np.argsort(low[:, 0], kind="stable")
    edges, starts, ends, low, high = (value[order] for value in (edges, starts, ends, low, high))
    count = 0
    for index in range(len(edges) - 1):
        stop = int(np.searchsorted(low[:, 0], high[index, 0] + tolerance, side="right"))
        candidates = np.arange(index + 1, stop, dtype=np.int64)
        if not len(candidates):
            continue
        overlap = np.all(low[candidates] <= high[index] + tolerance, axis=1)
        overlap &= np.all(high[candidates] >= low[index] - tolerance, axis=1)
        candidates = candidates[overlap]
        if not len(candidates):
            continue
        other_edges = edges[candidates]
        candidates = candidates[
            ~np.any(other_edges[:, :, None] == edges[index], axis=(1, 2))
        ]
        if not len(candidates):
            continue
        u = ends[index] - starts[index]
        v = ends[candidates] - starts[candidates]
        w = starts[index] - starts[candidates]
        a = float(np.dot(u, u))
        c = np.einsum("ij,ij->i", v, v)
        b = v @ u
        d = w @ u
        e = np.einsum("ij,ij->i", w, v)
        denominator = a * c - b * b
        parallel_epsilon = np.maximum(a * c, 1.0) * 1e-14
        s = np.zeros(len(candidates), dtype=np.float64)
        nonparallel = denominator > parallel_epsilon
        s[nonparallel] = (
            b[nonparallel] * e[nonparallel] - c[nonparallel] * d[nonparallel]
        ) / denominator[nonparallel]
        s = np.clip(s, 0.0, 1.0)
        t = np.clip((b * s + e) / np.maximum(c, 1e-30), 0.0, 1.0)
        s = np.clip((b * t - d) / max(a, 1e-30), 0.0, 1.0)
        t = np.clip((b * s + e) / np.maximum(c, 1e-30), 0.0, 1.0)
        distance = np.linalg.norm(w + s[:, None] * u - t[:, None] * v, axis=1)
        count += int(np.count_nonzero(distance <= tolerance))
    return count


def count_polyline_self_intersections_3d(
    points: np.ndarray,
    *,
    tolerance_mm: float | None = None,
) -> dict:
    """Count non-adjacent closed-polyline segment intersections in 3D.

    Segment pairs are classified by their minimum Euclidean distance in XYZ;
    no planar projection is used. Adjacent segments are omitted because they
    intentionally share their common endpoint.
    """
    ring = np.asarray(points, dtype=np.float64)
    if ring.ndim != 2 or ring.shape[1:] != (3,):
        raise ValueError("3D polyline points must have shape (N, 3)")
    if len(ring) < 3 or not np.isfinite(ring).all():
        return {
            "segment_intersection_count": 0,
            "minimum_nonadjacent_segment_distance_mm": None,
            "tolerance_mm": float(tolerance_mm or 0.0),
            "valid_input": False,
        }

    scale = max(float(np.linalg.norm(np.ptp(ring, axis=0))), 1.0)
    tolerance = (
        max(1e-8, scale * 1e-10)
        if tolerance_mm is None
        else max(float(tolerance_mm), 0.0)
    )
    near_tolerance = max(1e-6, scale * 1e-8)
    starts = ring
    ends = np.roll(ring, -1, axis=0)
    segment_count = len(ring)
    minimum_distance = float("inf")
    intersections: list[dict] = []
    near_contacts: list[dict] = []

    for left_index in range(segment_count):
        # j > i+1 excludes neighboring segments; (0, N-1) is also adjacent.
        right_indices = np.arange(left_index + 2, segment_count, dtype=np.int64)
        if left_index == 0 and len(right_indices):
            right_indices = right_indices[right_indices != segment_count - 1]
        if not len(right_indices):
            continue

        p0 = starts[left_index]
        u = ends[left_index] - p0
        q0 = starts[right_indices]
        v = ends[right_indices] - q0
        w = p0 - q0

        a = float(np.dot(u, u))
        c = np.einsum("ij,ij->i", v, v)
        b = v @ u
        d = w @ u
        e = np.einsum("ij,ij->i", w, v)
        denominator = a * c - b * b
        parallel_epsilon = np.maximum(a * c, 1.0) * 1e-14
        s = np.zeros(len(right_indices), dtype=np.float64)
        nonparallel = denominator > parallel_epsilon
        s[nonparallel] = (
            b[nonparallel] * e[nonparallel] - c[nonparallel] * d[nonparallel]
        ) / denominator[nonparallel]
        s = np.clip(s, 0.0, 1.0)
        t = np.clip((b * s + e) / np.maximum(c, 1e-30), 0.0, 1.0)
        s = np.clip((b * t - d) / max(a, 1e-30), 0.0, 1.0)
        t = np.clip((b * s + e) / np.maximum(c, 1e-30), 0.0, 1.0)

        delta = w + s[:, None] * u - t[:, None] * v
        distances = np.linalg.norm(delta, axis=1)
        local_minimum = float(np.min(distances))
        minimum_distance = min(minimum_distance, local_minimum)
        for offset in np.flatnonzero(distances <= tolerance):
            right_index = int(right_indices[int(offset)])
            left_point = p0 + s[int(offset)] * u
            right_point = q0[int(offset)] + t[int(offset)] * v[int(offset)]
            intersections.append({
                "segment_indices": [int(left_index), right_index],
                "segment_parameters": [float(s[int(offset)]), float(t[int(offset)])],
                "distance_mm": float(distances[int(offset)]),
                "point_mm": ((left_point + right_point) * 0.5).round(8).tolist(),
            })
        for offset in np.flatnonzero(
            (distances > tolerance) & (distances <= near_tolerance)
        ):
            right_index = int(right_indices[int(offset)])
            left_point = p0 + s[int(offset)] * u
            right_point = q0[int(offset)] + t[int(offset)] * v[int(offset)]
            near_contacts.append({
                "segment_indices": [int(left_index), right_index],
                "distance_mm": float(distances[int(offset)]),
                "point_mm": ((left_point + right_point) * 0.5).round(8).tolist(),
            })

    return {
        "segment_intersection_count": int(len(intersections)),
        "minimum_nonadjacent_segment_distance_mm": (
            None if not np.isfinite(minimum_distance) else minimum_distance
        ),
        "tolerance_mm": float(tolerance),
        "near_contact_tolerance_mm": float(near_tolerance),
        "valid_input": True,
        "intersections": intersections[:32],
        "near_contact_count": int(len(near_contacts)),
        "near_contacts": near_contacts[:32],
    }


def remove_true_self_intersections_3d(
    points: np.ndarray,
    vertex_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Remove the smaller loop at each true XYZ crossing of a closed contour.

    Intersections are measured before any interface-plane projection. A pair
    that crosses only in a 2D projection remains untouched. At a true crossing,
    the contour splits into two closed paths through the crossing point; retain
    the longer 3D perimeter path and discard the smaller loop.
    """
    ring = np.asarray(points, dtype=np.float64)
    if ring.ndim != 2 or ring.shape[1:] != (3,):
        raise ValueError("3D polyline points must have shape (N, 3)")
    ids = (
        np.arange(len(ring), dtype=np.int64)
        if vertex_ids is None
        else np.asarray(vertex_ids, dtype=np.int64)
    )
    if ids.shape != (len(ring),):
        raise ValueError("3D polyline vertex IDs must correspond to points")

    removed: list[dict] = []
    max_iterations = max(1, len(ring))
    for _ in range(max_iterations):
        audit = count_polyline_self_intersections_3d(ring)
        if not audit["segment_intersection_count"]:
            break
        crossing = audit["intersections"][0]
        left, right = map(int, crossing["segment_indices"])
        left_t, right_t = map(float, crossing["segment_parameters"])
        intersection = (
            ring[left] * (1.0 - left_t)
            + ring[(left + 1) % len(ring)] * left_t
        )

        def path_between(start_segment: int, stop_segment: int) -> tuple[list[np.ndarray], list[int]]:
            path_points = [intersection]
            path_ids = [-1]
            cursor = (start_segment + 1) % len(ring)
            while cursor != (stop_segment + 1) % len(ring):
                path_points.append(ring[cursor])
                path_ids.append(int(ids[cursor]))
                cursor = (cursor + 1) % len(ring)
            path_points.append(intersection)
            path_ids.append(-1)
            return path_points, path_ids

        first_points, first_ids = path_between(left, right)
        second_points, second_ids = path_between(right, left)

        def perimeter(path: list[np.ndarray]) -> float:
            values = np.asarray(path, dtype=np.float64)
            return float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())

        kept_points, kept_ids = (
            (first_points, first_ids)
            if perimeter(first_points) >= perimeter(second_points)
            else (second_points, second_ids)
        )
        # The repeated endpoint is implicit in a closed polyline.
        ring = np.asarray(kept_points[:-1], dtype=np.float64)
        ids = np.asarray(kept_ids[:-1], dtype=np.int64)
        removed.append({
            "segment_indices": [left, right],
            "discarded_path_perimeter_mm": float(
                min(perimeter(first_points), perimeter(second_points))
            ),
            "kept_path_perimeter_mm": float(
                max(perimeter(first_points), perimeter(second_points))
            ),
            "intersection_point_mm": intersection.round(8).tolist(),
        })
        if len(ring) < 3:
            raise ValueError("3D intersection cleanup left fewer than three boundary points")

    final_audit = count_polyline_self_intersections_3d(ring)
    return ring, ids, {
        "coordinate_space": "world_xyz_mm",
        "removed_crossing_loop_count": len(removed),
        "removed_loops": removed,
        "remaining_intersection_count": int(final_audit["segment_intersection_count"]),
        "remaining_near_contact_count": int(final_audit["near_contact_count"]),
        "projection_only_intersections_removed": 0,
    }
