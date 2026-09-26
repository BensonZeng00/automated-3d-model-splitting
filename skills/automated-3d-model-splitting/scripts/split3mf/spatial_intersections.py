from __future__ import annotations

import numpy as np


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
