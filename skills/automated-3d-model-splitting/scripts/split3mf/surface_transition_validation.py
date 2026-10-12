"""Geometry checks for a newly triangulated interface transition."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _segments_hit_triangles(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Vectorized segment-triangle intersections for corresponding triangle pairs."""
    edge_a = second[:, 1] - second[:, 0]
    edge_b = second[:, 2] - second[:, 0]
    hits = np.zeros(len(first), dtype=bool)
    for corner in range(3):
        start = first[:, corner]
        direction = first[:, (corner + 1) % 3] - start
        cross = np.cross(direction, edge_b)
        determinant = np.einsum("ij,ij->i", edge_a, cross)
        active = np.abs(determinant) > 1e-12
        inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=active)
        relative = start - second[:, 0]
        u = np.einsum("ij,ij->i", relative, cross) * inverse
        q = np.cross(relative, edge_a)
        v = np.einsum("ij,ij->i", direction, q) * inverse
        t = np.einsum("ij,ij->i", edge_b, q) * inverse
        hits |= active & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
    return hits


def transition_self_intersection_count(
    vertices: np.ndarray, faces: np.ndarray,
) -> int:
    """Count nonadjacent crossings among transition triangles.

    Source and target surfaces are checked separately by the complete-part
    validation. This guards the new ribbon against folding through itself.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) < 2:
        return 0
    triangles = vertices[faces]
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None, :], axis=2).max(axis=1)
    pairs = np.asarray(list(cKDTree(centers).query_pairs(2.0 * float(radii.max()))), dtype=np.int64)
    if not len(pairs):
        return 0
    pairs = pairs.reshape((-1, 2))
    left, right = pairs[:, 0], pairs[:, 1]
    close = np.linalg.norm(centers[left] - centers[right], axis=1) <= radii[left] + radii[right]
    pairs = pairs[close]
    if not len(pairs):
        return 0
    left, right = pairs[:, 0], pairs[:, 1]
    distinct = ~np.any(faces[left, :, None] == faces[right, None, :], axis=(1, 2))
    bounds_overlap = np.all(
        (triangles[left].min(axis=1) <= triangles[right].max(axis=1)) &
        (triangles[right].min(axis=1) <= triangles[left].max(axis=1)), axis=1,
    )
    pairs = pairs[distinct & bounds_overlap]
    count = 0
    for block in np.array_split(pairs, max(1, (len(pairs) + 19999) // 20000)):
        if not len(block):
            continue
        first = triangles[block[:, 0]]
        second = triangles[block[:, 1]]
        count += int(np.count_nonzero(
            _segments_hit_triangles(first, second) | _segments_hit_triangles(second, first)
        ))
    return count
