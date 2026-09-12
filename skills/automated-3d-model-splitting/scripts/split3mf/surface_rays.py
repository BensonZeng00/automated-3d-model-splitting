"""Deterministic parallel ray/triangle depth queries without an optional R-tree."""
from __future__ import annotations

import numpy as np


def first_surface_depth(mesh, points, direction):
    return first_surface_hit(mesh, points, direction)[0]


def first_surface_hit(mesh, points, direction):
    """Return signed front-hit travel relative to each query point.

    Rays enter from outside the complete bounding box. Both triangle sides are
    included: a reversed membrane still occludes an insert in a slicer viewport.
    The projected batch bounds reject unrelated triangles before exact tests.
    """
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if not len(points):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    axis = np.asarray(direction, dtype=np.float64)
    length = np.linalg.norm(axis)
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError('surface ray direction must be finite and nonzero')
    axis = axis / length
    seed = np.eye(3)[int(np.argmin(np.abs(axis)))]
    u = np.cross(axis, seed); u /= np.linalg.norm(u)
    basis = np.column_stack((u, np.cross(axis, u), axis))
    projected = points @ basis
    triangles = np.asarray(mesh.triangles) @ basis
    keep = (
        np.all(triangles[:, :, :2].max(axis=1) >= projected[:, :2].min(axis=0), axis=1)
        & np.all(triangles[:, :, :2].min(axis=1) <= projected[:, :2].max(axis=0), axis=1)
    )
    triangles = triangles[keep]
    face_ids = np.flatnonzero(keep)
    a = triangles[:, 0]
    e, g = triangles[:, 1] - a, triangles[:, 2] - a
    determinant = e[:, 0] * g[:, 1] - e[:, 1] * g[:, 0]
    usable = np.abs(determinant) > 1e-12
    a, e, g, determinant = a[usable], e[usable], g[usable], determinant[usable]
    face_ids = face_ids[usable]
    depths = np.full(len(points), np.inf)
    hits = np.full(len(points), -1, dtype=np.int64)
    for index, point in enumerate(projected):
        delta = point - a
        s = (delta[:, 0] * g[:, 1] - delta[:, 1] * g[:, 0]) / determinant
        t = (e[:, 0] * delta[:, 1] - e[:, 1] * delta[:, 0]) / determinant
        inside = (s >= -1e-10) & (t >= -1e-10) & (s + t <= 1 + 1e-10)
        if np.any(inside):
            z = a[:, 2] + s * e[:, 2] + t * g[:, 2] - point[2]
            selected = np.flatnonzero(inside)[np.argmin(z[inside])]
            depths[index] = float(z[selected])
            hits[index] = face_ids[selected]
    return depths, hits
