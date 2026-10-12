"""Clip source triangles at a smooth contour distance field.

The cut crosses source edges at interpolated positions.  It never inherits the
stair-step rim made by removing whole source triangles.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .boundary_matching import distances_to_loop


def contour_distance(vertices: np.ndarray, contour: np.ndarray, radius: float) -> np.ndarray:
    """Measure exact segment distance only near the area to be clipped."""
    vertices = np.asarray(vertices, dtype=np.float64)
    contour = np.asarray(contour, dtype=np.float64)
    nearest, _ = cKDTree(contour).query(vertices)
    longest_edge = float(np.linalg.norm(contour - np.roll(contour, -1, axis=0), axis=1).max())
    nearby = nearest <= radius + longest_edge
    result = nearest.copy()
    if np.any(nearby):
        result[nearby] = distances_to_loop(vertices[nearby], contour)
    return result


def clip_source_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    contours: list[np.ndarray],
    widths: list[float],
    source_boundary_loops: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Retain the exterior of every contour, splitting intersected triangles.

    Output face origins carry the source triangle's paint and material to each
    clipped triangle.  Original source vertex positions are never changed.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if (len(contours) != len(widths) or len(contours) != len(source_boundary_loops)
            or np.any(np.asarray(widths) < 0)):
        raise ValueError("each contour needs one nonnegative width and source boundary loop")
    if not contours or not any(width > 0 for width in widths):
        return vertices, faces, np.arange(len(faces), dtype=np.int64), {
            "source_faces_split": 0, "source_faces_removed": 0,
            "new_cut_vertices": 0,
        }

    field = np.full(len(vertices), np.inf, dtype=np.float64)
    for contour, width in zip(contours, widths):
        if width > 0:
            field = np.minimum(field, contour_distance(vertices, contour, width) - width)

    # Spatial proximity alone may also select a visible surface across a fold.
    # Only the band connected along source mesh edges to an interface opening is
    # allowed to be replaced; an unrelated nearby disk must stay intact.
    inside = field < 0.0
    inside_edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    inside_edges = inside_edges[np.all(inside[inside_edges], axis=1)]
    graph = coo_matrix((
        np.ones(2 * len(inside_edges), dtype=np.int8),
        (np.concatenate((inside_edges[:, 0], inside_edges[:, 1])),
         np.concatenate((inside_edges[:, 1], inside_edges[:, 0]))),
    ), shape=(len(vertices), len(vertices))).tocsr()
    _, labels = connected_components(graph, directed=False)
    seeds = np.concatenate([
        np.asarray(loop, dtype=np.int64)[inside[np.asarray(loop, dtype=np.int64)]]
        for loop, width in zip(source_boundary_loops, widths) if width > 0
    ])
    if len(seeds):
        seeded_labels = np.unique(labels[seeds])
        connected_band = inside & np.isin(labels, seeded_labels)
        field[inside & ~connected_band] *= -1.0

    outside = field >= 0.0
    flags = outside[faces]
    full = np.all(flags, axis=1)
    partial = np.any(flags, axis=1) & ~full
    kept_faces = faces[full].tolist()
    origins = np.flatnonzero(full).tolist()
    points = vertices.tolist()
    edge_crossings: dict[tuple[int, int], int] = {}

    def crossing(first: int, second: int) -> int:
        key = (min(first, second), max(first, second))
        if key not in edge_crossings:
            first_value, second_value = field[first], field[second]
            fraction = first_value / (first_value - second_value)
            point = vertices[first] + fraction * (vertices[second] - vertices[first])
            edge_crossings[key] = len(points)
            points.append(point.tolist())
        return edge_crossings[key]

    for face_index in np.flatnonzero(partial):
        triangle = [int(vertex) for vertex in faces[face_index]]
        polygon: list[int] = []
        for position, first in enumerate(triangle):
            second = triangle[(position + 1) % 3]
            first_outside = bool(outside[first])
            second_outside = bool(outside[second])
            if first_outside and second_outside:
                polygon.append(second)
            elif first_outside and not second_outside:
                polygon.append(crossing(first, second))
            elif not first_outside and second_outside:
                polygon.extend((crossing(first, second), second))
        for position in range(1, len(polygon) - 1):
            kept_faces.append([polygon[0], polygon[position], polygon[position + 1]])
            origins.append(int(face_index))

    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(kept_faces, dtype=np.int64).reshape((-1, 3)),
        np.asarray(origins, dtype=np.int64),
        {
            "source_faces_split": int(np.count_nonzero(partial)),
            "source_faces_removed": int(np.count_nonzero(~np.any(flags, axis=1))),
            "new_cut_vertices": int(len(edge_crossings)),
        },
    )
