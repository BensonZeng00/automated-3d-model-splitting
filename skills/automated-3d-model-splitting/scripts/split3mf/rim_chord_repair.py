"""Separate generated backing chords from internal source-surface edges."""
from collections import Counter

import numpy as np


def _edges(faces):
    return Counter(tuple(sorted((int(a), int(b)))) for face in faces
                   for a, b in zip(face, face[1:] + face[:1]))


def _split_face(face, midpoints):
    a, b, c = face
    ab, bc, ca = (midpoints.get(tuple(sorted(edge)))
                  for edge in ((a, b), (b, c), (c, a)))
    count = sum(i is not None for i in (ab, bc, ca))
    if count == 0:
        return [list(face)]
    if count == 1:
        if ab is not None:
            return [[a, ab, c], [ab, b, c]]
        if bc is not None:
            return [[b, bc, a], [bc, c, a]]
        return [[c, ca, b], [ca, a, b]]
    if count == 2:
        if ab is not None and bc is not None:
            return [[b, bc, ab], [a, ab, c], [ab, bc, c]]
        if bc is not None and ca is not None:
            return [[c, ca, bc], [b, bc, a], [bc, ca, a]]
        return [[a, ab, ca], [c, ca, b], [ca, ab, b]]
    return [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]


def separate_source_chords(vertices, faces, face_start, outer_ids, inward, depth_mm):
    """Split co-owned interior edges with a small inward collar.

    A constrained annulus may contain triangles joining three original rim
    vertices. If their diagonal is also internal to the original surface,
    sewing by vertex identity creates a four-owner edge. Only those generated
    edges are split. Every source coordinate, source face and actual rim edge
    is immutable; conforming edge subdivision retains projected coverage.
    """
    selected = faces[face_start:]
    source_edges, generated_edges = _edges(faces[:face_start]), _edges(selected)
    rim = {tuple(sorted((int(a), int(b)))) for a, b in
           zip(outer_ids, outer_ids[1:] + outer_ids[:1])}
    conflicts = sorted(edge for edge, count in generated_edges.items()
                       if count == 2 and source_edges[edge] and edge not in rim)
    if not conflicts:
        return {'split_source_chords': 0, 'added_faces': 0, 'inward_offset_mm': 0.0}
    axis = np.asarray(inward, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    offset = min(0.05, max(float(depth_mm), 0.) * 0.1)
    if offset <= 1e-9:
        raise ValueError('source chord repair has no inward depth budget')
    midpoints = {}
    for edge in conflicts:
        midpoints[edge] = len(vertices)
        vertices.append((np.asarray(vertices[edge[0]]) + vertices[edge[1]])*.5 + axis*offset)
    repaired = []
    for face in selected:
        repaired.extend(_split_face(face, midpoints))
    faces[face_start:] = repaired
    return {'split_source_chords': len(conflicts),
            'added_faces': len(repaired)-len(selected), 'inward_offset_mm': offset}
