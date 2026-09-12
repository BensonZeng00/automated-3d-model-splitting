"""Measure within-layer creases separately from intentional ring transitions."""
from collections import defaultdict

import numpy as np


def internal_dihedral_report(vertices, faces, perimeter_edges=()):
    """Report geometry without interpreting every sharp design corner as damage.

    Pairwise edge directions normalize normals for this measurement only;
    the independent winding audit remains responsible for mesh orientation.
    """
    faces = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths, 1e-30)[:, None]
    owners = defaultdict(list)
    perimeter = set(perimeter_edges)
    for i, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            edge = tuple(sorted((int(a), int(b))))
            if edge not in perimeter:
                owners[edge].append((i, int(a) < int(b)))
    angles = []
    for pair in owners.values():
        if len(pair) != 2:
            continue
        (a, da), (b, db) = pair
        if min(lengths[a], lengths[b]) <= 1e-12:
            continue
        cosine = float(np.dot(normals[a], normals[b])) * (1 if da != db else -1)
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
    return {
        'status': 'measured_advisory' if angles else 'not_evaluated',
        'checked_edges': len(angles),
        'p95_degrees': float(np.percentile(angles, 95)) if angles else None,
        'maximum_degrees': max(angles) if angles else None,
        'over_60_degrees': sum(a > 60 for a in angles),
        'interpretation': 'sharp design corners require context; projection folds are blocking',
    }
