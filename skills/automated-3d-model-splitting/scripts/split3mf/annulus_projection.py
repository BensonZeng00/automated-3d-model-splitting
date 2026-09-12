"""Geometric coverage checks shared by all connector strip strategies."""
from dataclasses import dataclass

import numpy as np


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _area(points):
    points = points - points[0]
    return float(_cross(points, np.roll(points, -1, axis=0)).sum() * 0.5)


def _crossing_count(faces, points):
    """Count proper nonincident edge crossings using a bounding-box sweep."""
    edges = np.asarray(sorted({tuple(sorted((int(a), int(b))))
                              for face in faces
                              for a, b in zip(face, np.roll(face, -1))}))
    if not len(edges):
        return 0
    segments = np.asarray([[points[a], points[b]] for a, b in edges])
    low, high = segments.min(axis=1), segments.max(axis=1)
    order = np.argsort(low[:, 0], kind='stable')
    edges, segments, low, high = (x[order] for x in (edges, segments, low, high))
    count = 0
    for i in range(len(edges) - 1):
        stop = np.searchsorted(low[:, 0], high[i, 0], side='right')
        candidates = np.arange(i + 1, stop)
        candidates = candidates[(low[candidates, 1] < high[i, 1])
                                & (high[candidates, 1] > low[i, 1])]
        if not len(candidates):
            continue
        other = edges[candidates]
        candidates = candidates[~np.any(other[:, :, None] == edges[i], axis=(1, 2))]
        a, b = segments[i]
        c, d = segments[candidates, 0], segments[candidates, 1]
        count += int(np.count_nonzero(
            (_cross(b-a, c-a) * _cross(b-a, d-a) < -1e-20)
            & (_cross(d-c, a-c) * _cross(d-c, b-c) < -1e-20)))
    return count


@dataclass(frozen=True)
class AnnulusProjectionAudit:
    valid: bool
    face_count: int
    reversed_faces: int
    absolute_area_mm2: float
    expected_area_mm2: float
    excess_area_mm2: float
    crossing_edges: int | None
    reason: str


def audit_projection(faces, points, outer_ids, inner_ids):
    """Require one oriented, noncrossing cover of the projected annulus.

    Accept either global winding direction, never a mixture. This is a
    structural audit; cosmetic resolution and print-area budgets do not relax
    it. A caller restoring proven source projection ears audits its simple
    core separately, while retaining the complete strip's 3-D topology audit.
    """
    triangles = np.asarray([[points[int(i)] for i in f] for f in faces])
    expected = abs(_area(np.asarray([points[i] for i in outer_ids]))) - abs(
        _area(np.asarray([points[i] for i in inner_ids])))
    if not len(triangles):
        return AnnulusProjectionAudit(False, 0, 0, 0., expected, 0., None, 'no_faces')
    areas = _cross(triangles[:, 1]-triangles[:, 0],
                   triangles[:, 2]-triangles[:, 0]) * 0.5
    tolerance = max(1e-7, abs(expected) * 1e-8)
    direction = 1. if areas.sum() >= 0. else -1.
    reversed_areas = -areas[areas * direction < 0.] * direction
    total = float(np.abs(areas).sum())
    excess = float(total - abs(areas.sum()))
    reasons = []
    if not np.all(np.isfinite(areas)) or expected <= 0.:
        reasons.append('invalid_projected_domain')
    if reversed_areas.sum() > tolerance:
        reasons.append('projected_fold')
    if abs(total - expected) > tolerance:
        reasons.append('projected_area_mismatch')
    # Fail cheap invalid candidates before the complete edge-intersection pass.
    crossings = None if reasons else _crossing_count(faces, points)
    if crossings:
        reasons.append('projected_edge_crossing')
    return AnnulusProjectionAudit(not reasons, len(faces), len(reversed_areas),
                                  total, expected, excess, crossings, ','.join(reasons))
