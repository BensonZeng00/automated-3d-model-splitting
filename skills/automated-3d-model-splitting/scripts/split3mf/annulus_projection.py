"""Geometric coverage checks shared by all connector strip strategies."""
from dataclasses import dataclass

import numpy as np

def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _area(points):
    points = points - points[0]
    return float(_cross(points, np.roll(points, -1, axis=0)).sum() * 0.5)


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


def audit_projection(
    faces,
    points,
    outer_ids,
    inner_ids,
):
    """Measure projected area and winding consistency for a connector strip.

    Accept either global winding direction, never a mixture. This is a
    structural diagnostic. It deliberately does not search for projected
    edge crossings; generated geometry is checked by the 3D audit instead.
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
    crossings = None
    return AnnulusProjectionAudit(not reasons, len(faces), len(reversed_areas),
                                  total, expected, excess, crossings, ','.join(reasons))
