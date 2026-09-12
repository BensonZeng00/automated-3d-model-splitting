"""Physical review candidates independent of tessellation density.

Hydraulic width is a screening estimate, not a measured minimum wall width.
It only requests human review; it never deletes or merges a painted feature.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StripReviewPolicy:
    maximum_width_estimate_mm: float = 1.2
    minimum_elongation: float = 12.0
    minimum_span_mm: float = 10.0


def strip_evidence(vertices, faces, group, visible_faces=None,
                   policy=StripReviewPolicy()):
    ids = np.asarray(group, dtype=np.int64)
    if visible_faces is not None:
        ids = ids[np.asarray(visible_faces, dtype=bool)[ids]]
    if not len(ids):
        return {"long_thin_candidate": False, "visible_area_mm2": 0.0}
    selected = np.asarray(faces)[ids]
    triangles = np.asarray(vertices)[selected]
    area = float(np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                        triangles[:, 2]-triangles[:, 0]), axis=1).sum()/2)
    edges = np.sort(np.concatenate((selected[:, [0, 1]], selected[:, [1, 2]],
                                    selected[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    perimeter = float(np.linalg.norm(np.asarray(vertices)[boundary[:, 0]]
                                    - np.asarray(vertices)[boundary[:, 1]], axis=1).sum())
    span = float(np.linalg.norm(np.ptp(triangles.reshape(-1, 3), axis=0)))
    width = 2*area/perimeter if perimeter > 0 else None
    elongation = perimeter*perimeter/(4*area) if area > 0 else 0.0
    candidate = (width is not None and 0 < width <= policy.maximum_width_estimate_mm
                 and elongation >= policy.minimum_elongation
                 and span >= policy.minimum_span_mm)
    return dict(long_thin_candidate=bool(candidate), visible_area_mm2=area,
                visible_face_count=len(ids), boundary_length_mm=perimeter,
                width_estimate_mm=width, elongation_estimate=elongation,
                bbox_diagonal_mm=span, measurement="visible_surface_2A_over_perimeter",
                thresholds=policy.__dict__)


def partition_review_groups(vertices, faces, groups, min_faces,
                            auto_noise_max_faces, visible_faces=None):
    """Long strips require review even above the ordinary face-count ceiling."""
    effective, automatic, review = [], [], []
    for raw in groups:
        group = np.asarray(raw, dtype=np.int64)
        evidence = strip_evidence(vertices, faces, group, visible_faces)
        if evidence["long_thin_candidate"] or auto_noise_max_faces < len(group) < min_faces:
            review.append(group)
        elif len(group) < min_faces and len(group) <= auto_noise_max_faces:
            automatic.append(group)
        else:
            effective.append(group)
    return effective, automatic, review
