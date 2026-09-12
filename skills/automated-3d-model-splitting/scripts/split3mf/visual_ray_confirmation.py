"""Narrow-phase confirmation of sparse visual intrusion candidates."""
import numpy as np

from .surface_rays import first_surface_hit, first_surface_depth


def confirm_front_intrusions(source, assembled, points, source_labels, direction,
                             depth_tolerance_mm):
    """Confirm sampled surface points, preserving the caller's depth limit.

    Missing source intersections are not evidence for dismissing a candidate.
    The assembly query includes both sides, so even a reversed occluding sheet
    prevents a hidden candidate from being advertised as visible.
    """
    source_travel, source_hit = first_surface_hit(source, points, -direction)
    assembled_travel = first_surface_depth(assembled, points, -direction)
    hidden = assembled_travel < -1e-6
    within_source = (source_hit >= 0) & (source_travel <= max(float(depth_tolerance_mm), 0.0))
    confirmed = ~hidden & ~within_source
    labels = np.zeros(len(points), dtype=np.int32)
    labels[source_hit >= 0] = np.asarray(source_labels)[source_hit[source_hit >= 0]]
    return confirmed, labels, {
        'candidate_pixels': len(points),
        'rejected_sparse_sample_pixels': int((~confirmed).sum()),
        'occluded_by_assembly': int(hidden.sum()),
        'within_source_depth_limit': int(within_source.sum()),
        'missing_source_hits': int((source_hit < 0).sum()),
    }
