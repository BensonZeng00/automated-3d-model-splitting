"""Bounded source-loop correspondence without changing either visible rim."""
import numpy as np
from .print_tolerance import current


def _project(points, loop):
    vectors = np.roll(loop, -1, axis=0) - loop
    squared = np.sum(vectors*vectors, axis=1)
    errors, phases = [], []
    for start in range(0, len(points), 128):
        delta = points[start:start+128, None, :] - loop[None, :, :]
        parameters = np.clip(np.divide(
            np.sum(delta*vectors, axis=2), squared,
            out=np.zeros(delta.shape[:2]), where=squared > 1e-24), 0, 1)
        distance = np.linalg.norm(delta-parameters[:, :, None]*vectors, axis=2)
        ids = np.argmin(distance, axis=1)
        rows = np.arange(len(ids))
        errors.extend(distance[rows, ids])
        phases.extend((ids + parameters[rows, ids]) % len(loop))
    return np.asarray(errors), np.asarray(phases)


def source_loop_print_match(source, requested, requested_limit):
    """Allow print-scale rounding only for a single ordered traversal.

    This is a fallback proof, not a replacement for exact source IDs. It
    leaves the prevalidated cap field and both source coordinate arrays intact.
    """
    limit = min(float(current().surface_distance_mm), 0.05)
    if current().micro_area_mm2 <= 0 or limit <= requested_limit:
        return None
    source = np.asarray(source, dtype=float)
    requested = np.asarray(requested, dtype=float)
    if min(len(source), len(requested)) < 3 or not (
            np.isfinite(source).all() and np.isfinite(requested).all()):
        return None
    errors, phases = _project(requested, source)
    reverse_errors, _ = _project(source, requested)
    maximum = max(float(errors.max()), float(reverse_errors.max()))
    if maximum > limit:
        return None
    # One traversal in either direction; prohibit reordered/crossing shortcuts.
    increments = np.diff(np.r_[phases, phases[0]])
    period = len(source)
    ordered = any(abs(float(np.mod(sign*increments, period).sum())-period) <= 1e-7
                  for sign in (1, -1))
    if not ordered:
        return None
    # Mid-edge probes detect a chord skipping a curved source section.
    for first, second in ((source, requested), (requested, source)):
        samples = np.concatenate([(1-t)*first+t*np.roll(first, -1, axis=0)
                                  for t in (.25, .5, .75)])
        sampled, _ = _project(samples, second)
        maximum = max(maximum, float(sampled.max()))
    if maximum > limit:
        return None
    return dict(method="ordered_bidirectional_source_loop_samples",
                tolerance_mm=limit, maximum_sampled_error_mm=maximum,
                visible_vertices_moved=0)
