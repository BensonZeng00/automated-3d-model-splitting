"""Choose a cap translation from measured depth; always remeasure afterwards."""
import numpy as np


def measured_backoff(displacement, axis, safe_depth, minimum_depth, excess):
    axial = np.asarray(displacement) @ axis
    headroom = float(axial.min())-minimum_depth
    if headroom <= 1e-7:
        return 0.0, "no_positive_depth_headroom"
    # Intersect ||displacement_i - t*axis|| <= safe_depth - 0.01 for all
    # boundary vertices. This solves the current measurement, not future rays.
    target = max(float(safe_depth)-0.01, 0.0)
    lateral_sq = np.maximum(np.sum(np.asarray(displacement)**2, axis=1)-axial**2, 0)
    radicand = target*target-lateral_sq
    if target > minimum_depth and np.all(radicand >= 0):
        radius = np.sqrt(radicand)
        lower = max(float((axial-radius).max()), 0.0)
        upper = min(float((axial+radius).min()), headroom-1e-7)
        if 0 < lower <= upper:
            return lower, "measured_quadratic_interval"
    return min(max(float(excess)+0.01, 0.01), headroom*0.5), "bounded_half_headroom"
