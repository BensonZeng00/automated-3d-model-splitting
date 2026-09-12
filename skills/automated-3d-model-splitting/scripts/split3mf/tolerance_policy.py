"""Explicit user policy for warnings and small-region ownership.

These thresholds never certify a mesh: generated topology, winding and Boolean
checks remain separate. Ratios use the current interface, not the whole model.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class TolerancePolicy:
    warning_ratio: float = 0.01
    micro_region_span_mm: float = 2.0
    transition_band_mm: float = 3.0

    def warning(self, count, total):
        if total < 0 or count < 0 or count > total:
            raise ValueError('Invalid numerator or denominator')
        return total > 0 and count > self.warning_ratio * total


def maximum_span(points, threshold=2.0):
    """Exact Euclidean diameter for small candidates, bounded-memory blocks."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    lower = float(np.ptp(points, axis=0).max())
    if lower > threshold:
        return lower  # Sufficient rejection; do not claim an exact diameter.
    largest = 0.0
    for start in range(0, len(points), 256):
        largest = max(largest, float(cdist(points[start:start + 256], points).max()))
        if largest > threshold:
            break
    return largest


def slope_warning(outside_count, sample_count):
    policy = TolerancePolicy()
    return dict(warning=policy.warning(outside_count, sample_count),
                evaluable=bool(sample_count),
                count=int(outside_count), total=int(sample_count),
                ratio=float(outside_count / sample_count) if sample_count else 0.0,
                threshold=policy.warning_ratio, comparison='strictly-greater-than',
                action='warning-only')
