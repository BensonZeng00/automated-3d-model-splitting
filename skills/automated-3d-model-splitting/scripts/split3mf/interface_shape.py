"""Strict scale-aware detector for genuinely curved 3D contact rims."""
from __future__ import annotations

import numpy as np


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, fraction: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, fraction * cumulative[-1])
    return float(values[order[min(index, len(order) - 1)]])


def classify_interface_shape(points: np.ndarray) -> dict:
    """Route only conspicuous, sustained out-of-plane bows as curved.

    Fit a plane using physical arc-length weights, trim the most deviant 10%
    once, and measure the 90th-percentile sagitta against the smaller in-plane
    span. Curved requires both at least 1 mm of sagitta and 20% of that span,
    with deviation distributed over at least 80% of the rim. Thus small local
    waviness, planar lips, and broad but shallow body contours remain planar.
    """
    values = np.asarray(points, dtype=np.float64)
    edges = np.linalg.norm(np.roll(values, -1, axis=0) - values, axis=1)
    weights = (edges + np.roll(edges, 1)) * 0.5
    if (values.ndim != 2 or values.shape[1:] != (3,) or len(values) < 3
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or weights.sum() <= 1e-12):
        raise ValueError("interface shape requires a finite nonzero closed rim")

    def fit(active_weights: np.ndarray):
        center = np.average(values, axis=0, weights=active_weights)
        delta = values - center
        _, vectors = np.linalg.eigh((delta * active_weights[:, None]).T @ delta)
        return center, vectors

    center, vectors = fit(weights)
    residuals = np.abs((values - center) @ vectors[:, 0])
    keep = residuals <= _weighted_quantile(residuals, weights, 0.90)
    if np.count_nonzero(keep) >= 3:
        center, vectors = fit(weights * keep)

    delta = values - center
    signed = delta @ vectors[:, 0]
    distances = np.abs(signed)
    spans = np.ptp(delta @ vectors[:, 1:], axis=0)
    reference_span = float(np.min(spans))
    tolerance = max(0.15, 0.01 * reference_span)
    p90 = _weighted_quantile(distances, weights, 0.90)
    active = distances > tolerance
    total = float(weights.sum())
    active_fraction = float(weights[active].sum() / total)
    relative_sagitta = p90 / max(reference_span, 1e-12)
    curved = bool(
        p90 >= 1.0
        and relative_sagitta >= 0.20
        and active_fraction >= 0.80
    )
    return {
        "classification": "curved" if curved else "planar",
        "method": "scale_normalized_sustained_sagitta",
        "normal": vectors[:, 0].tolist(),
        "reference_span_mm": reference_span,
        "distance_tolerance_mm": tolerance,
        "distance_p90_mm": p90,
        "distance_max_mm": float(np.max(distances)),
        "outlier_arc_fraction": active_fraction,
        "relative_sagitta_p90": relative_sagitta,
        "minimum_sagitta_mm": 1.0,
        "minimum_relative_sagitta": 0.20,
        "minimum_deviating_arc_fraction": 0.80,
        "sample_count": len(values),
    }
