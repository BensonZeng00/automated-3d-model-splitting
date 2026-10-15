"""Periodic smoothing of a closed 3-D contour using physical arc length."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

MAX_RELATIVE_SHAPE_CHANGE = 0.10


def smooth_closed_loop_by_arclength(points: np.ndarray, sigma_mm: float) -> tuple[np.ndarray, dict]:
    """Low-pass a contour in millimetres without assuming uniform input samples."""
    original = np.asarray(points, dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != 3 or len(original) < 3:
        raise ValueError("closed boundary needs at least three 3-D points")
    if not np.isfinite(original).all() or not np.isfinite(sigma_mm) or sigma_mm <= 0:
        raise ValueError("closed boundary and smoothing width must be finite and positive")
    edges = np.linalg.norm(np.roll(original, -1, axis=0) - original, axis=1)
    if np.any(edges <= 1e-9):
        raise ValueError("closed boundary has coincident consecutive points")
    distances = np.r_[0.0, np.cumsum(edges)]
    perimeter = float(distances[-1])
    sample_count = max(len(original), int(np.ceil(perimeter / 0.25)))
    positions = np.linspace(0.0, perimeter, sample_count, endpoint=False)
    extended = np.vstack((original, original[:1]))
    uniform = np.column_stack([
        np.interp(positions, distances, extended[:, axis]) for axis in range(3)
    ])
    filtered = gaussian_filter1d(
        uniform, sigma_mm / (perimeter / sample_count), axis=0, mode="wrap"
    )
    filtered_extended = np.vstack((filtered, filtered[:1]))
    result = np.column_stack([
        np.interp(distances[:-1], np.r_[positions, perimeter], filtered_extended[:, axis])
        for axis in range(3)
    ])
    displacement = np.linalg.norm(result - original, axis=1)
    return result, {
        "algorithm": "periodic_gaussian_uniform_arc_length",
        "sigma_mm": float(sigma_mm),
        "perimeter_mm": perimeter,
        "sample_count": sample_count,
        "maximum_displacement_mm": float(displacement.max()),
        "p95_displacement_mm": float(np.quantile(displacement, 0.95)),
    }
