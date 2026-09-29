"""Match independently sampled copies of one frozen simplified seam."""

from __future__ import annotations

import numpy as np


MINIMUM_BOUNDARY_COVERAGE = 0.80


def distances_to_loop(points: np.ndarray, loop: np.ndarray) -> np.ndarray:
    """Measure each point against a closed contour's segments, not just vertices."""
    starts = loop
    edges = np.roll(loop, -1, axis=0) - starts
    edge_lengths_squared = np.einsum("ij,ij->i", edges, edges)
    result = np.empty(len(points), dtype=np.float64)
    # Bound the temporary point/segment matrix for dense source contours.
    batch_size = max(1, 200_000 // len(loop))
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        offsets = batch[:, None, :] - starts[None, :, :]
        fractions = np.divide(
            np.einsum("ijk,jk->ij", offsets, edges),
            edge_lengths_squared[None, :],
            out=np.zeros((len(batch), len(loop)), dtype=np.float64),
            where=edge_lengths_squared[None, :] > 1e-18,
        )
        closest = starts[None, :, :] + np.clip(fractions, 0.0, 1.0)[:, :, None] * edges[None, :, :]
        result[start : start + len(batch)] = np.sqrt(
            np.min(np.sum((batch[:, None, :] - closest) ** 2, axis=2), axis=1)
        )
    return result


def distances_to_segments(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Measure points against an unordered collection of 3D line segments."""
    points = np.asarray(points, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not len(segments):
        raise ValueError("segments must be a nonempty Mx2x3 array")
    starts = segments[:, 0]
    edges = segments[:, 1] - starts
    edge_lengths_squared = np.einsum("ij,ij->i", edges, edges)
    result = np.empty(len(points), dtype=np.float64)
    # Keep each point/segment distance block bounded for dense host boundaries.
    batch_size = max(1, 200_000 // len(segments))
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        offsets = batch[:, None, :] - starts[None, :, :]
        fractions = np.divide(
            np.einsum("ijk,jk->ij", offsets, edges),
            edge_lengths_squared[None, :],
            out=np.zeros((len(batch), len(segments)), dtype=np.float64),
            where=edge_lengths_squared[None, :] > 1e-18,
        )
        closest = starts[None, :, :] + np.clip(fractions, 0.0, 1.0)[:, :, None] * edges[None, :, :]
        result[start : start + len(batch)] = np.sqrt(
            np.min(np.sum((batch[:, None, :] - closest) ** 2, axis=2), axis=1)
        )
    return result


def _distance_summary(distances: np.ndarray, tolerance: float) -> dict:
    distances = np.asarray(distances, dtype=np.float64)
    return {
        "coverage": float(np.mean(distances <= tolerance)),
        "p50_distance_mm": float(np.quantile(distances, 0.50)),
        "p90_distance_mm": float(np.quantile(distances, 0.90)),
        "maximum_distance_mm": float(distances.max()),
    }


def _loop_segments(loop: np.ndarray) -> np.ndarray:
    return np.stack((loop, np.roll(loop, -1, axis=0)), axis=1)


def compare_boundary_loops(left: np.ndarray, right: np.ndarray) -> dict | None:
    """Return symmetric coverage and distance evidence for two closed contours."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(left) < 3 or len(right) < 3:
        return None
    left_edges = np.linalg.norm(np.roll(left, -1, axis=0) - left, axis=1)
    right_edges = np.linalg.norm(np.roll(right, -1, axis=0) - right, axis=1)
    sample_spacing = min(float(np.median(left_edges)), float(np.median(right_edges)))
    span = min(float(np.ptp(left, axis=0).max()), float(np.ptp(right, axis=0).max()))
    tolerance = min(0.60, max(0.08, 0.30 * sample_spacing, 0.06 * span))
    left_center = (left.min(axis=0) + left.max(axis=0)) * 0.5
    right_center = (right.min(axis=0) + right.max(axis=0)) * 0.5
    center_distance = float(np.linalg.norm(left_center - right_center))
    if center_distance > max(0.10, 0.03 * span):
        return None
    if np.any(left.min(axis=0) > right.max(axis=0) + tolerance):
        return None
    if np.any(right.min(axis=0) > left.max(axis=0) + tolerance):
        return None
    left_to_right = _distance_summary(
        distances_to_loop(left, right), tolerance
    )
    right_to_left = _distance_summary(
        distances_to_loop(right, left), tolerance
    )
    return {
        "distance_tolerance_mm": tolerance,
        "center_distance_mm": center_distance,
        "left_to_right": left_to_right,
        "right_to_left": right_to_left,
        "matched": (
            left_to_right["coverage"] >= MINIMUM_BOUNDARY_COVERAGE
            and right_to_left["coverage"] >= MINIMUM_BOUNDARY_COVERAGE
        ),
    }


def compare_loop_to_boundary_segments(
    loop: np.ndarray, segments: np.ndarray, tolerance: float
) -> dict | None:
    """Compare a closed loop with fragmented host edges using symmetric coverage."""
    loop = np.asarray(loop, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.float64)
    if len(loop) < 3 or segments.ndim != 3 or segments.shape[1:] != (2, 3) or not len(segments):
        return None
    lower = loop.min(axis=0) - tolerance
    upper = loop.max(axis=0) + tolerance
    local = np.all(segments.max(axis=1) >= lower, axis=1) & np.all(
        segments.min(axis=1) <= upper, axis=1
    )
    segments = segments[local]
    if not len(segments):
        return None
    loop_distances = distances_to_segments(loop, segments)
    segment_midpoints = segments.mean(axis=1)
    host_distances = distances_to_loop(segment_midpoints, loop)
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    positive = lengths > 1e-12
    lengths, host_distances = lengths[positive], host_distances[positive]
    if not len(lengths) or float(lengths.sum()) <= 1e-12:
        return None
    host_coverage = float(lengths[host_distances <= tolerance].sum() / lengths.sum())
    return {
        "distance_tolerance_mm": float(tolerance),
        "loop_to_host": _distance_summary(loop_distances, tolerance),
        "host_to_loop": {
            "coverage": host_coverage,
            "p50_distance_mm": float(np.quantile(host_distances, 0.50)),
            "p90_distance_mm": float(np.quantile(host_distances, 0.90)),
            "maximum_distance_mm": float(host_distances.max()),
            "sampled_local_segment_count": int(len(lengths)),
        },
        "matched": (
            float(np.mean(loop_distances <= tolerance)) >= MINIMUM_BOUNDARY_COVERAGE
            and host_coverage >= MINIMUM_BOUNDARY_COVERAGE
        ),
    }


def maximum_distance_to_loop(points: np.ndarray, loop: np.ndarray) -> float:
    """Measure a closed contour against the segments of another contour."""
    return float(distances_to_loop(points, loop).max())


def matching_seam(left: np.ndarray, right: np.ndarray) -> bool:
    """Return whether two closed boundaries meet the symmetric coverage rule."""
    comparison = compare_boundary_loops(left, right)
    return bool(comparison and comparison["matched"])
