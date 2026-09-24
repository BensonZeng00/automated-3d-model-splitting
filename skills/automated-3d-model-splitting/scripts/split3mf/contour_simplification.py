"""Topology-safe, error-bounded simplification of closed planar contours."""

from __future__ import annotations

import numpy as np


def _signed_area(points: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(points[:, 0] * np.roll(points[:, 1], -1)
               - points[:, 1] * np.roll(points[:, 0], -1))
    )


def _rdp_indices(points: np.ndarray, tolerance: float) -> list[int]:
    """Return RDP indices for an open polyline, including both endpoints."""
    keep = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        first, last = pending.pop()
        if last <= first + 1:
            continue
        chord = points[last] - points[first]
        chord_squared = float(np.dot(chord, chord))
        candidates = points[first + 1:last]
        if chord_squared <= 1e-24:
            distances = np.linalg.norm(candidates - points[first], axis=1)
        else:
            fractions = np.clip(
                ((candidates - points[first]) @ chord) / chord_squared, 0.0, 1.0
            )
            projections = points[first] + fractions[:, None] * chord
            distances = np.linalg.norm(candidates - projections, axis=1)
        relative = int(np.argmax(distances))
        if float(distances[relative]) > tolerance:
            split = first + 1 + relative
            keep.add(split)
            pending.extend(((first, split), (split, last)))
    return sorted(keep)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = b - a
    second = c - a
    return float(first[0] * second[1] - first[1] * second[0])


def _has_self_intersection(points: np.ndarray) -> bool:
    count = len(points)
    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 2, count):
            if first == 0 and second == count - 1:
                continue
            c, d = points[second], points[(second + 1) % count]
            scale = max(
                np.linalg.norm(b - a), np.linalg.norm(d - c), 1.0
            )
            epsilon = 1e-12 * scale
            o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
            o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
            if o1 * o2 < -epsilon * epsilon and o3 * o4 < -epsilon * epsilon:
                return True
            # A non-adjacent touch or overlap is also invalid for a contour.
            if (
                abs(o1) <= epsilon and _on_segment(a, b, c, epsilon)
                or abs(o2) <= epsilon and _on_segment(a, b, d, epsilon)
                or abs(o3) <= epsilon and _on_segment(c, d, a, epsilon)
                or abs(o4) <= epsilon and _on_segment(c, d, b, epsilon)
            ):
                return True
    return False


def _on_segment(a: np.ndarray, b: np.ndarray, point: np.ndarray, epsilon: float) -> bool:
    return bool(
        np.all(point >= np.minimum(a, b) - epsilon)
        and np.all(point <= np.maximum(a, b) + epsilon)
    )


def simplify_closed_contour(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Return source indices for a valid closed RDP contour.

    The result uses a subset of the input vertices, has the same winding, and
    is accepted only when it remains a simple polygon.  The tolerance is a
    geometric error bound, not a requested vertex count.
    """
    contour = np.asarray(points, dtype=np.float64)
    if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3:
        raise ValueError("closed contour must contain at least three 2-D points")
    if not np.all(np.isfinite(contour)):
        raise ValueError("closed contour contains non-finite points")
    if tolerance <= 0.0 or len(contour) <= 3:
        return np.arange(len(contour), dtype=np.int64)
    original_area = _signed_area(contour)
    if abs(original_area) <= 1e-15:
        raise ValueError("input contour is not an oriented polygon")

    # Split the cycle at two well-separated existing vertices so neither the
    # closing edge nor an arbitrary array seam receives special treatment.
    anchor = int(np.argmin(contour[:, 0]))
    opposite = int(np.argmax(np.linalg.norm(contour - contour[anchor], axis=1)))
    if anchor > opposite:
        anchor, opposite = opposite, anchor
    first_arc = np.arange(anchor, opposite + 1, dtype=np.int64)
    second_arc = np.concatenate((
        np.arange(opposite, len(contour), dtype=np.int64),
        np.arange(0, anchor + 1, dtype=np.int64),
    ))
    first_keep = first_arc[_rdp_indices(contour[first_arc], float(tolerance))]
    second_keep = second_arc[_rdp_indices(contour[second_arc], float(tolerance))]
    indices = np.concatenate((first_keep[:-1], second_keep[:-1]))
    simplified = contour[indices]
    if (
        len(indices) < 3
        or _signed_area(simplified) * original_area <= 0.0
        or _has_self_intersection(simplified)
    ):
        return np.arange(len(contour), dtype=np.int64)
    return indices
