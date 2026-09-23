"""Remove local projection folds for solving, retaining their exact 3-D ears."""
from dataclasses import dataclass

import numpy as np

from .print_tolerance import small_patch_report


@dataclass(frozen=True)
class ProjectedFoldRepair:
    retained_indices: tuple[int, ...]
    ear_indices: tuple[tuple[int, int, int], ...]
    evidence: dict


def _cross(left, right):
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def crossing_pairs(points):
    """Find strict crossings through a bounded uniform-grid broad phase."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 4:
        return []
    ends = np.roll(points, -1, axis=0)
    minimum = np.minimum(points, ends)
    maximum = np.maximum(points, ends)
    origin = points.min(axis=0)
    extent = np.maximum(np.ptp(points, axis=0), 1e-12)
    side = max(1, int(np.ceil(np.sqrt(len(points)))))
    scale = np.asarray([side, side], dtype=np.float64) / extent
    lower = np.clip(
        np.floor((minimum - origin) * scale).astype(np.int64), 0, side - 1
    )
    upper = np.clip(
        np.floor((maximum - origin) * scale).astype(np.int64), 0, side - 1
    )
    buckets = {}
    long_segments = []
    for index, (start, stop) in enumerate(zip(lower, upper)):
        if int(np.prod(stop - start + 1)) > 64:
            long_segments.append(index)
            continue
        for x in range(int(start[0]), int(stop[0]) + 1):
            for y in range(int(start[1]), int(stop[1]) + 1):
                buckets.setdefault((x, y), []).append(index)
    candidates = set()
    for members in buckets.values():
        for position, left in enumerate(members):
            candidates.update(
                (left, right) if left < right else (right, left)
                for right in members[position + 1 :]
            )
    for left in long_segments:
        candidates.update(
            (left, right) if left < right else (right, left)
            for right in range(len(points)) if right != left
        )
    pairs = []
    for index, other in sorted(candidates):
        cyclic_distance = other - index
        if cyclic_distance <= 1 or cyclic_distance >= len(points) - 1:
            continue
        if np.any(maximum[index] < minimum[other]) or np.any(
            maximum[other] < minimum[index]
        ):
            continue
        starts, stops = points[other], ends[other]
        hit = (
            _cross(ends[index] - points[index], starts - points[index])
            * _cross(ends[index] - points[index], stops - points[index]) < -1e-20
        ) & (
            _cross(stops - starts, points[index] - starts)
            * _cross(stops - starts, ends[index] - starts) < -1e-20
        )
        if bool(hit):
            pairs.append((index, other))
    return pairs


def isolate_micro_folds(points, projected, *, maximum_ears=32):
    """Peel short crossing arcs; never move/drop a boundary vertex or face.

    Returned ears must be appended to the solved reduced annulus. Their summed
    physical area and connected spans share the ordinary micro-patch budget.
    The caller must still audit the complete restored strip in three dimensions.
    """
    points = np.asarray(points, dtype=np.float64)
    projected = np.asarray(projected, dtype=np.float64)
    active = list(range(len(points)))
    ears = []
    for _ in range(maximum_ears + 1):
        pairs = crossing_pairs(projected[active])
        if not pairs:
            if not ears:
                return None
            evidence = small_patch_report(
                points, ears, maximum_span_mm=5.0, separate_components=True)
            return ProjectedFoldRepair(tuple(active), tuple(ears), evidence)
        if len(ears) == maximum_ears or len(active) <= 4:
            return None
        candidates = set()
        for left, right in pairs:
            if right - left <= 4:
                candidates.update(range(left + 1, right + 1))
            elif len(active) + left - right <= 4:
                candidates.update(index % len(active)
                                  for index in range(right + 1, len(active) + left + 1))
        ranked = []
        for index in candidates:
            ear = (active[(index - 1) % len(active)], active[index],
                   active[(index + 1) % len(active)])
            triangle = points[list(ear)]
            area = float(np.linalg.norm(np.cross(
                triangle[1] - triangle[0], triangle[2] - triangle[0])) * 0.5)
            if area <= 1e-12:
                continue
            evidence = small_patch_report(
                points, ears + [ear], maximum_span_mm=5.0, separate_components=True)
            if evidence['accepted']:
                ranked.append((area, index, ear))
        if not ranked:
            return None
        _, index, ear = min(ranked)
        ears.append(ear)
        active.pop(index)
    return None
