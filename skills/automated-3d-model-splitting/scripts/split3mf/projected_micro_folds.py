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
    """Find strict segment crossings, excluding adjacent cyclic edges."""
    points = np.asarray(points, dtype=np.float64)
    ends = np.roll(points, -1, axis=0)
    pairs = []
    for index in range(len(points) - 2):
        others = np.arange(index + 2, len(points) - (index == 0))
        starts, stops = points[others], ends[others]
        hit = (
            _cross(ends[index] - points[index], starts - points[index])
            * _cross(ends[index] - points[index], stops - points[index]) < -1e-20
        ) & (
            _cross(stops - starts, points[index] - starts)
            * _cross(stops - starts, ends[index] - starts) < -1e-20
        )
        pairs.extend((index, int(other)) for other in others[hit])
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
