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
    """Find strict crossings through a bounded sweep-line broad phase."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 4:
        return []
    ends = np.roll(points, -1, axis=0)
    minimum = np.minimum(points, ends)
    maximum = np.maximum(points, ends)
    # Sweep by minimum X.  The previous uniform-grid implementation expanded
    # every dense bucket into a Python ``set`` of segment pairs; a folded
    # 28k-edge paint contour produced millions of boxed tuples before the
    # vectorized exact test even began.  The active sweep retains only segments
    # whose X intervals can still overlap and applies the Y interval test
    # immediately, so memory follows local geometric density rather than the
    # square of tessellation density.
    order = np.lexsort((np.arange(len(points)), minimum[:, 0]))
    active: list[int] = []
    pairs: list[tuple[int, int]] = []
    for current_value in order:
        current = int(current_value)
        current_minimum_x = float(minimum[current, 0])
        active = [
            index
            for index in active
            if float(maximum[index, 0]) >= current_minimum_x
        ]
        if not active:
            active.append(current)
            continue
        active_ids = np.asarray(active, dtype=np.int64)
        eligible = (
            (maximum[active_ids, 1] >= minimum[current, 1])
            & (maximum[current, 1] >= minimum[active_ids, 1])
        )
        candidate_ids = active_ids[eligible]
        if not len(candidate_ids):
            active.append(current)
            continue
        left_ids = np.minimum(candidate_ids, current)
        right_ids = np.maximum(candidate_ids, current)
        cyclic_distance = right_ids - left_ids
        nonadjacent = (cyclic_distance > 1) & (
            cyclic_distance < len(points) - 1
        )
        left_ids = left_ids[nonadjacent]
        right_ids = right_ids[nonadjacent]
        if not len(left_ids):
            active.append(current)
            continue
        left_starts = points[left_ids]
        left_stops = ends[left_ids]
        right_starts = points[right_ids]
        right_stops = ends[right_ids]
        left_vector = left_stops - left_starts
        right_vector = right_stops - right_starts
        hit = (
            _cross(left_vector, right_starts - left_starts)
            * _cross(left_vector, right_stops - left_starts) < -1e-20
        ) & (
            _cross(right_vector, left_starts - right_starts)
            * _cross(right_vector, left_stops - right_starts) < -1e-20
        )
        pairs.extend(
            (int(left), int(right))
            for left, right in zip(left_ids[hit], right_ids[hit])
        )
        active.append(current)
    return sorted(pairs)


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
