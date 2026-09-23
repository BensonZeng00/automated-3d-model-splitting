"""Bounded projected-loop surgery; proposals are NOT source-vertex assignments.

Small crossing lobes are removed from the ordered curve, not hidden in a plot.
Each new vertex carries sparse source weights. Depth disagreement is retained
as evidence requiring review, never interpreted as a physical intersection.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def signed_area(points):
    return float(np.sum(cross2(points, np.roll(points, -1, axis=0))) / 2)


def crossings(points):
    """Proper nonadjacent intersections with a sweep broad phase for dense rings."""
    p = np.asarray(points, dtype=float)
    edges = np.roll(p, -1, axis=0) - p
    if len(p) > 2048:
        return _sweep_crossings(p, edges)
    found = []
    for i in range(len(p) - 2):
        js = np.arange(i + 2, len(p) - (i == 0))
        delta = p[js] - p[i]
        denominator = cross2(edges[i], edges[js])
        valid = np.abs(denominator) > 1e-12
        t = np.divide(cross2(delta, edges[js]), denominator,
                      out=np.zeros(len(js)), where=valid)
        s = np.divide(cross2(delta, edges[i]), denominator,
                      out=np.zeros(len(js)), where=valid)
        hits = valid & (t > 1e-9) & (t < 1 - 1e-9) & (s > 1e-9) & (s < 1 - 1e-9)
        found.extend((i, int(js[k]), float(t[k]), float(s[k])) for k in np.flatnonzero(hits))
    return found


def _sweep_crossings(points, edges):
    """Conservative segment-AABB sweep followed by the same exact predicate."""
    low = np.minimum(points, np.roll(points, -1, axis=0))
    high = np.maximum(points, np.roll(points, -1, axis=0))
    order = np.lexsort((np.arange(len(points)), low[:, 0]))
    active, found = [], []
    count = len(points)
    for raw_j in order:
        j = int(raw_j)
        active = [i for i in active if high[i, 0] >= low[j, 0]]
        for i in active:
            if abs(i - j) <= 1 or {i, j} == {0, count - 1}:
                continue
            if high[i, 1] < low[j, 1] or high[j, 1] < low[i, 1]:
                continue
            a, b = sorted((i, j))
            delta = points[b] - points[a]
            denominator = float(cross2(edges[a], edges[b]))
            if abs(denominator) <= 1e-12:
                continue
            t = float(cross2(delta, edges[b]) / denominator)
            s = float(cross2(delta, edges[a]) / denominator)
            if 1e-9 < t < 1 - 1e-9 and 1e-9 < s < 1 - 1e-9:
                found.append((a, b, t, s))
        active.append(j)
    return sorted(found, key=lambda item: item[:2])


@dataclass(frozen=True)
class ClearCurveProposal:
    points: np.ndarray
    source_weights: object | None
    record: dict


def resample_approved_curve(proposal: ClearCurveProposal, reference: np.ndarray) -> np.ndarray:
    """Resample an approved contour onto an existing boundary ring.

    Approval permits the contour change, not arbitrary vertex insertion into a
    source mesh.  Keeping the existing ring cardinality lets the established
    surface-band remesher move the boundary and audit every affected face.
    The cyclic phase and direction are selected geometrically rather than by
    treating proposal rows as source vertex identities.
    """
    candidate = np.asarray(proposal.points, dtype=np.float64)
    old = np.asarray(reference, dtype=np.float64)
    if proposal.record.get("status") != "proposed":
        raise ValueError("Only a complete clear-curve proposal can be applied")
    if len(candidate) < 3 or len(old) < 3:
        raise ValueError("Approved curve and reference must be closed rings")
    edges = np.roll(candidate, -1, axis=0) - candidate
    lengths = np.linalg.norm(edges, axis=1)
    total = float(lengths.sum())
    if total <= 1e-12:
        raise ValueError("Approved curve has zero perimeter")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    distances = np.arange(len(old), dtype=np.float64) * total / len(old)
    indices = np.searchsorted(cumulative, distances, side="right") - 1
    indices = np.clip(indices, 0, len(candidate) - 1)
    blend = np.divide(distances - cumulative[indices], lengths[indices],
                      out=np.zeros_like(distances), where=lengths[indices] > 1e-12)
    sampled = candidate[indices] + edges[indices] * blend[:, None]
    # Only two orientations and the nearest phase can represent the same
    # closed contour.  This avoids an O(n^2) cyclic correspondence search.
    choices = []
    for values in (sampled, sampled[::-1]):
        start = int(np.argmin(np.linalg.norm(values - old[0], axis=1)))
        aligned = np.roll(values, -start, axis=0)
        score = float(np.sum((aligned[:min(32, len(old))] - old[:min(32, len(old))]) ** 2))
        choices.append((score, aligned))
    return min(choices, key=lambda item: item[0])[1]


def _thin_width(points):
    """Twice area/perimeter estimates mean width without a convex hull."""
    length = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum()
    return 2 * abs(signed_area(points)) / max(float(length), 1e-12)


def _canonical_weights(points):
    # Coordinate ordering, not input starting index, makes surgery cyclic and
    # reversal invariant. Sparse identities are restored by the returned map.
    start = int(np.lexsort(np.asarray(points).T[::-1])[0])
    order = np.roll(np.arange(len(points)), -start)
    if tuple(points[order[1]]) > tuple(points[order[-1]]):
        order = np.r_[order[:1], order[:0:-1]]
    from scipy.sparse import eye
    return eye(len(points), format='csr')[order]


def propose_clear_curve(points, origin, u, v, normal, *, max_attempts=3,
                        max_removed_area_fraction=0.12, depth_tolerance_mm=0.20):
    """Return a main-contour proposal only for a crossing projected loop.

    No convex hull, global smoothing, closest-point ID pairing or mesh edits.
    Each of at most three width profiles has a bounded number of loop cuts.
    Any topology change requires remeshing and explicit review downstream.
    """
    original = np.asarray(points, dtype=float)
    if original.ndim != 2 or original.shape[1] != 3 or len(original) < 3:
        raise ValueError('Expected at least three 3-D curve points')
    if not np.isfinite(original).all():
        raise ValueError('Curve points must be finite')
    basis = np.column_stack((u, v))
    projected = (original - origin) @ basis
    initial = crossings(projected)
    base = dict(method='bounded-small-lobe-main-contour-v1',
                projected_crossings_before=len(initial), search_attempt_limit=3,
                source_vertices=len(original), topology_change=False,
                status='clear_direct', attempts=[], requires_user_confirmation=False)
    if not initial:
        return ClearCurveProposal(original.copy(), None, base)
    from scipy.sparse import vstack
    weights0 = _canonical_weights(original)
    area_budget = max(abs(signed_area(projected)), 1e-12) * max_removed_area_fraction
    best = None
    for width in (0.10, 0.25, 0.50)[:max(0, min(int(max_attempts), 3))]:
        weights = weights0.copy()
        events, removed_area = [], 0.0
        for _ in range(min(len(original), 64)):
            current = weights @ original
            xy = (current - origin) @ basis
            candidates = []
            for i, j, t, s in crossings(xy):
                at_i = (1-t)*weights[i] + t*weights[(i+1) % weights.shape[0]]
                at_j = (1-s)*weights[j] + s*weights[(j+1) % weights.shape[0]]
                junction = (at_i + at_j) / 2
                first = vstack((junction, weights[i+1:j+1]), format='csr')
                second = vstack((junction, weights[j+1:], weights[:i+1]), format='csr')
                pieces = [first, second]
                areas = [abs(signed_area((part @ original-origin) @ basis)) for part in pieces]
                small = int(np.argmin(areas))
                keep, discard = pieces[1-small], pieces[small]
                narrow = _thin_width((discard @ original-origin) @ basis)
                if keep.shape[0] < 3 or narrow > width or removed_area + areas[small] > area_budget:
                    continue
                depth = abs(float((((at_i-at_j) @ original) @ normal).item()))
                event = dict(removed_area_mm2=areas[small], mean_lobe_width_mm=narrow,
                             junction_depth_disagreement_mm=depth,
                             depth_within_tolerance=bool(depth <= depth_tolerance_mm))
                # Minimum local area loss, then deterministic geometric tie-break.
                candidates.append((areas[small], tuple((junction @ original).ravel()), keep, event))
            if not candidates:
                break
            _, _, weights, event = min(candidates, key=lambda item: item[:2])
            events.append(event)
            removed_area += event['removed_area_mm2']
        remaining = len(crossings((weights @ original-origin) @ basis))
        attempt = dict(width_limit_mm=width, projected_crossings_after=remaining,
                       removed_area_mm2=removed_area, edits=events)
        base['attempts'].append(attempt)
        if events and (best is None or (remaining, removed_area) < best[0]):
            best = ((remaining, removed_area), weights, attempt)
        if events and remaining == 0:
            break
    if best is None:
        base.update(status='unresolved_review', requires_user_confirmation=True)
        return ClearCurveProposal(original.copy(), None, base)
    (_, _), weights, chosen = best
    result = weights @ original
    base.update(status='proposed' if chosen['projected_crossings_after'] == 0 else 'unresolved_review',
                topology_change=True, requires_user_confirmation=True,
                requires_surface_remesh=True, source_id_assignment_valid=False,
                projected_crossings_after=chosen['projected_crossings_after'],
                target_vertices=len(result), removed_area_mm2=chosen['removed_area_mm2'],
                depth_ambiguous_junctions=sum(not e['depth_within_tolerance'] for e in chosen['edits']),
                removed_lobes=len(chosen['edits']),
                candidate_sha256=hashlib.sha256(result.astype('<f8').tobytes()).hexdigest())
    return ClearCurveProposal(result, weights, base)
