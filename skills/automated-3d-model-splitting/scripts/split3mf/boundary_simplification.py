"""Transfer small crossing lobes to adjacent regions before shared smoothing."""
from __future__ import annotations
import numpy as np
from scipy.sparse.csgraph import connected_components
from .mesh import boundary_cycles_from_edges
from .curve_clarity import crossings

MAX_EXACT_CROSSING_LOOP_VERTICES = 4096
MAX_EXACT_CROSSING_COMPARISONS = 1_000_000
MAX_EXACT_CROSSING_SEAM_EDGES = 20_000


def _crossing_vertices(graph, owners, owner):
    """Inspect bounded inter-owner seam loops, not every triangle of a region.

    Exact projected segment intersection is quadratic in loop length.  Large
    loops are left untouched for the normal topology assessment and explicit
    review instead of risking an out-of-memory failure in this optional cleanup.
    """
    owners = np.asarray(owners)
    adjacent_owners = owners[graph.adjacency]
    selected = (
        (adjacent_owners[:, 0] == owner)
        ^ (adjacent_owners[:, 1] == owner)
    )
    seam_edges = graph.edges[selected]
    if len(seam_edges) > MAX_EXACT_CROSSING_SEAM_EDGES:
        return set(), 0
    bad, count = set(), 0
    remaining_comparisons = MAX_EXACT_CROSSING_COMPARISONS
    for loop in boundary_cycles_from_edges(seam_edges):
        comparisons = max(0, len(loop) * (len(loop) - 3) // 2)
        if (
            len(loop) > MAX_EXACT_CROSSING_LOOP_VERTICES
            or comparisons > remaining_comparisons
        ):
            continue
        remaining_comparisons -= comparisons
        ids = np.asarray(loop, dtype=int)
        if len(ids) < 4:
            continue
        points = graph.vertices[ids]
        _, _, basis = np.linalg.svd(points-points.mean(axis=0), full_matrices=False)
        pairs = crossings((points-points.mean(axis=0)) @ basis[:2].T)
        count += len(pairs)
        for left, right, _, _ in pairs:
            arcs = (list(range(left+1, right+1)),
                    [i % len(ids) for i in range(right+1, len(ids)+left+1)])
            bad.update(ids[min(arcs, key=len)].tolist())
    return bad, count


def _same_connectivity(graph, before, after):
    changed = np.flatnonzero(before != after)
    if not len(changed):
        return True
    affected_owners = np.unique(np.r_[before[changed], after[changed]])
    for owner in affected_owners:
        a, b = np.flatnonzero(before == owner), np.flatnonzero(after == owner)
        if not len(b):
            return False
        if connected_components(graph.graph[b][:, b], directed=False, return_labels=False) > \
                connected_components(graph.graph[a][:, a], directed=False, return_labels=False):
            return False
    return True


def simplify_crossing_ownership(graph, owners, budget, *, max_passes=8):
    """Bounded, transactional cleanup; no source vertex, triangle or paint edit."""
    result = np.asarray(owners).astype(str).copy()
    records = []
    a, b = graph.adjacency.T if len(graph.adjacency) else ([], [])
    # This optional auto-cleanup is intended for small local crossing lobes.
    # A model-wide seam above the exact-work budget proceeds unchanged to the
    # topology assessment and explicit review; correctness gates are not skipped.
    if len(a) and np.count_nonzero(result[a] != result[b]) > MAX_EXACT_CROSSING_SEAM_EDGES:
        return result, records
    for owner in np.unique(result):
        if owner == '':
            continue
        candidate = result.copy()
        initial_count = None
        for _ in range(max_passes):
            ids = np.flatnonzero(candidate == owner)
            bad, count = _crossing_vertices(graph, candidate, owner)
            if initial_count is None:
                initial_count = count
            if not count:
                break
            removed = ids[np.isin(graph.faces[ids], list(bad)).any(axis=1)]
            # Use only neighboring regions across the actual shared edges.
            incident = np.isin(a, removed) ^ np.isin(b, removed)
            contacts = graph.adjacency[incident].ravel()
            targets = candidate[contacts]
            targets = targets[(targets != owner) & (targets != '')]
            if not len(targets):
                break
            labels, frequencies = np.unique(targets, return_counts=True)
            candidate[removed] = labels[np.argmax(frequencies)]
            policy = budget.evaluate(result, candidate)
            if not policy['automatic'] or not _same_connectivity(graph, result, candidate):
                break
        if not initial_count:
            continue
        _, remaining = _crossing_vertices(graph, candidate, owner)
        policy = budget.evaluate(result, candidate)
        accepted = bool(remaining == 0 and policy['automatic']
                        and graph.assess(candidate).clear
                        and _same_connectivity(graph, result, candidate))
        records.append(dict(owner=str(owner), projected_crossings_before=initial_count,
                            projected_crossings_after=remaining, applied=accepted,
                            area_policy=policy, next_action='continue_smoothing' if accepted
                            else 'completion_principle_assessment',
                            proposed_changed_faces=np.flatnonzero(candidate != result).tolist()))
        if accepted:
            budget.evaluate(result, candidate, commit=True)
            result = candidate
    return result, records
