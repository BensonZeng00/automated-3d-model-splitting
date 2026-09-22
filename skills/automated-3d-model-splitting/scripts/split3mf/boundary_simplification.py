"""Transfer small crossing lobes to adjacent regions before shared smoothing."""
from __future__ import annotations
import numpy as np
from scipy.sparse.csgraph import connected_components
from .mesh import boundary_loops
from .curve_clarity import crossings


def _crossing_vertices(graph, face_ids):
    bad, count = set(), 0
    for loop in boundary_loops(graph.faces[face_ids]):
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
    for owner in np.unique(before):
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
    for owner in np.unique(result):
        if owner == '':
            continue
        candidate = result.copy()
        initial_count = None
        for _ in range(max_passes):
            ids = np.flatnonzero(candidate == owner)
            bad, count = _crossing_vertices(graph, ids)
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
        _, remaining = _crossing_vertices(graph, np.flatnonzero(candidate == owner))
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
