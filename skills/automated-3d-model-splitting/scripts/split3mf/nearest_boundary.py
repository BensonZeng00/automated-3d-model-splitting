"""Nearest reliable ownership for uncertain endpoint neighborhoods."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from .tolerance_policy import TolerancePolicy


def _endpoint_count(assessment):
    return sum(item['unexplained_endpoints'] for item in assessment.pairs)


def merge_nearest_ownership(graph, owners, max_rounds=3):
    """Move local face ownership toward nearest stable neighboring material.

No vertex is moved and no face is deleted. Accept only endpoint improvements
that preserve material populations and do not split an existing region. An
unresolved endpoint stays explicitly unresolved; it is never fake-welded.
"""
    result = np.asarray(owners).astype(str).copy()
    original = result.copy()
    before = graph.assess(result)
    centroids = graph.vertices[graph.faces].mean(axis=1)
    changes = []
    for _ in range(max_rounds):
        assessment = graph.assess(result)
        band = assessment.uncertain_faces
        if not len(band):
            break
        candidate = result.copy()
        stable = np.ones(len(result), dtype=bool)
        stable[band] = False
        for item in assessment.reasons:
            if item['branching_vertices']:
                continue
            pair = item['pair']
            target = band[np.isin(result[band], pair)]
            # Restrict anchors to the adjacent ring of this uncertain neighborhood.
            adjacent = graph.band(target, rings=1)
            anchors = adjacent[stable[adjacent] & np.isin(result[adjacent], pair)]
            if not len(anchors) or not len(target):
                continue
            # Query both materials separately. Stable label order resolves ties.
            distances, labels = [], []
            for owner in sorted(pair):
                ids = anchors[result[anchors] == owner]
                if len(ids):
                    distances.append(cKDTree(centroids[ids]).query(centroids[target])[0])
                    labels.append(owner)
            if len(labels) == 2:
                winners = np.argmin(np.asarray(distances), axis=0)
                candidate[target] = np.asarray(labels)[winners]
        changed = np.flatnonzero(candidate != result)
        after = graph.assess(candidate)
        if not len(changed) or _endpoint_count(after) >= _endpoint_count(assessment):
            break
        if any(r['branching_vertices'] for r in after.reasons):
            break
        accepted = True
        for owner in np.unique(np.r_[result[changed], candidate[changed]]):
            old_ids = np.flatnonzero(result == owner)
            new_ids = np.flatnonzero(candidate == owner)
            if not len(new_ids):
                accepted = False
                break
            old_count = connected_components(graph.graph[old_ids][:, old_ids], directed=False, return_labels=False)
            new_count = connected_components(graph.graph[new_ids][:, new_ids], directed=False, return_labels=False)
            if new_count > old_count:
                accepted = False
                break
        if not accepted:
            break
        changes.extend(changed.tolist())
        result = candidate
    after = graph.assess(result)
    policy = TolerancePolicy()
    pairs = []
    for item in after.pairs:
        total = item['boundary_vertices']
        count = item['unexplained_endpoints']
        pairs.append(dict(**item, endpoint_ratio=count / max(total, 1),
                          warning=policy.warning(count, total)))
    return result, dict(method='nearest-stable-adjacent-material',
                        vertices_moved=0, faces_deleted=0,
                        changed_face_count=int(np.count_nonzero(result != original)),
                        changed_face_ids=np.flatnonzero(result != original).tolist(),
                        endpoints_before=_endpoint_count(before),
                        endpoints_after=_endpoint_count(after),
                        warning_threshold=policy.warning_ratio,
                        ratios_evaluated_after_merge=True, pairs=pairs,
                        warning=any(p['warning'] for p in pairs))
