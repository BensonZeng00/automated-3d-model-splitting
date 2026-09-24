"""Cheap shared-edge boundary routing; never equate triangulation with ambiguity."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class BoundaryAssessment:
    reasons: list[dict] = field(default_factory=list)
    pairs: list[dict] = field(default_factory=list)
    uncertain_faces: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def clear(self) -> bool:
        return not self.reasons

    def as_record(self) -> dict:
        return dict(status='clear' if self.clear else 'needs_review',
                    reasons=self.reasons, pairs=self.pairs,
                    uncertain_face_count=len(self.uncertain_faces))


class BoundaryGraph:
    """One adjacency build shared by assessment and bounded local proposals."""
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.adjacency, self.edges = trimesh.graph.face_adjacency(
            faces=self.faces, return_edges=True)
        a, b = self.adjacency.T if len(self.adjacency) else ([], [])
        self.graph = csr_matrix((np.ones(2*len(a)),
            (np.r_[a, b].astype(int), np.r_[b, a].astype(int))),
            shape=(len(self.faces), len(self.faces)))

    def band(self, seeds, rings=2):
        mask = np.zeros(len(self.faces), dtype=bool)
        mask[np.asarray(seeds, dtype=int)] = True
        for _ in range(rings):
            mask |= np.asarray(self.graph @ mask.astype(float)).ravel() > 0
        return np.flatnonzero(mask)

    def assess(self, owners) -> BoundaryAssessment:
        owners = np.asarray(owners)
        if owners.shape != (len(self.faces),):
            raise ValueError('Boundary labels must match face count')
        result = BoundaryAssessment()
        if not len(self.adjacency):
            return result
        changed = owners[self.adjacency[:, 0]] != owners[self.adjacency[:, 1]]
        seam_edges = self.edges[changed]
        seam_faces = self.adjacency[changed]
        if not len(seam_edges):
            return result
        pair_values = np.sort(owners[seam_faces], axis=1)
        pairs, inverse = np.unique(pair_values, axis=0, return_inverse=True)
        # A valid three-material junction has pair endpoints; it is not a broken loop.
        total_degree = np.bincount(seam_edges.ravel(), minlength=len(self.vertices))
        seeds = []
        for i, pair in enumerate(pairs):
            select = inverse == i
            edges = seam_edges[select]
            vertices, degree = np.unique(edges, return_counts=True)
            bad = vertices[(degree > 2) | ((degree == 1) & (total_degree[vertices] == 1))]
            record = dict(pair=[str(x) for x in pair], edges=len(edges), boundary_vertices=len(vertices),
                          branching_vertices=int(np.sum(degree > 2)),
                          unexplained_endpoints=int(np.sum((degree == 1) & (total_degree[vertices] == 1))))
            result.pairs.append(record)
            if len(bad):
                result.reasons.append(dict(kind='ambiguous_boundary_topology', **record))
                affected = np.any(np.isin(edges, bad), axis=1)
                seeds.extend(seam_faces[select][affected].ravel().tolist())
        result.uncertain_faces = self.band(seeds) if seeds else np.empty(0, dtype=int)
        return result

    def fingerprint(self, owners, context=''):
        digest = hashlib.sha256()
        digest.update(b'boundary-clarity-v1:two-rings:eight-rounds:weights-0-1-3')
        for array in (self.vertices.astype('<f8'), self.faces.astype('<i8')):
            digest.update(array.tobytes())
        digest.update('\0'.join(map(str, owners)).encode())
        digest.update(context.encode())
        return digest.hexdigest()

    def propose(self, owners, assessment, max_candidates=3):
        """Local ownership proposals, NOT paint changes or automatically approved cuts.

        Only uncertain faces can change. Confidence anchors outside the band
        never move. Different edge weights favor length or geometric creases.
        """
        base = np.asarray(owners).astype(str)
        a, b = self.adjacency.T
        lengths = np.linalg.norm(self.vertices[self.edges[:, 1]]-self.vertices[self.edges[:, 0]], axis=1)
        tri = self.vertices[self.faces]
        normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
        bend = np.clip(np.einsum('ij,ij->i', normals[a], normals[b]), -1, 1)
        proposals, seen = [], set()
        for crease_weight in (0.0, 1.0, 3.0)[:max_candidates]:
            result = base.copy()
            weights = np.maximum(lengths, 1e-12) * np.exp(crease_weight*(bend-1))
            graph = csr_matrix((np.r_[weights, weights], (np.r_[a, b], np.r_[b, a])), shape=self.graph.shape)
            for _ in range(8):
                updated = result.copy()
                for face in assessment.uncertain_faces:
                    start, end = graph.indptr[face:face+2]
                    neighbors = graph.indices[start:end]
                    if not len(neighbors):
                        continue
                    labels, ids = np.unique(result[neighbors], return_inverse=True)
                    scores = np.bincount(ids, weights=graph.data[start:end])
                    current = np.flatnonzero(labels == result[face])
                    if len(current):
                        scores[current[0]] += 0.25 * graph.data[start:end].sum()
                    winner = int(np.argmax(scores))
                    if np.sum(np.isclose(scores, scores[winner], rtol=1e-10, atol=1e-15)) == 1:
                        updated[face] = labels[winner]
                if np.array_equal(updated, result):
                    break
                result = updated
            changed = np.flatnonzero(result != base)
            if not len(changed) or not set(np.unique(base)).issubset(set(np.unique(result))):
                continue
            # Never propose a new disconnected region or remove a whole part.
            disconnected = False
            for owner in np.unique(base):
                before_ids, after_ids = np.flatnonzero(base == owner), np.flatnonzero(result == owner)
                before = connected_components(self.graph[before_ids][:, before_ids], directed=False, return_labels=False)
                after = connected_components(self.graph[after_ids][:, after_ids], directed=False, return_labels=False)
                disconnected |= after > before
            if disconnected:
                continue
            key = self.fingerprint(result)
            if key in seen:
                continue
            seen.add(key)
            audit = self.assess(result)
            proposals.append(dict(id=f'C{len(proposals)+1:02d}', owners=result,
                changed_faces=changed, crease_weight=crease_weight, assessment=audit,
                admissible=audit.clear))
        return proposals
