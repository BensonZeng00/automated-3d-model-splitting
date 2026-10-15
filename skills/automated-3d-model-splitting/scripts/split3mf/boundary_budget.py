"""Area-based cumulative budget for automatic local ownership cleanup."""
from __future__ import annotations

import numpy as np

DEFAULT_BOUNDARY_ANOMALY_RATIO = 0.01


class BoundaryAreaBudget:
    """Freeze each source region's area; never dilute a defect with body area."""

    def __init__(self, vertices, faces, owners):
        triangles = np.asarray(vertices)[np.asarray(faces)]
        self.areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                            triangles[:, 2]-triangles[:, 0]), axis=1)/2
        self.original = np.asarray(owners).astype(str).copy()
        self.affected = {owner: set() for owner in np.unique(self.original)}
        self.region_areas = {owner: float(self.areas[self.original == owner].sum())
                             for owner in self.affected}

    def evaluate(self, before, after, *, commit=False):
        before, after = np.asarray(before).astype(str), np.asarray(after).astype(str)
        if before.shape != self.original.shape or after.shape != self.original.shape:
            raise ValueError('Boundary ownership shape changed')
        changed = np.flatnonzero(before != after)
        affected = {key: set(ids) for key, ids in self.affected.items()}
        identities_preserved = set(np.unique(after)) == set(self.region_areas)
        if not set(np.unique(after)).issubset(affected):
            raise ValueError('Boundary cleanup introduced unknown part identities')
        for face in changed:
            affected[before[face]].add(int(face))
            affected[after[face]].add(int(face))
        regions = []
        for owner, ids in sorted(affected.items()):
            if not ids:
                continue
            area = float(self.areas[sorted(ids)].sum())
            total = self.region_areas[owner]
            ratio = area / total if total > 0 else float('inf')
            regions.append(dict(owner=owner, affected_area_mm2=area,
                                source_region_area_mm2=total, affected_area_ratio=ratio))
        automatic = identities_preserved and all(
            item['affected_area_ratio'] < DEFAULT_BOUNDARY_ANOMALY_RATIO for item in regions)
        result = dict(method='cumulative-source-region-surface-area',
                      threshold=DEFAULT_BOUNDARY_ANOMALY_RATIO, comparison='strictly_less_than',
                      automatic=automatic, identities_preserved=identities_preserved,
                      changed_face_count=len(changed), regions=regions,
                      above_threshold_action='completion_principle_assessment_not_automatic_rejection')
        if commit:
            self.affected = affected
        return result
