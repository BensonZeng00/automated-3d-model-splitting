"""Check actual insert/parent front ownership, including near-coplanar covers."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .surface_rays import first_surface_depth


def assess_insert_visibility(source_patch, insert, parent, *, reference=None,
                             sample_count=384, boundary_band_mm=0.15):
    areas = np.asarray(source_patch.area_faces)
    total_area = float(areas.sum())
    weighted_normal = (source_patch.face_normals * areas[:, None]).sum(axis=0)
    coherence = float(np.linalg.norm(weighted_normal) / max(total_area, 1e-12))
    record = {'method': 'exact_parallel_ray_parent_insert_ownership',
              'normal_coherence': coherence, 'boundary_band_mm': boundary_band_mm}
    if coherence < 0.5 or total_area <= 1e-12:
        return {**record, 'status': 'not_evaluated', 'reason': 'no_coherent_front'}
    axis = -weighted_normal / np.linalg.norm(weighted_normal)
    # Equal-area stratification avoids letting thousands of microscopic rim
    # triangles dominate an otherwise smooth, large visible patch.
    count = min(int(sample_count), max(len(areas), 1))
    ids = np.searchsorted(np.cumsum(areas),
                          (np.arange(count) + .5) * total_area / count)
    points = source_patch.triangles_center[ids]
    eligible = (source_patch.face_normals[ids] @ -axis) > 0.5
    edge_counts = np.bincount(source_patch.edges_unique_inverse)
    boundary = np.unique(source_patch.edges_unique[edge_counts == 1])
    if len(boundary):
        distances = cKDTree(source_patch.vertices[boundary]).query(points)[0]
        eligible &= distances >= float(boundary_band_mm)
    if reference is not None:
        eligible &= first_surface_depth(reference, points, axis) >= -1e-5
    owned_depth = first_surface_depth(insert, points, axis)
    parent_depth = first_surface_depth(parent, points, axis)
    eligible &= np.isfinite(owned_depth)
    covered = eligible & (parent_depth < owned_depth - 1e-6)
    checked = int(eligible.sum())
    return {**record, 'status': 'measured' if checked else 'not_evaluated',
            'inward_axis': axis.tolist(), 'sample_count': count,
            'checked_samples': checked, 'covered_samples': int(covered.sum()),
            'covered_fraction': float(covered.sum() / max(checked, 1)),
            'covered_area_estimate_mm2': float(covered.sum() * total_area / count)}
