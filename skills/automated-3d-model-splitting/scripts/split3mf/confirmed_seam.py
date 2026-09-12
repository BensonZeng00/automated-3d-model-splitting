"""Immutable visible seams explicitly approved by the user, not a failed-fit fallback."""
from __future__ import annotations

import numpy as np
from .surface_quality import directed_edge_topology_issues


def preserve_confirmed_loop(points, source_vertex_ids):
    source = np.asarray(points, dtype=np.float64)
    ids = np.asarray(source_vertex_ids, dtype=np.int64)
    if source.shape != (len(ids), 3) or not np.isfinite(source).all():
        raise ValueError('Confirmed seam needs finite source coordinates matching vertex ids')
    if len(ids) < 3 or len(np.unique(ids)) != len(ids):
        raise ValueError('Confirmed seam must contain at least three unique source ids')
    return source.copy(), dict(
        mode='user-confirmed-immutable-seam', status='source-locked',
        source_vertices=len(ids), target_samples=len(ids), smooth_passes=0,
        source_order_preserved=True, source_arc_parameters_preserved=True,
        maximum_arc_parameter_change=0.0, maximum_target_offset_mm=0.0,
        rms_target_offset_mm=0.0, visible_top_source_preserved=True,
        visible_boundary_retopologized=False,
        surface_restore_policy='explicit-user-confirmation-no-surface-motion',
    )


def audit_unchanged_surface(points, faces, loops):
    """Audit indexed topology without repairing, moving or deleting source faces."""
    source = np.asarray(points, dtype=np.float64)
    if not np.isfinite(source).all():
        raise ValueError('Confirmed source contains nonfinite vertices')
    edge_sets = []
    for loop in loops:
        ids = np.asarray(loop, dtype=np.int64)
        edges = {tuple(sorted((int(a), int(b)))) for a, b in zip(ids, np.roll(ids, -1))}
        if any(edges & previous for previous in edge_sets):
            raise ValueError('Confirmed loops share a boundary edge')
        edge_sets.append(edges)
    over_shared = inconsistent = source_degenerate = 0
    if faces is not None:
        indexed = np.asarray(faces, dtype=np.int64)
        over_shared_edges, inconsistent_edges = directed_edge_topology_issues(indexed)
        over_shared, inconsistent = len(over_shared_edges), len(inconsistent_edges)
        triangles = source[indexed]
        twice_area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                            triangles[:, 2]-triangles[:, 0]), axis=1)
        source_degenerate = int(np.count_nonzero(twice_area <= 1e-12))
        if over_shared or inconsistent:
            raise ValueError(f'Confirmed source topology: over_shared={over_shared}, inconsistent={inconsistent}')
    return dict(valid=True, strategy='immutable-source-no-deformation',
                surface_band_moved_vertices=0, surface_band_affected_faces=0,
                boundary_match_error_mm=0.0, generated_degenerate_face_count=0,
                unchanged_source_degenerate_face_count=source_degenerate,
                over_shared_edges=over_shared, inconsistent_edges=inconsistent,
                source_quality_not_repaired=True)
