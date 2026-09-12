"""Prove an edge-subdivision fan covers a source triangle in linear time."""
import numpy as np


def prove_edge_subdivision(source, replacements, tolerance_mm):
    old = np.asarray(source, dtype=float)
    new = np.asarray(replacements, dtype=float).reshape(-1, 3, 3)
    if old.shape != (3, 3) or not len(new) or not (
            np.isfinite(old).all() and np.isfinite(new).all()):
        return None
    tolerance = max(float(tolerance_mm), 1e-8)
    for apex_index in range(3):
        apex = old[apex_index]
        start, end = old[(apex_index+1) % 3], old[(apex_index+2) % 3]
        edge = end-start
        length = float(np.linalg.norm(edge))
        if length <= 1e-12:
            continue
        apex_distances = np.linalg.norm(new-apex, axis=2)
        apex_slots = np.argmin(apex_distances, axis=1)
        apex_error = float(apex_distances[np.arange(len(new)), apex_slots].max())
        if apex_error > tolerance:
            continue
        apex_points = new[np.arange(len(new)), apex_slots]
        if np.linalg.norm(apex_points-apex_points[0], axis=1).max() > 1e-8:
            continue
        mask = np.arange(3)[None, :] != apex_slots[:, None]
        ends = new[mask].reshape(-1, 2, 3)
        phases = ((ends-start) @ edge)/(length*length)
        projections = start + phases[:, :, None]*edge
        line_error = float(np.linalg.norm(ends-projections, axis=2).max())
        if line_error > tolerance:
            continue
        slots = np.argsort(phases, axis=1)
        ordered_ends = np.take_along_axis(ends, slots[:, :, None], axis=1)
        intervals = np.sort(phases, axis=1)
        order = np.argsort(intervals[:, 0])
        intervals = intervals[order]
        ordered_ends = ordered_ends[order]
        # Shared subdivision coordinates must still meet numerically: the
        # physical tolerance is for source rounding, not actual gaps/overlaps.
        endpoint_error = max(abs(float(intervals[0, 0])),
                             abs(float(intervals[-1, 1])-1))*length
        joins = np.abs(intervals[1:, 0]-intervals[:-1, 1])*length
        actual_joins = np.linalg.norm(ordered_ends[1:, 0]-ordered_ends[:-1, 1], axis=1)
        source_error = max(apex_error, float(np.hypot(line_error, endpoint_error)))
        if (endpoint_error > tolerance or np.any(joins > 1e-8) or np.any(actual_joins > 1e-8)
                or source_error > tolerance
                or np.any(intervals[:, 1]-intervals[:, 0] <= 1e-12)):
            continue
        normals = np.cross(new[:, 1]-new[:, 0], new[:, 2]-new[:, 0])
        if np.any(np.linalg.norm(normals, axis=1) <= 1e-12):
            continue
        return dict(method="traced_complete_edge_fan", tolerance_mm=tolerance,
                    maximum_source_error_mm=source_error,
                    replacement_faces=len(new))
    return None
