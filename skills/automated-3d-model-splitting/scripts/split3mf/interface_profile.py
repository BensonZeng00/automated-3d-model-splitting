"""Elliptical end profile restored from the original interface implementation."""
import numpy as np
from .mesh import signed_area

def _simple_planar_inner_contour(
    desired_xy: np.ndarray, outer_xyz: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Build an ellipse in the plane normal to the tenon insertion axis."""
    desired = np.asarray(desired_xy, dtype=np.float64)
    center = desired.mean(axis=0)
    covariance = np.cov((desired - center).T, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues[::-1], 1e-8)
    axes = eigenvectors[:, ::-1]
    if np.linalg.det(axes) < 0.0:
        axes[:, 1] *= -1.0
    radii = np.sqrt(2.0 * eigenvalues)
    relative_first = (desired[0] - center) @ axes
    phase = float(np.arctan2(
        relative_first[1] / radii[1], relative_first[0] / radii[0]
    ))
    edges = np.maximum(np.linalg.norm(
        np.roll(outer_xyz, -1, axis=0) - outer_xyz, axis=1
    ), 1e-9)
    progress = np.concatenate(([0.0], np.cumsum(edges[:-1]))) / edges.sum()
    orientation = 1.0 if signed_area(desired) >= 0.0 else -1.0
    base_angles = phase + orientation * (2.0 * np.pi * progress)
    # A single first-point match can twist a long concave boundary around the
    # ellipse. Pick one cyclic phase for the whole ring by minimizing total
    # outer-to-inner travel; the plane normal and insertion axis stay fixed.
    phase_candidates = np.linspace(-np.pi, np.pi, 65)[:-1]
    errors = []
    for offset in phase_candidates:
        trial = center + np.column_stack((
            radii[0] * np.cos(base_angles + offset),
            radii[1] * np.sin(base_angles + offset),
        )) @ axes.T
        errors.append(float(np.mean(np.sum((trial - desired) ** 2, axis=1))))
    phase_offset = float(phase_candidates[int(np.argmin(errors))])
    angles = base_angles + phase_offset
    repaired = center + np.column_stack((
        radii[0] * np.cos(angles), radii[1] * np.sin(angles)
    )) @ axes.T
    return repaired, {
        "strategy": "arclength_parameterized_planar_ellipse",
        "original_projection_crossings": None,
        "projection_intersection_scan_performed": False,
        "ellipse_phase_offset_rad": phase_offset,
        "ellipse_mean_squared_xy_adjustment_mm2": min(errors),
        "max_xy_adjustment_mm": float(np.max(np.linalg.norm(repaired - desired, axis=1))),
    }
