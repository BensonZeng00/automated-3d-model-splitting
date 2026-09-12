"""Fit policy: subtract full-size solids, then uniformly shrink inserts."""
from __future__ import annotations

import math
from .overlap_policy import DEFAULT_IGNORE_OVERLAP_RATIO, validate_ignore_overlap_ratio


def configure_uniform_fit(args) -> None:
    validate_ignore_overlap_ratio(getattr(args, 'assembly_ignore_overlap_ratio',
                                           DEFAULT_IGNORE_OVERLAP_RATIO))
    if ((getattr(args, 'post_fit_parent_difference', False) or getattr(args, 'repair_thin_backing', False))
            and args.visual_validation_profile == 'off'):
        raise ValueError('shape repair requires final visual validation')
    scale = args.post_split_uniform_scale
    if not math.isfinite(scale) or not 0.0 < scale < 1.0:
        raise ValueError("--post-split-uniform-scale must be finite and between 0 and 1")
    if args.interface_geometry != "local-connector":
        raise ValueError("post-split fit requires local-connector Boolean cutting")
    # Scaling supplies the fit; never add the old clearance on top of it.
    args.clearance_profile = "fixed"
    for name in ("fit_clearance_mm", "bottom_clearance_mm", "sibling_clearance_mm",
                 "flat_clearance_mm", "clearance_min_mm"):
        setattr(args, name, 0.0)


def exact_unscaled_cutter(mesh):
    """A private copy, without overshoot, caps replacing the surface, or dilation."""
    return mesh.copy(), {
        "strategy": "exact_unscaled_child_then_uniform_shrink",
        "exterior_overshoot_mm": 0.0,
        "additional_clearance_cutters": 0,
    }


def scale_finished_insert(mesh, factor: float):
    import numpy as np

    if not math.isfinite(factor) or not 0.0 < factor < 1.0:
        raise ValueError("uniform factor must be finite and between 0 and 1")
    before = np.asarray(mesh.vertices, dtype=np.float64)
    if not len(before) or not np.isfinite(before).all():
        raise ValueError("cannot scale an empty or non-finite mesh")
    center = (before.min(axis=0) + before.max(axis=0)) * 0.5
    result = mesh.copy()
    result.vertices = center + factor * (before - center)
    return result, {
        "method": "xyz_uniform_after_all_subtractions",
        "factor_xyz": [factor] * 3,
        "center_policy": "own_bounding_box_center",
        "center_mm": center.tolist(),
        "size_before_mm": np.ptp(before, axis=0).tolist(),
        "size_after_mm": np.ptp(result.vertices, axis=0).tolist(),
        "maximum_vertex_displacement_mm": float(np.linalg.norm(result.vertices - before, axis=1).max()),
        "visible_surface_scaled": True,
        "cutters_built_before_scaling": True,
    }
