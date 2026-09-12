"""Directional, area-weighted evidence; never claim a global minimum wall."""
import numpy as np
from .curved_backing import first_exit, segment_distance
from .local_ray_probe import LocalRayProbe


def audit_local_backing(patch, solid, *, boundary_band_mm=.6, minimum_mm=.45,
                        scale_factor=.99, area_budget_mm2=1.):
    """Check every interior source-face centroid, with no global-axis filtering.

    This is a finite local-normal screen, not a global minimum-wall proof.
    No exit within the bound certifies only that ray segment on a valid solid.
    """
    if not solid.is_watertight or not solid.is_winding_consistent or solid.volume <= 0:
        raise ValueError('Backing thickness requires an oriented closed solid')
    if not 0 < scale_factor <= 1 or minimum_mm <= 0 or boundary_band_mm < 0 or area_budget_mm2 < 0:
        raise ValueError('Invalid backing thickness policy')
    rim = patch.edges_unique[np.bincount(patch.edges_unique_inverse) == 1]
    distance = (segment_distance(patch.triangles_center, patch.vertices[rim]) if len(rim)
                else np.full(len(patch.faces), np.inf))
    interior = distance > boundary_band_mm
    bound = minimum_mm/scale_factor
    depths, hits = LocalRayProbe(solid).exits(patch.triangles_center[interior],
                                            -patch.face_normals[interior], bound)
    thin = np.isfinite(depths) & (depths < bound-1e-8)
    area = float(patch.area_faces[interior][thin].sum())
    return dict(method='all_source_face_centroids_local_normal_bounded_exit',
        global_minimum_wall_thickness_measured=False,
        source_faces=len(patch.faces), interior_samples=int(interior.sum()),
        boundary_band_excluded_mm=boundary_band_mm, minimum_after_scale_mm=minimum_mm,
        scale_factor=scale_factor, tested_unscaled_depth_mm=bound,
        thin_interior_area_mm2=area, thin_interior_faces=int(thin.sum()),
        thin_exits_on_source=int(np.sum(thin & (hits < len(patch.faces)))),
        thin_exits_on_backing=int(np.sum(thin & (hits >= len(patch.faces)))),
        smallest_failing_chord_mm=float(depths[thin].min()) if thin.any() else None,
        area_budget_mm2=area_budget_mm2, valid=area <= area_budget_mm2,
        status='evaluated' if interior.any() else 'no_interior_outside_taper_band')


def audit_backing(patch, solid, inward, *, boundary_band_mm=.6, minimum_mm=.45):
    axis = np.asarray(inward,dtype=float)
    axis /= np.linalg.norm(axis)
    points = patch.triangles_center
    eligible = (patch.face_normals @ -axis) > .5
    thickness = first_exit(solid.triangles,points,axis)
    counts = np.bincount(patch.edges_unique_inverse)
    rim = patch.edges_unique[counts==1]
    distance = segment_distance(points,patch.vertices[rim])
    measured = eligible & np.isfinite(thickness)
    interior = measured & (distance>boundary_band_mm)
    thin = interior & (thickness<minimum_mm)
    return {
        'method':'source_face_centroid_directional_chord',
        'global_minimum_wall_thickness_measured':False,
        'checked_area_mm2':float(patch.area_faces[measured].sum()),
        'eligible_area_mm2':float(patch.area_faces[eligible].sum()),
        'missing_eligible_faces':int((eligible&~np.isfinite(thickness)).sum()),
        'boundary_band_mm':boundary_band_mm,'minimum_backing_mm':minimum_mm,
        'thin_interior_area_mm2':float(patch.area_faces[thin].sum()),
        'thin_below_0_1_area_mm2':float(patch.area_faces[measured&(thickness<.1)].sum()),
        'minimum_interior_chord_mm':float(thickness[interior].min()) if interior.any() else None,
        'taper_band_excluded_from_load_bearing_test':True,
    }
