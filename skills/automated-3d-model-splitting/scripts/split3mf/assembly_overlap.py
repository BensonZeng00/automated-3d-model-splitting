"""Conservative physical bounds for small connected intersection patches."""
import numpy as np
import trimesh
from .local_connectors import _manifold64
from .overlap_policy import (overlap_is_ignored, validate_ignore_overlap_threshold,
                             ratio_volume_limit, ratio_measurement)


def measure_overlap(first, second, *, depth_tolerance_mm=0.0,
                    area_budget_mm2=1.0, volume_tolerance_mm3=1e-8,
                    ignore_overlap_below_mm3=0.0, cutting_volume_mm3=None,
                    ignore_overlap_ratio=0.0):
    validate_ignore_overlap_threshold(ignore_overlap_below_mm3)
    ratio_limit = ratio_volume_limit(cutting_volume_mm3, ignore_overlap_ratio)
    if not np.isfinite(depth_tolerance_mm) or depth_tolerance_mm < 0:
        raise ValueError('depth tolerance must be finite and nonnegative')
    intersection = _manifold64(first) ^ _manifold64(second)
    if str(intersection.status()) != 'Error.NoError':
        raise ValueError(f'intersection failed: {intersection.status()}')
    volume = abs(float(intersection.volume()))
    record = dict(intersection_mm3=volume, accepted=volume <= volume_tolerance_mm3,
                  depth_tolerance_mm=depth_tolerance_mm, components=[])
    record.update(ratio_measurement(volume, cutting_volume_mm3, ignore_overlap_ratio))
    if overlap_is_ignored(volume, ratio_limit):
        record['accepted'] = True
    if overlap_is_ignored(volume, ignore_overlap_below_mm3):
        record.update(accepted=True, ignored_by_volume_policy=True,
                      ignore_overlap_below_mm3=ignore_overlap_below_mm3)
    if record['accepted']:
        return record
    raw = intersection.to_mesh64()
    mesh = trimesh.Trimesh(np.asarray(raw.vert_properties)[:, :3],
                           np.asarray(raw.tri_verts), process=False)
    return dict(record, **measure_patch_bounds(mesh, depth_tolerance_mm=depth_tolerance_mm,
                                               area_budget_mm2=area_budget_mm2))


def measure_patch_bounds(mesh, *, depth_tolerance_mm, area_budget_mm2):
    """Bound every connected surface patch, including roundoff partition seams."""
    record = {'components': []}
    area = float(mesh.area)
    bounds = []
    for component in mesh.split(only_watertight=False):
        points = np.asarray(component.vertices)
        # A containing slab supplies a conservative thickness bound; this is
        # not a sampled maximum penetration depth or a volume-to-length conversion.
        centered = points-points.mean(axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        width = float(np.ptp(centered @ axes.T, axis=0).min())
        pca_width = width
        # PCA minimizes variance, not containing-slab width. A real face normal
        # can give a tighter, still conservative bound for a skewed micro-patch.
        # Bound work; missing a better plane can only reject, never falsely pass.
        if area <= area_budget_mm2 and width > depth_tolerance_mm > 0:
            normals = np.asarray(component.face_normals)
            normals = normals[np.linalg.norm(normals, axis=1) > .5]
            if len(normals) > 256:
                normals = normals[np.linspace(0, len(normals)-1, 256, dtype=int)]
            for start in range(0, len(normals), 64):
                batch = normals[start:start+64]
                batch = batch/np.linalg.norm(batch, axis=1)[:, None]
                width = min(width, float(np.ptp(centered @ batch.T, axis=0).min()))
                if width <= depth_tolerance_mm:
                    break
        bounds.append(width)
        record['components'].append(dict(slab_thickness_bound_mm=width,
                                         pca_slab_thickness_bound_mm=pca_width,
                                         surface_area_mm2=float(component.area)))
    record.update(intersection_surface_area_mm2=area,
                  maximum_slab_thickness_bound_mm=max(bounds, default=float('inf')))
    record['accepted'] = bool(bounds and depth_tolerance_mm > 0
                              and max(bounds) <= depth_tolerance_mm
                              and area <= area_budget_mm2)
    return record


def union_mesh(meshes):
    solids = iter(_manifold64(mesh) for mesh in meshes)
    solid = next(solids)
    for other in solids:
        solid = solid + other
    if str(solid.status()) != 'Error.NoError':
        raise ValueError(f'assembly union failed: {solid.status()}')
    raw = solid.to_mesh64()
    return trimesh.Trimesh(np.asarray(raw.vert_properties)[:, :3],
                           np.asarray(raw.tri_verts), process=False)
