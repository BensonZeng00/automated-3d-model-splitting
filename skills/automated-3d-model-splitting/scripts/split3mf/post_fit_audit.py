"""Bounded, source-local safety measurements for post-fit subtractive repair."""
import numpy as np
from scipy.spatial import cKDTree
import trimesh

from .curved_backing import first_exit, segment_distance
from .assembly_overlap import measure_patch_bounds


def audit_volume_partition(child, result, overlap, *, depth_tolerance_mm=0.0,
                           area_budget_mm2=1.0):
    """Bound independently computed Boolean partition roundoff, not fit clearance."""
    delta = abs(float(result.volume() - child.volume() + overlap.volume()))
    numerical = max(1e-6, abs(float(child.volume())) * 1e-9)
    record = dict(volume_balance_error_mm3=delta, numerical_tolerance_mm3=numerical,
                  valid=delta <= numerical)
    if record['valid'] or delta > 1e-5 or depth_tolerance_mm <= 0:
        return record
    partition = result + overlap
    discrepancy = (child - partition) + (partition - child)
    if str(discrepancy.status()) != 'Error.NoError':
        return record
    raw = discrepancy.to_mesh64()
    mesh = trimesh.Trimesh(np.asarray(raw.vert_properties)[:, :3],
                           np.asarray(raw.tri_verts), process=False)
    bounds = measure_patch_bounds(mesh, depth_tolerance_mm=depth_tolerance_mm,
                                  area_budget_mm2=area_budget_mm2)
    record.update(partition_discrepancy=bounds, roundoff_envelope_cap_mm3=1e-5,
                  valid=bool(bounds['accepted']))
    return record


def nearest_faces(mesh, points):
    """Exact closest triangles, using a conservative centroid/radius broad phase."""
    triangles = np.asarray(mesh.triangles)
    centers = triangles.mean(axis=1)
    tree = cKDTree(centers)
    radius = np.linalg.norm(triangles - centers[:, None], axis=2).max()
    ids, distances = [], []
    for point in np.asarray(points):
        _, seed = tree.query(point)
        seed_point = trimesh.triangles.closest_point(triangles[[seed]], point[None])[0]
        bound = np.linalg.norm(seed_point - point)
        candidates = np.asarray(tree.query_ball_point(point, bound + radius + 1e-10))
        closest = trimesh.triangles.closest_point(
            triangles[candidates], np.broadcast_to(point, (len(candidates), 3)))
        squared = np.sum((closest - point) ** 2, axis=1)
        best = int(np.argmin(squared))
        ids.append(int(candidates[best]))
        distances.append(float(np.sqrt(squared[best])))
    return np.asarray(ids, dtype=np.int64), np.asarray(distances)


def audit_cut_surface(patch, before, after, *, area_budget_mm2=1.0,
                      surface_tolerance_mm=0.05, minimum_thickness_mm=0.45,
                      boundary_band_mm=0.6, sample_count=512):
    """Sample local-normal chords; never advertise a global minimum-thickness proof."""
    if not len(patch.faces) or patch.area <= 0:
        raise ValueError('post-fit repair needs a nonempty source surface')
    cumulative = np.cumsum(patch.area_faces)
    probes = (np.arange(sample_count) + .5) * cumulative[-1] / sample_count
    selected = np.searchsorted(cumulative, probes)
    points = patch.triangles_center[selected]
    _, source_distance = nearest_faces(before, points)
    _, result_distance = nearest_faces(after, points)
    missing = (result_distance > surface_tolerance_mm) & (source_distance <= surface_tolerance_mm)
    edges = patch.edges_unique[np.bincount(patch.edges_unique_inverse) == 1]
    rim_distance = (segment_distance(points, patch.vertices[edges]) if len(edges)
                    else np.full(len(points), np.inf))
    interior = (rim_distance > boundary_band_mm) & (source_distance <= surface_tolerance_mm)
    depths = np.full(sample_count, np.nan)
    for index in np.flatnonzero(interior & ~missing):
        depths[index] = first_exit(after.triangles, points[index:index+1],
                                   -patch.face_normals[selected[index]])[0]
    thin = interior & (np.isnan(depths) | (depths < minimum_thickness_mm))
    weight = float(patch.area / sample_count)
    record = dict(method='equal_area_source_centroid_local_normal_chords',
                  global_minimum_wall_thickness_measured=False,
                  samples=sample_count, interior_samples=int(interior.sum()),
                  boundary_band_excluded_mm=boundary_band_mm,
                  surface_loss_area_estimate_mm2=float(missing.sum() * weight),
                  thin_or_missing_interior_area_estimate_mm2=float(thin.sum() * weight),
                  minimum_sampled_interior_chord_mm=(float(np.nanmin(depths))
                      if np.isfinite(depths).any() else None),
                  minimum_required_mm=minimum_thickness_mm,
                  area_budget_mm2=area_budget_mm2)
    record['valid'] = bool(record['surface_loss_area_estimate_mm2'] <= area_budget_mm2
                           and record['thin_or_missing_interior_area_estimate_mm2'] <= area_budget_mm2
                           and interior.any())
    return record
