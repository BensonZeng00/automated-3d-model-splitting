"""Accept traced subdivision and bounded print-scale source surface changes."""
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from .print_tolerance import current, small_patch_report, triangle_areas
from .subdivision_proof import prove_edge_subdivision


def face_key(points):
    return tuple(sorted(map(tuple, np.round(points, 10))))


def audit_replaced_surface(source_inventory, final_inventory, result):
    missing = Counter({k: n-final_inventory.get(k,0) for k,n in source_inventory.items()
                       if n > final_inventory.get(k,0)})
    original_count = sum(missing.values())
    equivalent = 0
    subdivision_proofs = []
    credited_replacements = Counter()
    for item in result.metadata.get('boundary_closure_subdivisions', []):
        old = np.asarray(item['source_triangle'])
        new = np.asarray(item['replacement_triangles'])
        key = face_key(old)
        if missing.get(key,0) <= 0:
            continue
        normal = np.cross(old[1]-old[0], old[2]-old[0])
        norm = np.linalg.norm(normal)
        if norm <= 1e-15:
            continue
        plane_error = np.abs((new.reshape(-1,3)-old[0]) @ (normal/norm)).max()
        area_error = abs(float(triangle_areas(new).sum()) - norm*0.5)
        new_counts = Counter(face_key(t) for t in new)
        present = all(final_inventory.get(k,0) >= n+credited_replacements[k]
                      for k,n in new_counts.items())
        strict_match = plane_error <= 1e-8 and area_error <= max(1e-10,norm*1e-7)
        distance_limit = (min(current().surface_distance_mm, 0.001)
                          if current().micro_area_mm2 > 0 else 0.0)
        # Require a complete non-overlapping edge fan even if aggregate areas
        # happen to match. Proof concerns the actual faces still in the result.
        proof = (prove_edge_subdivision(old, new, 1e-8 if strict_match else distance_limit)
                 if present else None)
        if proof is not None:
            missing[key] -= 1
            equivalent += 1
            subdivision_proofs.append(proof)
            credited_replacements.update(new_counts)
    remaining = np.asarray([key for key,n in missing.items() for _ in range(n)], dtype=float)
    report = dict(original_face_matches_missing=original_count,
                  equivalent_subdivided_source_faces=equivalent,
                  subdivision_proofs=subdivision_proofs,
                  print_tolerance_accepted_source_faces=0, accepted=not len(remaining))
    if not len(remaining):
        return report
    patch = small_patch_report(remaining.reshape(-1,3), np.arange(remaining.size//3).reshape(-1,3))
    report.update(patch)
    if not patch['accepted']:
        return report
    # Local nearest-triangle distances include vertices, edge midpoints and centroids.
    triangles = np.asarray(result.triangles)
    tree = cKDTree(triangles.mean(axis=1))
    samples = np.concatenate([remaining.reshape(-1,3), remaining.mean(axis=1),
                              ((remaining+np.roll(remaining,-1,axis=1))*0.5).reshape(-1,3)])
    import trimesh
    distances = []
    for start in range(0,len(samples),256):
        points = samples[start:start+256]
        _, ids = tree.query(points,k=min(64,len(triangles)))
        ids = np.asarray(ids).reshape(len(points),-1)
        repeated = np.repeat(points,ids.shape[1],axis=0)
        nearest = trimesh.triangles.closest_point(triangles[ids.ravel()],repeated)
        distances.extend(np.linalg.norm(nearest-repeated,axis=1).reshape(len(points),-1).min(axis=1))
    maximum = float(max(distances,default=float('inf')))
    report.update(maximum_sampled_surface_distance_mm=maximum,
                  surface_distance_limit_mm=current().surface_distance_mm,
                  accepted=maximum <= current().surface_distance_mm)
    if report['accepted']:
        report['print_tolerance_accepted_source_faces'] = len(remaining)
    return report
