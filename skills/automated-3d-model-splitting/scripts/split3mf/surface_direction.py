"""Check connector material-side selection against its oriented source rim."""
import numpy as np


def audit_inward_axis(vertices, faces, boundary_ids, inward):
    edges = {tuple(sorted((int(a), int(b)))) for a, b in
             zip(boundary_ids, boundary_ids[1:] + boundary_ids[:1])}
    selected = [face for face in faces if any(
        tuple(sorted((int(a), int(b)))) in edges
        for a, b in zip(face, face[1:] + face[:1]))]
    if not selected:
        return {'status': 'not_evaluated', 'source_rim_faces': 0}
    triangles = np.asarray(vertices, dtype=float)[np.asarray(selected, dtype=int)]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    average = normals.sum(axis=0)
    length = float(np.linalg.norm(average))
    area = float(np.linalg.norm(normals, axis=1).sum())
    coherence = length / max(area, 1e-30)
    direction = np.asarray(inward, dtype=float)
    cosine = float(average @ direction / max(length*np.linalg.norm(direction), 1e-30))
    record = {'status': 'measured', 'source_rim_faces': len(selected),
              'normal_coherence': coherence, 'outward_dot_inward': cosine}
    if coherence > 0.5 and cosine > 1e-6:
        raise ValueError('connector inward axis points outside the oriented source surface: '
                         f'outward_dot_inward={cosine:.6f}, source_rim_faces={len(selected)}')
    return record
