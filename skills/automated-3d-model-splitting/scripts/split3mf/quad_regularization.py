"""Batch immutable quad geometry; preserve the original ordered flip decisions."""
import numpy as np
from .connector_geometry import project_connector_points


def quality(triangles):
    edges = np.roll(triangles, -1, axis=1) - triangles
    squared = np.einsum('nij,nij->n', edges, edges)
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    area = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    return np.divide(2 * np.sqrt(3.) * area, squared,
                     out=np.zeros_like(area), where=squared > 1e-24)


def side(a, b, c):
    delta, relative = b - a, c - a
    return delta[:, 0] * relative[:, 1] - delta[:, 1] * relative[:, 0]


def quad_candidates(vertices, projected, faces, edges, owners, limit):
    a, b = edges.T
    first, second = faces[owners[:, 0]], faces[owners[:, 1]]
    c = np.where((first != a[:, None]) & (first != b[:, None]), first, -1).max(axis=1)
    d = np.where((second != a[:, None]) & (second != b[:, None]), second, -1).max(axis=1)
    pa, pb, pc, pd = projected[a], projected[b], projected[c], projected[d]
    valid = ((c >= 0) & (d >= 0) & (c != d)
             & (side(pa, pb, pc) * side(pa, pb, pd) < -1e-14)
             & (side(pc, pd, pa) * side(pc, pd, pb) < -1e-14))
    candidate = np.stack((np.stack((c, d, a), axis=1),
                          np.stack((d, c, b), axis=1)), axis=1)
    triangles = vertices[candidate]
    normals = np.cross(triangles[:, :, 1] - triangles[:, :, 0],
                       triangles[:, :, 2] - triangles[:, :, 0])
    norm = np.linalg.norm(normals, axis=2)
    valid &= (norm > 1e-8).all(axis=1)
    source = vertices[np.stack((first, second), axis=1)]
    reference = np.cross(source[:, :, 1] - source[:, :, 0],
                         source[:, :, 2] - source[:, :, 0]).sum(axis=1)
    reverse = (normals * reference[:, None, :]).sum(axis=2) < 0
    for index in (0, 1):
        ids = np.flatnonzero(reverse[:, index])
        candidate[ids, index, 1:3] = candidate[ids, index, 1:3][:, ::-1]
    normals = normals * np.where(reverse[:, :, None], -1., 1.)
    lengths = np.linalg.norm(np.roll(triangles, -1, axis=2) - triangles, axis=3)
    with np.errstate(divide='ignore', invalid='ignore'):
        aspect = lengths.max(axis=2) ** 2 / norm
        normal_dot = (normals[:, 0] * normals[:, 1]).sum(axis=1) / (norm[:, 0] * norm[:, 1])
    valid &= (aspect.max(axis=1) < 500.) & (normal_dot >= -0.05)
    current_quality = np.minimum(quality(projected[first]), quality(projected[second]))
    candidate_quality = np.minimum(quality(projected[candidate[:, 0]]),
                                   quality(projected[candidate[:, 1]]))
    old_length = np.linalg.norm(pa - pb, axis=1)
    new_length = np.linalg.norm(pc - pd, axis=1)
    valid &= np.linalg.norm(vertices[c] - vertices[d], axis=1) <= limit + 1e-9
    valid &= (candidate_quality > current_quality + 1e-4) | (new_length < old_length * .97)
    return candidate, np.stack((c, d), axis=1), valid


def regularize(*, vertices, faces, boundary_edges, plan,
               maximum_internal_edge_mm, passes=6):
    points = np.asarray(vertices, dtype=np.float64)
    result = np.asarray(faces, dtype=np.int64).copy()
    vertex_ids = np.unique(result)
    projected = np.zeros((len(points), 2))
    projected[vertex_ids] = project_connector_points(points[vertex_ids], plan)
    adjacency = {}
    for left, right in boundary_edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    components = {}
    for seed in sorted(adjacency):
        if seed in components:
            continue
        component_id = len(components) + 1
        pending = [seed]
        while pending:
            current = pending.pop()
            if current in components:
                continue
            components[current] = component_id
            pending.extend(adjacency.get(current, ()))
    flips = 0
    for iteration in range(max(0, int(passes))):
        edge_faces = {}
        for face_id, (a, b, c) in enumerate(result.tolist()):
            for left, right in ((a, b), (b, c), (c, a)):
                edge_faces.setdefault(tuple(sorted((left, right))), []).append(face_id)
        eligible = [(edge, owner) for edge, owner in sorted(edge_faces.items())
                    if edge not in boundary_edges and len(owner) == 2]
        if not eligible:
            break
        edges = np.asarray([entry[0] for entry in eligible], dtype=np.int64)
        owners = np.asarray([entry[1] for entry in eligible], dtype=np.int64)
        candidates, replacements, valid = quad_candidates(
            points, projected, result, edges, owners, maximum_internal_edge_mm)
        occupied, locked, changed = set(edge_faces), set(), 0
        for index in np.flatnonzero(valid):
            first_id, second_id = map(int, owners[index])
            if first_id in locked or second_id in locked:
                continue
            c, d = map(int, replacements[index])
            replacement = tuple(sorted((c, d)))
            if replacement in occupied:
                continue
            if c in components and d in components and components[c] == components[d]:
                continue
            result[first_id], result[second_id] = candidates[index]
            locked.update((first_id, second_id))
            occupied.discard(tuple(map(int, edges[index])))
            occupied.add(replacement)
            changed += 1
        flips += changed
        print(f'connector_regularize_pass={iteration + 1} faces={len(result)} flips={changed}', flush=True)
        if not changed:
            break
    return result.tolist(), flips
