"""Restore tiny source-boundary ears skipped by a generated closure chord."""
import numpy as np
import trimesh

from .micro_mesh_repair import edge_topology
from .print_tolerance import small_patch_report


def _source_ear_path(faces, protected_count, edge_owners, chord, seed, open_edges):
    """Find a bounded source patch whose remaining boundary is an open path."""
    patch = {int(seed)}
    for _ in range(4):
        counts = {}
        for index in patch:
            face = faces[index]
            for a, b in zip(face, face[1:] + face[:1]):
                edge = tuple(sorted((a, b)))
                counts[edge] = counts.get(edge, 0) + 1
        boundary = {edge for edge, count in counts.items() if count == 1}
        unresolved = boundary - open_edges - {chord}
        if not unresolved:
            path = [chord[0]]
            remaining = boundary - {chord}
            while remaining:
                choices = [edge for edge in remaining if path[-1] in edge]
                if len(choices) != 1:
                    return None
                edge = choices[0]
                path.append(edge[1] if edge[0] == path[-1] else edge[0])
                remaining.remove(edge)
            return (path, sorted(patch)) if path[-1] == chord[1] else None
        for edge in unresolved:
            adjacent = [int(i) for i in edge_owners.get(edge, ())
                        if int(i) < protected_count and int(i) not in patch]
            if len(adjacent) != 1:
                return None
            patch.add(adjacent[0])
        if len(patch) > 4:
            return None
    return None


def restore_source_ears(mesh, protected_count):
    vertices = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.faces).copy().tolist()
    original = np.asarray(mesh.faces)
    affected = []
    restored = []
    accepted_evidence = None
    for _ in range(32):
        edges, owners, unique, inverse, counts, _, order, starts = edge_topology(faces)
        open_edges = {tuple(edge) for edge in unique[counts == 1]}
        edge_owners = {tuple(edge): owners[order[starts[i]:starts[i+1]]]
                       for i, edge in enumerate(unique)}
        selected = None
        for index in np.flatnonzero(counts == 3):
            left, right = map(int, unique[index])
            incident = owners[order[starts[index]:starts[index + 1]]]
            generated = [int(face) for face in incident if face >= protected_count]
            if len(generated) != 1:
                continue
            for source in incident[incident < protected_count]:
                candidate = _source_ear_path(faces, protected_count, edge_owners,
                                             (left, right), source, open_edges)
                if candidate is None:
                    continue
                path, source_patch = candidate
                target = generated[0]
                triangle = faces[target]
                if any(vertex in triangle for vertex in path[1:-1]):
                    continue
                corner = next(int(vertex) for vertex in triangle if vertex not in (left, right))
                position = next(k for k in range(3)
                                if {triangle[k], triangle[(k + 1) % 3]} == {left, right})
                a, b = triangle[position], triangle[(position + 1) % 3]
                if path[0] != a:
                    path = path[::-1]
                # Keep the broad triangle exactly planar outside a tiny collar.
                # Moving its entire diagonal to the ear would alter a large
                # backing patch even though the missing rim is microscopic.
                midpoint = (vertices[a] + vertices[b]) * .5
                travel = vertices[corner] - midpoint
                distance = float(np.linalg.norm(travel))
                if distance <= 1e-12:
                    continue
                width = min(.05, float(np.linalg.norm(vertices[a] - vertices[b])) * .25)
                fraction = min(.25, width / distance)
                center = midpoint + fraction * travel
                center_id = len(vertices)
                candidate_vertices = np.vstack((vertices, center))
                collar = [[first, second, center_id] for first, second in zip(path, path[1:])]
                replacements = collar + [[b, corner, center_id], [corner, a, center_id]]
                patch = affected + [faces[i] for i in source_patch] + [[a, b, center_id]] + collar
                evidence = small_patch_report(candidate_vertices, patch, maximum_span_mm=5.,
                                              separate_components=True)
                points = candidate_vertices[np.asarray(replacements)]
                areas = np.linalg.norm(np.cross(points[:, 1] - points[:, 0],
                                                points[:, 2] - points[:, 0]), axis=1) * .5
                if not evidence['accepted'] or np.any(areas <= 1e-12):
                    continue
                selected = target, replacements, patch, source_patch, evidence, candidate_vertices
                break
            if selected is not None:
                break
        if selected is None:
            break
        target, replacements, affected, source, evidence, vertices = selected
        accepted_evidence = evidence
        faces[target] = replacements[0]
        faces.extend(replacements[1:])
        restored.extend(source)
    if not restored:
        return mesh, {'applied': False}
    result = trimesh.Trimesh(vertices=vertices.copy(), faces=np.asarray(faces),
                             process=False, metadata=mesh.metadata.copy())
    # This operation may only change generated closure faces.
    if not np.array_equal(result.faces[:protected_count], original[:protected_count]):
        raise ValueError('source-ear restoration changed protected source faces')
    record = dict(applied=True, restored_source_ears=restored,
                  generated_faces_added=len(faces) - len(original), **accepted_evidence)
    result.metadata['source_ear_restoration'] = record
    return result, record
