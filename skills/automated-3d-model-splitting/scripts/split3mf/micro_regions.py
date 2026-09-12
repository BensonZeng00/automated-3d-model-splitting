"""Absorb <=2 mm material regions into an adjacent physical part."""
import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from .tolerance_policy import maximum_span
from .recognition import make_component_from_global_faces, triangle_areas


def merge_micro_regions(vertices, faces, components, threshold=2.0):
    owners = np.full(len(faces), -1, dtype=int)
    for i, component in enumerate(components):
        owners[component.global_faces] = i
    adjacency = trimesh.graph.face_adjacency(faces=faces)
    a, b = adjacency.T if len(adjacency) else ([], [])
    graph = csr_matrix((np.ones(2 * len(a)),
                        (np.r_[a, b].astype(int), np.r_[b, a].astype(int))),
                       shape=(len(faces), len(faces)))
    part_faces = {i: np.asarray(c.global_faces) for i, c in enumerate(components)}
    areas = triangle_areas(vertices, faces)
    records = []
    merged_face = {}
    changed_owners = set()
    for source_id, component in enumerate(components):
        ids = part_faces[source_id]
        if not len(ids):
            continue
        points = vertices[np.unique(faces[ids])]
        span = maximum_span(points, threshold)
        if span > threshold:
            continue
        neighbors = np.unique(owners[graph[ids].indices])
        neighbors = neighbors[(neighbors >= 0) & (neighbors != source_id)]
        if not len(neighbors):
            records.append(dict(source_part=source_id + 1, span_mm=span,
                                action='kept', reason='no-shared-edge-neighbor'))
            continue
        center = points.mean(axis=0)
        candidates = []
        for neighbor in neighbors:
            neighbor_faces = part_faces[int(neighbor)]
            neighbor_points = vertices[np.unique(faces[neighbor_faces])]
            distance = float(np.linalg.norm(neighbor_points - center, axis=1).min())
            candidates.append((distance, int(neighbor)))
        distance, target_id = min(candidates)
        merged_face[source_id + 1] = int(ids[0])
        owners[ids] = target_id
        part_faces[target_id] = np.r_[part_faces[target_id], ids]
        part_faces[source_id] = np.empty(0, dtype=int)
        changed_owners.update((source_id, target_id))
        records.append(dict(source_part=source_id + 1, target_original_part=target_id + 1,
                            span_mm=span, faces=len(ids), distance_mm=distance,
                            action='merged', material_faces_preserved=True))
    if not changed_owners:
        return components, records
    result = []
    old_to_new = {}
    for old_id, component in enumerate(components):
        ids = np.flatnonzero(owners == old_id)
        if not len(ids):
            continue
        updated = make_component_from_global_faces(vertices, faces, areas, ids, component.color_code)
        result.append(updated)
        old_to_new[old_id + 1] = len(result)
    for record in records:
        target = record.get('target_original_part')
        if target is not None:
            record['target_part'] = old_to_new[int(owners[merged_face[record['source_part']]]) + 1]
    return result, records
