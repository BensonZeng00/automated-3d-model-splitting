"""Close small source openings into adjacent parts without separate connectors."""
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from .mesh import boundary_loops, triangulate_ordered_loop_3d
from .recognition import make_component_from_global_faces, triangle_areas
from .tolerance_policy import maximum_span


def _oriented_edges(triangles):
    return [(int(a), int(b)) for tri in triangles
            for a, b in zip(tri, np.roll(tri, -1))]


def seal_micro_openings(vertices, faces, colors, components, threshold=2.0):
    """Append audited cap triangles; keep every original vertex/face/color.

    The cap inherits the nearest incident source face's color and part. Any
    failed cap is left unchanged and reported, not declared repaired.
    """
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    sorted_edges = np.sort(edges, axis=1)
    keys = sorted_edges[:, 0] * len(vertices) + sorted_edges[:, 1]
    unique_keys, first, counts = np.unique(keys, return_index=True, return_counts=True)
    open_rows = first[counts == 1]
    open_edges = {tuple(sorted_edges[i]): (tuple(edges[i]), int(i // 3)) for i in open_rows}
    owners = np.full(len(faces), -1, dtype=int)
    for index, component in enumerate(components):
        owners[component.global_faces] = index
    appended, appended_colors, appended_owners, records = [], [], [], []
    used_new_edges = Counter()
    for loop in boundary_loops(faces):
        points = vertices[np.asarray(loop)]
        span = maximum_span(points, threshold)
        if span > threshold:
            continue
        rim_keys = [tuple(sorted((a, b))) for a, b in zip(loop, np.roll(loop, -1))]
        if any(key not in open_edges for key in rim_keys):
            continue
        neighbors = np.unique([open_edges[key][1] for key in rim_keys])
        if np.any(owners[neighbors] < 0):
            records.append(dict(span_mm=span, action='unresolved', reason='unowned-neighbor'))
            continue
        neighbor_triangles = vertices[faces[neighbors]]
        normal = np.cross(neighbor_triangles[:, 1] - neighbor_triangles[:, 0],
                          neighbor_triangles[:, 2] - neighbor_triangles[:, 0]).sum(axis=0)
        try:
            cap_local, _, _ = triangulate_ordered_loop_3d(points, normal)
            cap = np.asarray(loop)[np.asarray(cap_local, dtype=int)]
            if cap.shape != (len(loop) - 2, 3):
                raise ValueError('incomplete-boundary-triangulation')
            directed = _oriented_edges(cap)
            first_rim = rim_keys[0]
            source_direction = open_edges[first_rim][0]
            if source_direction in directed:
                cap = cap[:, ::-1]
                directed = _oriented_edges(cap)
            rim_set = set(rim_keys)
            cap_counts = Counter(tuple(sorted(edge)) for edge in directed)
            if {edge for edge, count in cap_counts.items() if count == 1} != rim_set:
                raise ValueError('cap-boundary-mismatch')
            for edge, count in cap_counts.items():
                if edge in rim_set:
                    if count != 1 or open_edges[edge][0] in directed:
                        raise ValueError('inconsistent-rim-winding')
                else:
                    key = edge[0] * len(vertices) + edge[1]
                    position = np.searchsorted(unique_keys, key)
                    occupied = position < len(unique_keys) and unique_keys[position] == key
                    if occupied or used_new_edges[edge] or count != 2:
                        raise ValueError('occupied-or-nonmanifold-cap-edge')
                    if (edge[0], edge[1]) not in directed or (edge[1], edge[0]) not in directed:
                        raise ValueError('inconsistent-cap-winding')
            xyz = vertices[cap]
            area2 = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
            if np.any(area2 <= 1e-12):
                raise ValueError('degenerate-cap')
        except ValueError as exc:
            records.append(dict(span_mm=span, vertices=len(loop), action='unresolved', reason=str(exc)))
            continue
        nearest = cKDTree(neighbor_triangles.mean(axis=1)).query(xyz.mean(axis=1))[1]
        nearest_faces = neighbors[nearest]
        appended.extend(cap.tolist())
        appended_colors.extend(str(colors[i]) for i in nearest_faces)
        appended_owners.extend(int(owners[i]) for i in nearest_faces)
        used_new_edges.update(cap_counts)
        records.append(dict(span_mm=span, vertices=len(loop), cap_faces=len(cap),
                            action='merged-into-adjacent-surface', separate_connector=False,
                            target_parts=(np.unique(owners[nearest_faces]) + 1).tolist()))
    if not appended:
        return faces, list(colors), components, records
    updated_faces = np.vstack((faces, np.asarray(appended, dtype=int)))
    updated_colors = list(colors) + appended_colors
    updated_owners = np.r_[owners, appended_owners]
    areas = triangle_areas(vertices, updated_faces)
    updated_components = [make_component_from_global_faces(
        vertices, updated_faces, areas, np.flatnonzero(updated_owners == i), component.color_code)
        for i, component in enumerate(components)]
    return updated_faces, updated_colors, updated_components, records
