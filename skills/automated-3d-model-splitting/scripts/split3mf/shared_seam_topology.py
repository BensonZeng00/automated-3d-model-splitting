"""Classify source-ID boundary sharing without conflating it with nonmanifoldness."""
from collections import Counter, defaultdict
import numpy as np
from .planar_arc import PlanarArcError


def canonical_loop(loop):
    ids = tuple(int(i) for i in loop)
    if len(ids) < 3 or len(set(ids)) != len(ids):
        raise PlanarArcError('Shared seam loop requires at least three unique source IDs')
    offset = ids.index(min(ids))
    forward = ids[offset:] + ids[:offset]
    reverse = (forward[0],) + tuple(reversed(forward[1:]))
    return min(forward, reverse)


def classify_shared_seams(vertices, faces, loops):
    points, triangles = np.asarray(vertices), np.asarray(faces, dtype=np.int64)
    canonical = sorted(set(canonical_loop(loop) for loop in loops))
    occurrences, edge_owners, neighbors = Counter(), defaultdict(set), defaultdict(set)
    for index, loop in enumerate(canonical):
        if min(loop) < 0 or max(loop) >= len(points):
            raise PlanarArcError('Shared seam source ID is outside current mesh')
        occurrences.update(loop)
        for a, b in zip(loop, loop[1:] + loop[:1]):
            edge_owners[tuple(sorted((a, b)))].add(index)
            neighbors[a].add(b); neighbors[b].add(a)
    shared_edges = sorted(edge for edge, owners in edge_owners.items() if len(owners) > 1)
    shared_ids = sorted(i for i, count in occurrences.items() if count > 1)
    if any(len(owners) > 2 for owners in edge_owners.values()):
        raise PlanarArcError('More than two distinct region loops own one source edge')
    if shared_edges:
        # Count actual triangle incidence, independently of the loop records.
        edges = np.sort(np.concatenate((triangles[:,[0,1]], triangles[:,[1,2]], triangles[:,[2,0]])), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        dtype = np.dtype((np.void, unique.dtype.itemsize * 2))
        keys = np.ascontiguousarray(unique).view(dtype).ravel()
        order = np.argsort(keys); ordered = keys[order]
        wanted = np.ascontiguousarray(shared_edges, dtype=unique.dtype).view(dtype).ravel()
        positions = np.searchsorted(ordered, wanted)
        if np.any(positions >= len(ordered)):
            raise PlanarArcError('Shared seam edge is absent from the current source')
        if np.any(ordered[positions] != wanted) or np.any(counts[order[positions]] != 2):
            raise PlanarArcError('Shared seam must have exactly two source triangle owners')
    affected = np.any(np.isin(triangles, shared_ids), axis=1)
    patch = points[triangles[affected]]
    area = float(np.linalg.norm(np.cross(patch[:,1]-patch[:,0], patch[:,2]-patch[:,0]),axis=1).sum()/2)
    length = sum(float(np.linalg.norm(points[a]-points[b])) for a,b in shared_edges)
    return canonical, neighbors, dict(
        shared_edges=shared_edges, shared_source_ids=shared_ids,
        shared_edge_count=len(shared_edges), shared_vertex_count=len(shared_ids),
        duplicate_loop_records=len(loops)-len(canonical),
        shared_edge_length_mm=length, affected_source_area_mm2=area,
        source_face_count=len(triangles), classification='source_id_shared_region_boundary')


def validate_region_boundary_preservation(before, after, loops, groups):
    """Every planned seam must retain its incident source-region identities."""
    wanted = sorted({tuple(sorted((a,b))) for loop in loops
                     for a,b in zip(loop, loop[1:]+loop[:1])})
    dtype = np.dtype((np.void, 16))
    requested = np.ascontiguousarray(wanted, dtype=np.int64).view(dtype).ravel()
    signatures = []
    for triangles in (before, after):
        edges = np.sort(np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])), axis=1)
        keys = np.ascontiguousarray(edges, dtype=np.int64).view(dtype).ravel()
        order = np.argsort(keys); keys = keys[order]
        owners = np.tile(groups,3)[order]
        left, right = np.searchsorted(keys, requested, 'left'), np.searchsorted(keys, requested, 'right')
        signatures.append([tuple(sorted(owners[a:b].tolist())) for a,b in zip(left,right)])
    if any(not original or original != result for original,result in zip(*signatures)):
        raise PlanarArcError('Shared surface changed a planned seam or its incident regions')
