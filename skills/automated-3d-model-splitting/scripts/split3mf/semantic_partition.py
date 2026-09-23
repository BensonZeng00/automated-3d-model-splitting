"""Apply user-reviewed semantic planes without changing source geometry or material."""
from __future__ import annotations

import numpy as np
import trimesh

from .recognition import make_component_from_global_faces, triangle_areas


def apply_semantic_partitions(vertices, faces, components, specifications, minimum_confidence):
    """Split recognized components by reviewed planes, with topology safeguards."""
    result = list(components)
    records = {"applied": [], "rejected": []}
    areas = triangle_areas(vertices, faces)
    # Descending source indices keep earlier part references stable while a
    # replacement is inserted immediately after its source component.
    for specification in sorted(specifications, key=lambda item: int(item["part_index"]), reverse=True):
        part_index = int(specification["part_index"])
        rejection = {"part_index": part_index, "label": specification.get("label", "")}
        if not bool(specification.get("user_confirmed", False)):
            records["rejected"].append({**rejection, "reason": "user_confirmation_required"})
            continue
        if float(specification.get("confidence_score", 0.0)) < float(minimum_confidence):
            records["rejected"].append({**rejection, "reason": "confidence_below_threshold"})
            continue
        if not 1 <= part_index <= len(result):
            records["rejected"].append({**rejection, "reason": "part_index_out_of_range"})
            continue
        normal = np.asarray(specification.get("normal"), dtype=np.float64)
        origin = np.asarray(specification.get("origin"), dtype=np.float64)
        if normal.shape != (3,) or origin.shape != (3,) or not np.all(np.isfinite(np.r_[normal, origin])):
            records["rejected"].append({**rejection, "reason": "invalid_plane"})
            continue
        length = float(np.linalg.norm(normal))
        if length <= 1e-9:
            records["rejected"].append({**rejection, "reason": "zero_plane_normal"})
            continue
        normal /= length
        component = result[part_index - 1]
        source_ids = np.asarray(component.global_faces, dtype=np.int64)
        centers = np.asarray(vertices)[np.asarray(faces)[source_ids]].mean(axis=1)
        negative = source_ids[((centers - origin) @ normal) <= 0.0]
        positive = source_ids[((centers - origin) @ normal) > 0.0]
        minimum_faces = max(3, int(round(len(source_ids) * 0.03)))
        if min(len(negative), len(positive)) < minimum_faces:
            records["rejected"].append({**rejection, "reason": "insufficient_partition_support"})
            continue
        region_counts = []
        for side in (source_ids, negative, positive):
            local_faces = np.asarray(faces)[side]
            adjacency = trimesh.graph.face_adjacency(faces=local_faces)
            groups = trimesh.graph.connected_components(
                adjacency, nodes=np.arange(len(side)), min_len=1, engine="scipy"
            )
            region_counts.append(len(groups))
        # Dense paint consolidation may intentionally place disconnected
        # same-material islands in one Component.  A valid semantic cut may
        # retain those islands, but it may split at most one existing region.
        if region_counts[1] + region_counts[2] > region_counts[0] + 1:
            records["rejected"].append({**rejection, "reason": "partition_creates_extra_regions"})
            continue
        local_faces = np.asarray(faces)[source_ids]
        adjacency, adjacency_edges = trimesh.graph.face_adjacency(
            faces=local_faces, return_edges=True
        )
        local_positive = ((centers - origin) @ normal) > 0.0
        seam_edges = adjacency_edges[
            local_positive[adjacency[:, 0]] != local_positive[adjacency[:, 1]]
        ]
        if len(seam_edges) < 3:
            records["rejected"].append({**rejection, "reason": "partition_boundary_too_short"})
            continue
        for side in (negative, positive):
            edge_count = {}
            for triangle in np.asarray(faces)[side]:
                for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                    edge = tuple(sorted((int(a), int(b))))
                    edge_count[edge] = edge_count.get(edge, 0) + 1
            boundary_vertices = np.asarray(
                [vertex for edge, count in edge_count.items() if count == 1 for vertex in edge],
                dtype=np.int64,
            )
            _, degree = np.unique(boundary_vertices, return_counts=True)
            if np.any(degree % 2):
                records["rejected"].append({**rejection, "reason": "partition_boundary_not_cycle_decomposable"})
                break
        else:
            degree = None
        if degree is not None:
            continue
        replacements = [
            make_component_from_global_faces(
                vertices, faces, areas, side, component.color_code
            )
            for side in (negative, positive)
        ]
        result[part_index - 1:part_index] = replacements
        records["applied"].append({
            **rejection,
            "normal": normal.tolist(),
            "origin": origin.tolist(),
            "source_face_count": int(len(source_ids)),
            "partition_face_counts": [int(len(negative)), int(len(positive))],
            "boundary_edge_count": int(len(seam_edges)),
            "source_region_count": int(region_counts[0]),
            "partition_region_counts": [int(region_counts[1]), int(region_counts[2])],
            "source_material_preserved": True,
        })
    return result, records
