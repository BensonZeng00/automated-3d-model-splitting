"""Apply user-reviewed semantic planes without changing source geometry or material."""
from __future__ import annotations

import numpy as np
import trimesh

from .recognition import (
    connected_face_regions,
    make_component_from_global_faces,
    triangle_areas,
)


def apply_semantic_partitions(vertices, faces, components, specifications, minimum_confidence):
    """Split recognized components by reviewed planes, with topology safeguards."""
    result = list(components)
    records = {"applied": [], "rejected": []}
    areas = triangle_areas(vertices, faces)
    expanded_specifications = []
    for specification in specifications:
        if str(specification.get("scope", "part")).strip().lower() == "all_components":
            # A user-directed global section describes two physical sides of
            # one painted shell. Combine recognized material regions into one
            # temporary ownership component so its colors are cut together
            # and the resulting interface remains a single geometric seam.
            expanded_specifications.append({
                **specification,
                "part_index": 1,
                "_combine_all_components": True,
            })
        else:
            expanded_specifications.append(specification)
    # Descending source indices keep earlier part references stable while a
    # replacement is inserted immediately after its source component.
    for specification in sorted(expanded_specifications, key=lambda item: int(item["part_index"]), reverse=True):
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
        combined_component_count = 1
        if bool(specification.get("_combine_all_components", False)):
            combined_component_count = len(result)
            source_ids = np.unique(np.concatenate([
                np.asarray(component.global_faces, dtype=np.int64)
                for component in result
            ]))
            combined = make_component_from_global_faces(
                vertices, faces, areas, source_ids, result[0].color_code
            )
            result = [combined]
            part_index = 1
            rejection["part_index"] = part_index
        component = result[part_index - 1]
        source_ids = np.asarray(component.global_faces, dtype=np.int64)
        minimum_faces = max(3, int(round(len(source_ids) * 0.03)))
        source_regions = connected_face_regions(vertices, faces, source_ids)
        positive_assignment = np.zeros(len(faces), dtype=bool)
        split_region_count = 0
        retained_region_count = 0
        for region in source_regions:
            region_centers = np.asarray(vertices)[np.asarray(faces)[region]].mean(axis=1)
            signed_distance = (region_centers - origin) @ normal
            region_positive = signed_distance > 0.0
            if min(int(np.count_nonzero(region_positive)),
                   int(len(region) - np.count_nonzero(region_positive))) >= minimum_faces:
                positive_assignment[region] = region_positive
                split_region_count += 1
            else:
                positive_assignment[region] = bool(
                    np.count_nonzero(region_positive) > len(region) / 2
                    or (
                        np.count_nonzero(region_positive) == len(region) / 2
                        and float(signed_distance.mean()) > 0.0
                    )
                )
                retained_region_count += 1
        negative = source_ids[~positive_assignment[source_ids]]
        positive = source_ids[positive_assignment[source_ids]]
        if min(len(negative), len(positive)) < minimum_faces:
            records["rejected"].append({**rejection, "reason": "insufficient_partition_support"})
            continue
        output_region_counts = []
        for side in (negative, positive):
            local_faces = np.asarray(faces)[side]
            adjacency = trimesh.graph.face_adjacency(faces=local_faces)
            groups = trimesh.graph.connected_components(
                adjacency, nodes=np.arange(len(side)), min_len=1, engine="scipy"
            )
            output_region_counts.append(len(groups))
        local_faces = np.asarray(faces)[source_ids]
        adjacency, adjacency_edges = trimesh.graph.face_adjacency(
            faces=local_faces, return_edges=True
        )
        local_positive = positive_assignment[source_ids]
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
            "source_region_count": int(len(source_regions)),
            "partition_region_counts": [int(output_region_counts[0]), int(output_region_counts[1])],
            "plane_split_region_count": int(split_region_count),
            "whole_region_assignments": int(retained_region_count),
            "combined_source_component_count": int(combined_component_count),
            "source_material_preserved": True,
        })
    return result, records
