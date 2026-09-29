"""Shared mesh-topology audit and acceptance policy."""

from __future__ import annotations

import numpy as np
import trimesh

from .mesh import watertight_component_orientation_audit

TOPOLOGY_DEFECT_RATIO_LIMIT = 0.01


def audit_mesh_topology(mesh: trimesh.Trimesh) -> dict:
    """Count boundary and over-shared edges without welding coincident vertices."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    directed_edges = (
        np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        if len(faces)
        else np.empty((0, 2), dtype=np.int64)
    )
    if len(directed_edges):
        edge_min = np.minimum(directed_edges[:, 0], directed_edges[:, 1])
        edge_max = np.maximum(directed_edges[:, 0], directed_edges[:, 1])
        maximum_vertex_id = int(max(edge_min.max(), edge_max.max()))
        if maximum_vertex_id < (1 << 32):
            edge_keys = (
                edge_min.astype(np.uint64) << np.uint64(32)
            ) | edge_max.astype(np.uint64)
            unique_keys, inverse_edges, counts = np.unique(
                edge_keys,
                return_inverse=True,
                return_counts=True,
            )
            unique_edges = np.column_stack((
                (unique_keys >> np.uint64(32)).astype(np.int64),
                (unique_keys & np.uint64(0xFFFFFFFF)).astype(np.int64),
            ))
        else:
            unique_edges, inverse_edges, counts = np.unique(
                np.column_stack((edge_min, edge_max)),
                axis=0,
                return_inverse=True,
                return_counts=True,
            )
        direction_sign = np.where(directed_edges[:, 0] == edge_min, 1, -1)
    else:
        unique_edges = np.empty((0, 2), dtype=np.int64)
        inverse_edges = np.array([], dtype=np.int64)
        counts = np.array([], dtype=np.int64)
        direction_sign = np.array([], dtype=np.int8)

    open_edge_ids = unique_edges[counts == 1]
    over_edge_ids = unique_edges[counts > 2]
    direction_balance = (
        np.bincount(inverse_edges, weights=direction_sign, minlength=len(unique_edges))
        if len(inverse_edges)
        else np.array([], dtype=np.float64)
    )
    inconsistent_edge_ids = unique_edges[
        (counts == 2) & (np.abs(direction_balance) == 2)
    ]
    unique_edge_count = int(len(unique_edges))
    defect_edge_count = int(len(open_edge_ids) + len(over_edge_ids))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    def edge_metrics(selected: np.ndarray) -> dict:
        if not len(selected):
            return {
                "count": 0,
                "total_length_mm": 0.0,
                "max_length_mm": 0.0,
                "bbox_min": None,
                "bbox_max": None,
            }
        segments = vertices[selected]
        lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
        points = segments.reshape(-1, 3)
        return {
            "count": int(len(selected)),
            "total_length_mm": float(lengths.sum()),
            "max_length_mm": float(lengths.max()),
            "bbox_min": points.min(axis=0).round(6).tolist(),
            "bbox_max": points.max(axis=0).round(6).tolist(),
        }

    component_orientation = watertight_component_orientation_audit(mesh)
    return {
        "watertight": bool(
            len(faces) and not len(open_edge_ids) and not len(over_edge_ids)
        ),
        "winding_consistent": not bool(len(inconsistent_edge_ids)),
        "open_edges": int(len(open_edge_ids)),
        "over_shared_edges": int(len(over_edge_ids)),
        "inconsistent_shared_edges": int(len(inconsistent_edge_ids)),
        "unique_edges": unique_edge_count,
        "defect_edges": defect_edge_count,
        "topology_defect_ratio": float(
            defect_edge_count / max(unique_edge_count, 1)
        ),
        "open_edge_ratio": float(len(open_edge_ids) / max(unique_edge_count, 1)),
        "over_shared_edge_ratio": float(len(over_edge_ids) / max(unique_edge_count, 1)),
        "inconsistent_orientation_ratio": float(
            len(inconsistent_edge_ids) / max(unique_edge_count, 1)
        ),
        "open_edge_metrics": edge_metrics(open_edge_ids),
        "over_shared_edge_metrics": edge_metrics(over_edge_ids),
        "inconsistent_edge_metrics": edge_metrics(inconsistent_edge_ids),
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "closed_component_orientation": component_orientation,
        "connected_component_count": component_orientation.get("component_count"),
        "inward_closed_components": component_orientation.get(
            "inward_closed_component_count"
        ),
        "all_closed_components_outward": component_orientation.get(
            "all_closed_components_outward"
        ),
    }


def topology_defects_within_tolerance(audit: dict) -> bool:
    """Apply the single strict-less-than-1% topology tolerance."""
    return (
        int(audit.get("unique_edges", 0)) > 0
        and float(audit.get("topology_defect_ratio", 1.0))
        < TOPOLOGY_DEFECT_RATIO_LIMIT
    )
