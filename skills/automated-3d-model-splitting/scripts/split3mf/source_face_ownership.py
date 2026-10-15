"""Keep source triangles while excluding tiny paint regions from part recognition."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .common import Component, DSU


def confirmed_part_ids(interface_plan: dict) -> set[str]:
    return {
        str(part["part"])
        for part in interface_plan.get("parts", [])
        if part.get("source_region_classification") == "part"
    }


def color_assigned_faces_as_part(
    source_tokens: np.ndarray,
    completed_face_ids: np.ndarray,
    recognized_face_ids: np.ndarray,
    part_color_code: str,
) -> tuple[np.ndarray, int]:
    """Recolor faces gained during ownership completion, preserving recognized paint."""
    tokens = np.asarray(source_tokens).astype(object, copy=True)
    completed_ids = np.asarray(completed_face_ids, dtype=np.int64)
    if len(tokens) != len(completed_ids):
        raise ValueError("source face colors do not align with completed ownership")
    assigned = ~np.isin(completed_ids, np.asarray(recognized_face_ids, dtype=np.int64))
    changed = int(np.count_nonzero(assigned & (tokens != part_color_code)))
    tokens[assigned] = part_color_code
    return tokens, changed


def complete_component_face_ownership(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    *,
    protected_part_ids: set[str] | None = None,
) -> tuple[list[Component], dict]:
    """Attach unrecognized edge-connected patches to a touching recognized part.

    Recognition identity remains unchanged. Only the source-face selections
    used to build output meshes are expanded; original face indices remain
    available so assigned faces can inherit their host part's output color.
    """
    faces = np.asarray(faces, dtype=np.int64)
    owner = np.full(len(faces), -1, dtype=np.int32)
    recognized_vertices: list[np.ndarray] = []
    for part_index, component in enumerate(components):
        selected = np.asarray(component.global_faces, dtype=np.int64)
        if np.any(owner[selected] >= 0):
            raise ValueError("recognized parts have overlapping source-face ownership")
        owner[selected] = part_index
        recognized_vertices.append(np.unique(faces[selected]))
    if not recognized_vertices:
        raise ValueError("source faces cannot be assigned without a recognized part")

    unassigned = np.flatnonzero(owner < 0)
    if not len(unassigned):
        return list(components), {"assigned_faces": 0, "unresolved_faces": 0, "patches": []}

    selected_faces = faces[unassigned]
    edges = np.sort(np.concatenate((
        selected_faces[:, [0, 1]], selected_faces[:, [1, 2]],
        selected_faces[:, [2, 0]],
    ), axis=0), axis=1)
    edge_face_positions = np.tile(np.arange(len(unassigned), dtype=np.int64), 3)
    keys = edges[:, 0] * int(len(vertices)) + edges[:, 1]
    order = np.argsort(keys, kind="mergesort")
    ordered_keys = keys[order]
    ordered_faces = edge_face_positions[order]
    groups = DSU(len(unassigned))
    for position in np.flatnonzero(np.diff(ordered_keys) == 0):
        groups.union(int(ordered_faces[position]), int(ordered_faces[position + 1]))

    patch_faces: dict[int, list[int]] = defaultdict(list)
    for position, face_id in enumerate(unassigned):
        patch_faces[groups.find(position)].append(int(face_id))

    assigned_by_part: dict[int, list[int]] = defaultdict(list)
    patch_records = []
    part_trees = [cKDTree(np.asarray(vertices)[ids]) for ids in recognized_vertices]
    protected_indices = {
        index for index in range(len(components))
        if f"P{index + 1:02d}" in (protected_part_ids or set())
    }
    for patch in patch_faces.values():
        patch_ids = np.asarray(patch, dtype=np.int64)
        patch_vertices = np.unique(faces[patch_ids])
        votes = []
        for part_vertices in recognized_vertices:
            locations = np.searchsorted(part_vertices, patch_vertices)
            in_range = locations < len(part_vertices)
            votes.append(int(np.count_nonzero(
                in_range & (part_vertices[np.minimum(locations, len(part_vertices) - 1)] == patch_vertices)
            )))
        available_hosts = [
            index for index, vote in enumerate(votes)
            if vote > 0 and index not in protected_indices
        ]
        eligible = available_hosts or list(range(len(components)))
        best_vote = max((votes[index] for index in eligible), default=0)
        winners = [index for index in eligible if votes[index] == best_vote]
        if best_vote and len(winners) == 1:
            part_index = winners[0]
            assignment_method = "unique_shared_vertices"
        else:
            candidates = winners if best_vote else range(len(components))
            sample = np.asarray(vertices)[patch_vertices]
            part_index = min(
                candidates,
                key=lambda index: (
                    float(np.min(part_trees[index].query(sample)[0])), index
                ),
            )
            assignment_method = "nearest_source_part_after_tie" if best_vote else "nearest_source_part"
        assigned_by_part[part_index].extend(patch)
        patch_records.append({
            "source_min_face_index": int(patch_ids.min()),
            "face_count": int(len(patch_ids)),
            "assigned_part": f"P{part_index + 1:02d}" if part_index is not None else None,
            "shared_source_vertex_count": best_vote,
            "assignment_method": assignment_method,
            "status": "assigned",
        })

    completed = []
    for part_index, component in enumerate(components):
        added = np.asarray(assigned_by_part.get(part_index, []), dtype=np.int64)
        if not len(added):
            completed.append(component)
            continue
        source_faces = np.unique(np.concatenate((component.global_faces, added)))
        triangles = np.asarray(vertices, dtype=np.float64)[faces[added]]
        added_area = np.linalg.norm(np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ), axis=1).sum() * 0.5
        completed.append(Component(
            color_code=component.color_code,
            global_faces=source_faces,
            face_count=int(len(source_faces)),
            area=float(component.area + added_area),
            bbox_min=np.minimum(component.bbox_min, triangles.min(axis=(0, 1))),
            bbox_max=np.maximum(component.bbox_max, triangles.max(axis=(0, 1))),
            center=component.center,
        ))
    assigned_faces = sum(len(values) for values in assigned_by_part.values())
    return completed, {
        "assigned_faces": assigned_faces,
        "unresolved_faces": int(len(unassigned) - assigned_faces),
        "patches": patch_records,
    }
