from __future__ import annotations

import collections
import heapq
from dataclasses import dataclass

import numpy as np

from .common import Component


@dataclass(frozen=True)
class InterfaceRetreatResult:
    components: list[Component]
    record: dict


def _component_from_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
    color_code: str,
) -> Component:
    face_ids = np.asarray(face_ids, dtype=np.int64)
    triangles = vertices[faces[face_ids]]
    points = triangles.reshape(-1, 3)
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    return Component(
        color_code=str(color_code),
        global_faces=face_ids,
        face_count=int(len(face_ids)),
        area=float(0.5 * np.linalg.norm(cross, axis=1).sum()),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        center=points.mean(axis=0),
    )


def _edge_owners(
    faces: np.ndarray,
    allowed_faces: set[int],
) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for face_id in sorted(allowed_faces):
        face = faces[int(face_id)]
        for left, right in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            result[tuple(sorted((int(left), int(right))))].append(int(face_id))
    return result


def _connected_face_regions(
    face_ids: set[int],
    edge_owners: dict[tuple[int, int], list[int]],
) -> list[set[int]]:
    if not face_ids:
        return []
    adjacency: dict[int, set[int]] = collections.defaultdict(set)
    for owners in edge_owners.values():
        active = [owner for owner in owners if owner in face_ids]
        for owner in active:
            adjacency[owner].update(other for other in active if other != owner)
    pending = set(face_ids)
    regions: list[set[int]] = []
    while pending:
        seed = pending.pop()
        region = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            neighbors = adjacency.get(current, set()) & pending
            pending.difference_update(neighbors)
            region.update(neighbors)
            stack.extend(neighbors)
        regions.append(region)
    return regions


def _connected_face_count(
    face_ids: set[int],
    edge_owners: dict[tuple[int, int], list[int]],
) -> int:
    return len(_connected_face_regions(face_ids, edge_owners))


def _shared_edges(
    child_faces: set[int],
    parent_faces: set[int],
    edge_owners: dict[tuple[int, int], list[int]],
) -> list[tuple[int, int]]:
    return [
        edge
        for edge, owners in edge_owners.items()
        if any(owner in child_faces for owner in owners)
        and any(owner in parent_faces for owner in owners)
    ]


def retreat_interface_ownership(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    *,
    child_index: int,
    parent_index: int,
    seed_point_mm: np.ndarray,
    seed_radius_mm: float,
    retreat_distance_mm: float,
    maximum_parent_face_fraction: float = 0.25,
) -> InterfaceRetreatResult:
    """Move a local parent surface patch into a child without changing paint.

    The old child/parent seam supplies eligible seed faces. A deterministic
    multi-source Dijkstra walk then grows only across the parent surface. This
    replaces a locally thin physical rim while leaving remote seam spans fixed.
    Material ownership intentionally lives outside ``Component`` and is not
    changed by this operation.
    """
    child_index = int(child_index)
    parent_index = int(parent_index)
    if child_index == parent_index:
        raise ValueError("interface retreat child and parent must differ")
    if child_index < 1 or parent_index < 1:
        raise ValueError("interface retreat part indices must be positive")
    if child_index > len(components) or parent_index > len(components):
        raise ValueError("interface retreat references an unknown part")
    seed_radius_mm = float(seed_radius_mm)
    retreat_distance_mm = float(retreat_distance_mm)
    maximum_parent_face_fraction = float(maximum_parent_face_fraction)
    if seed_radius_mm <= 0.0 or retreat_distance_mm <= 0.0:
        raise ValueError("interface retreat distances must be positive")
    if not 0.0 < maximum_parent_face_fraction < 1.0:
        raise ValueError("maximum parent face fraction must be between zero and one")
    seed_point = np.asarray(seed_point_mm, dtype=np.float64)
    if seed_point.shape != (3,) or not np.all(np.isfinite(seed_point)):
        raise ValueError("interface retreat seed point must contain three finite values")

    child = components[child_index - 1]
    parent = components[parent_index - 1]
    child_faces = {int(value) for value in child.global_faces}
    parent_faces = {int(value) for value in parent.global_faces}
    if child_faces & parent_faces:
        raise ValueError("interface retreat components overlap before transfer")

    allowed_faces = child_faces | parent_faces
    edge_owners = _edge_owners(faces, allowed_faces)
    old_shared_edges = _shared_edges(child_faces, parent_faces, edge_owners)
    if len(old_shared_edges) < 3:
        raise ValueError("interface retreat requires a shared child/parent boundary")

    parent_boundary_faces = {
        owner
        for edge in old_shared_edges
        for owner in edge_owners[edge]
        if owner in parent_faces
    }
    centroids = vertices[faces].mean(axis=1)
    seed_faces = sorted(
        face_id
        for face_id in parent_boundary_faces
        if float(np.linalg.norm(centroids[face_id] - seed_point))
        <= seed_radius_mm
    )
    if not seed_faces:
        nearest_distance = min(
            float(np.linalg.norm(centroids[face_id] - seed_point))
            for face_id in parent_boundary_faces
        )
        raise ValueError(
            "interface retreat seed does not reach the shared boundary; "
            f"nearest parent face is {nearest_distance:.6f} mm away"
        )

    parent_adjacency: dict[int, set[int]] = collections.defaultdict(set)
    for owners in edge_owners.values():
        active = [owner for owner in owners if owner in parent_faces]
        for owner in active:
            parent_adjacency[owner].update(
                other for other in active if other != owner
            )

    distances = {face_id: 0.0 for face_id in seed_faces}
    queue = [(0.0, face_id) for face_id in seed_faces]
    heapq.heapify(queue)
    while queue:
        distance, face_id = heapq.heappop(queue)
        if distance > distances[face_id] + 1e-12:
            continue
        if distance > retreat_distance_mm:
            continue
        for neighbor in parent_adjacency.get(face_id, set()):
            step = float(np.linalg.norm(centroids[neighbor] - centroids[face_id]))
            candidate = distance + step
            if candidate > retreat_distance_mm + 1e-12:
                continue
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))

    transferred = {
        face_id
        for face_id, distance in distances.items()
        if distance <= retreat_distance_mm + 1e-12
    }
    maximum_transfer_faces = int(
        np.floor(len(parent_faces) * maximum_parent_face_fraction)
    )
    if not transferred:
        raise ValueError("interface retreat selected no parent faces")
    if len(transferred) > maximum_transfer_faces:
        raise ValueError(
            "interface retreat exceeded its parent-face safety cap: "
            f"{len(transferred)} > {maximum_transfer_faces}"
        )

    old_parent_regions = _connected_face_regions(parent_faces, edge_owners)
    old_parent_region_by_face = {
        face_id: region_index
        for region_index, region in enumerate(old_parent_regions)
        for face_id in region
    }
    new_child_faces = child_faces | transferred
    new_parent_faces = parent_faces - transferred
    if not new_parent_faces:
        raise ValueError("interface retreat consumed the entire parent")
    absorbed_isolated_faces: set[int] = set()
    remaining_by_old_region: dict[int, list[set[int]]] = collections.defaultdict(list)
    for region in _connected_face_regions(new_parent_faces, edge_owners):
        old_region_index = old_parent_region_by_face[next(iter(region))]
        remaining_by_old_region[old_region_index].append(region)
    for regions in remaining_by_old_region.values():
        if len(regions) <= 1:
            continue
        keep = max(regions, key=lambda region: (len(region), -min(region)))
        for region in regions:
            if region is not keep:
                absorbed_isolated_faces.update(region)
    if absorbed_isolated_faces:
        transferred.update(absorbed_isolated_faces)
        new_child_faces.update(absorbed_isolated_faces)
        new_parent_faces.difference_update(absorbed_isolated_faces)
    if len(transferred) > maximum_transfer_faces:
        raise ValueError(
            "interface retreat exceeded its parent-face safety cap after "
            f"topology closure: {len(transferred)} > {maximum_transfer_faces}"
        )
    old_child_region_count = _connected_face_count(child_faces, edge_owners)
    old_parent_region_count = _connected_face_count(parent_faces, edge_owners)
    new_child_region_count = _connected_face_count(new_child_faces, edge_owners)
    new_parent_region_count = _connected_face_count(new_parent_faces, edge_owners)
    if new_child_region_count > old_child_region_count:
        raise ValueError("interface retreat would add a disconnected child region")
    if new_parent_region_count > old_parent_region_count:
        raise ValueError("interface retreat would add a disconnected parent region")
    new_shared_edges = _shared_edges(
        new_child_faces,
        new_parent_faces,
        edge_owners,
    )
    if len(new_shared_edges) < 3:
        raise ValueError("interface retreat removed the printable shared boundary")

    updated = list(components)
    updated[child_index - 1] = _component_from_faces(
        vertices,
        faces,
        np.asarray(sorted(new_child_faces), dtype=np.int64),
        child.color_code,
    )
    updated[parent_index - 1] = _component_from_faces(
        vertices,
        faces,
        np.asarray(sorted(new_parent_faces), dtype=np.int64),
        parent.color_code,
    )
    transferred_ids = np.asarray(sorted(transferred), dtype=np.int64)
    transferred_points = vertices[faces[transferred_ids]].reshape(-1, 3)
    return InterfaceRetreatResult(
        components=updated,
        record={
            "mode": "local_geodesic_surface_ownership_retreat",
            "child_index": child_index,
            "parent_index": parent_index,
            "seed_point_mm": seed_point.tolist(),
            "seed_radius_mm": seed_radius_mm,
            "retreat_distance_mm": retreat_distance_mm,
            "seed_face_count": int(len(seed_faces)),
            "transferred_face_count": int(len(transferred)),
            "topology_closure_absorbed_face_count": int(
                len(absorbed_isolated_faces)
            ),
            "transferred_parent_face_fraction": float(
                len(transferred) / max(len(parent_faces), 1)
            ),
            "maximum_parent_face_fraction": maximum_parent_face_fraction,
            "old_shared_edge_count": int(len(old_shared_edges)),
            "new_shared_edge_count": int(len(new_shared_edges)),
            "transferred_bbox_min_mm": transferred_points.min(axis=0).tolist(),
            "transferred_bbox_max_mm": transferred_points.max(axis=0).tolist(),
            "source_materials_preserved": True,
            "old_child_connected_region_count": int(old_child_region_count),
            "new_child_connected_region_count": int(new_child_region_count),
            "old_parent_connected_region_count": int(old_parent_region_count),
            "new_parent_connected_region_count": int(new_parent_region_count),
        },
    )


def apply_visual_interface_retreats(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    directives: list[dict],
    minimum_confidence: float,
) -> tuple[list[Component], dict]:
    current = list(components)
    applied: list[dict] = []
    rejected: list[dict] = []
    for directive in directives:
        summary = {
            "child_index": int(directive.get("child_index", 0)),
            "parent_index": int(directive.get("parent_index", 0)),
            "confidence": directive.get("confidence", "UNKNOWN"),
            "confidence_score": float(directive.get("confidence_score", 0.0)),
        }
        if not bool(directive.get("apply", True)):
            rejected.append({**summary, "reject_reason": "directive_disabled"})
            continue
        if summary["confidence_score"] < float(minimum_confidence):
            rejected.append(
                {**summary, "reject_reason": "semantic_confidence_below_threshold"}
            )
            continue
        try:
            result = retreat_interface_ownership(
                vertices,
                faces,
                current,
                child_index=summary["child_index"],
                parent_index=summary["parent_index"],
                seed_point_mm=directive.get("seed_point_mm"),
                seed_radius_mm=directive.get("seed_radius_mm", 2.0),
                retreat_distance_mm=directive.get("retreat_distance_mm"),
                maximum_parent_face_fraction=directive.get(
                    "maximum_parent_face_fraction",
                    0.25,
                ),
            )
        except (TypeError, ValueError) as error:
            rejected.append(
                {**summary, "reject_reason": "topology_or_geometry_gate", "error": str(error)}
            )
            continue
        current = result.components
        applied.append({**summary, **result.record})
    return current, {"applied": applied, "rejected": rejected}
