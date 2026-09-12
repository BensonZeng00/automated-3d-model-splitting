"""Reusable triangle-shape and local winding audits."""

from __future__ import annotations

import numpy as np


SOURCE_NORMAL_MIN_SHAPE_QUALITY = 0.08


def sparse_local_inversion_audit(
    faces: np.ndarray,
    inverted_mask: np.ndarray,
    *,
    audited_face_count: int,
    result_minimum_angles_degrees: np.ndarray,
    maximum_ratio: float = 0.0005,
    maximum_edge_connected_cluster: int = 2,
    minimum_audited_faces: int = 1000,
    minimum_result_angle_degrees: float = 3.0,
) -> dict:
    """Allow only sparse, isolated normal-reference outliers on large bands.

    A face can reverse relative to its *source* normal while the result keeps
    consistent indexed winding, non-zero area, and a visually smooth surface.
    This classifier never accepts a connected folded patch and is deliberately
    unavailable to small synthetic meshes where one bad face would dominate.
    """

    mesh_faces = np.asarray(faces, dtype=np.int64)
    mask = np.asarray(inverted_mask, dtype=bool)
    if mask.shape != (len(mesh_faces),):
        raise ValueError("inverted face mask must match the indexed face array")
    inverted_ids = np.flatnonzero(mask)
    count = int(len(inverted_ids))
    population = max(int(audited_face_count), 0)
    allowed_count = int(np.floor(population * float(maximum_ratio) + 1e-12))

    selected_by_edge: dict[tuple[int, int], list[int]] = {}
    for face_id in inverted_ids:
        for edge in face_edge_keys(mesh_faces[int(face_id)]):
            selected_by_edge.setdefault(edge, []).append(int(face_id))
    adjacency = {int(face_id): set() for face_id in inverted_ids}
    for owners in selected_by_edge.values():
        if len(owners) < 2:
            continue
        for owner in owners:
            adjacency[owner].update(other for other in owners if other != owner)
    maximum_cluster = 0
    pending = set(adjacency)
    while pending:
        seed = min(pending)
        stack = [seed]
        pending.remove(seed)
        cluster_count = 0
        while stack:
            current = stack.pop()
            cluster_count += 1
            for neighbor in sorted(adjacency[current]):
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
        maximum_cluster = max(maximum_cluster, cluster_count)

    angles = np.asarray(result_minimum_angles_degrees, dtype=np.float64)
    minimum_angle = float(angles.min()) if len(angles) else None
    accepted = bool(
        count > 0
        and population >= int(minimum_audited_faces)
        and count <= allowed_count
        and maximum_cluster <= int(maximum_edge_connected_cluster)
        and minimum_angle is not None
        and minimum_angle >= float(minimum_result_angle_degrees) - 1e-9
    )
    return {
        "accepted": accepted,
        "count": count,
        "audited_face_count": population,
        "ratio": float(count / max(population, 1)),
        "maximum_ratio": float(maximum_ratio),
        "maximum_allowed_count": allowed_count,
        "maximum_edge_connected_cluster": int(maximum_cluster),
        "maximum_allowed_edge_connected_cluster": int(
            maximum_edge_connected_cluster
        ),
        "minimum_result_angle_degrees": minimum_angle,
        "required_minimum_result_angle_degrees": float(
            minimum_result_angle_degrees
        ),
        "policy": "large_band_sparse_isolated_source_normal_outliers",
    }


def face_edge_keys(face: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return the three undirected edge keys for one triangle."""

    return tuple(
        tuple(sorted((int(left), int(right))))
        for left, right in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        )
    )


def build_edge_face_map(
    faces: np.ndarray,
) -> dict[tuple[int, int], list[int]]:
    """Build deterministic edge ownership once for an indexed triangle mesh."""

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(np.asarray(faces, dtype=np.int64)):
        for edge in face_edge_keys(face):
            edge_faces.setdefault(edge, []).append(int(face_id))
    return edge_faces


def triangle_shape_quality(triangles: np.ndarray) -> np.ndarray:
    """Return a dimensionless 0..1 triangle quality independent of scale."""

    selected = np.asarray(triangles, dtype=np.float64)
    edge_squared = np.stack(
        (
            np.sum((selected[:, 1] - selected[:, 0]) ** 2, axis=1),
            np.sum((selected[:, 2] - selected[:, 1]) ** 2, axis=1),
            np.sum((selected[:, 0] - selected[:, 2]) ** 2, axis=1),
        ),
        axis=1,
    )
    double_area = np.linalg.norm(
        np.cross(
            selected[:, 1] - selected[:, 0],
            selected[:, 2] - selected[:, 0],
        ),
        axis=1,
    )
    return (
        2.0
        * np.sqrt(3.0)
        * double_area
        / np.maximum(edge_squared.sum(axis=1), 1e-30)
    )


def triangle_minimum_angles_degrees(triangles: np.ndarray) -> np.ndarray:
    """Return the minimum interior angle of each nondegenerate triangle."""

    selected = np.asarray(triangles, dtype=np.float64)
    minimum_angles = np.zeros(len(selected), dtype=np.float64)
    for face_id, triangle in enumerate(selected):
        angles: list[float] = []
        for vertex_id in range(3):
            left = triangle[(vertex_id + 1) % 3] - triangle[vertex_id]
            right = triangle[(vertex_id - 1) % 3] - triangle[vertex_id]
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator <= 1e-30:
                angles.append(0.0)
                continue
            cosine = float(
                np.clip(np.dot(left, right) / denominator, -1.0, 1.0)
            )
            angles.append(float(np.degrees(np.arccos(cosine))))
        minimum_angles[int(face_id)] = min(angles, default=0.0)
    return minimum_angles


def directed_edge_topology_issues(
    faces: np.ndarray,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Return over-shared and same-direction shared edge keys."""

    mesh_faces = np.asarray(faces, dtype=np.int64)
    if not len(mesh_faces):
        return set(), set()
    directed_edges = np.vstack(
        (
            mesh_faces[:, [0, 1]],
            mesh_faces[:, [1, 2]],
            mesh_faces[:, [2, 0]],
        )
    )
    edge_min = np.minimum(directed_edges[:, 0], directed_edges[:, 1])
    edge_max = np.maximum(directed_edges[:, 0], directed_edges[:, 1])
    unique_edges, inverse, counts = np.unique(
        np.column_stack((edge_min, edge_max)),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    direction_balance = np.bincount(
        inverse,
        weights=np.where(directed_edges[:, 0] == edge_min, 1, -1),
        minlength=len(unique_edges),
    )
    over_shared = unique_edges[counts > 2]
    inconsistent = unique_edges[
        (counts == 2) & (np.abs(direction_balance) == 2)
    ]
    return (
        {tuple(int(value) for value in edge) for edge in over_shared},
        {tuple(int(value) for value in edge) for edge in inconsistent},
    )


def _directed_edge_sign(face: np.ndarray, edge: tuple[int, int]) -> int:
    for left, right in (
        (face[0], face[1]),
        (face[1], face[2]),
        (face[2], face[0]),
    ):
        if tuple(sorted((int(left), int(right)))) == edge:
            return 1 if int(left) < int(right) else -1
    raise ValueError("face does not contain requested edge")


def replacement_preserves_winding(
    faces: np.ndarray,
    edge_faces: dict[tuple[int, int], list[int]],
    face_ids: list[int] | tuple[int, ...] | np.ndarray,
    proposed_faces: np.ndarray,
) -> bool:
    """Require a local retriangulation to preserve its rim and edge winding."""

    current = np.asarray(faces, dtype=np.int64)
    selected_ids = np.asarray(face_ids, dtype=np.int64)
    proposed = np.asarray(proposed_faces, dtype=np.int64)
    if proposed.shape != (len(selected_ids), 3):
        return False
    patch_ids = set(int(value) for value in selected_ids)

    def boundary_edges(selected_faces: np.ndarray) -> set[tuple[int, int]]:
        counts: dict[tuple[int, int], int] = {}
        for selected_face in selected_faces:
            for edge in face_edge_keys(selected_face):
                counts[edge] = counts.get(edge, 0) + 1
        return {edge for edge, count in counts.items() if count == 1}

    if boundary_edges(current[selected_ids]) != boundary_edges(proposed):
        return False

    proposed_signs: dict[tuple[int, int], list[int]] = {}
    for proposed_face in proposed:
        for edge in face_edge_keys(proposed_face):
            proposed_signs.setdefault(edge, []).append(
                _directed_edge_sign(proposed_face, edge)
            )
    for edge, signs in proposed_signs.items():
        combined = list(signs)
        combined.extend(
            _directed_edge_sign(current[int(owner)], edge)
            for owner in edge_faces.get(edge, ())
            if int(owner) not in patch_ids
        )
        if len(combined) > 2:
            return False
        if len(combined) == 2 and sum(combined) != 0:
            return False
    return True
