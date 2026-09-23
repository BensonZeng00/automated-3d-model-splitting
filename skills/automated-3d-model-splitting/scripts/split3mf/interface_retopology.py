"""Production boundary retopology facade used by every split interface."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from . import common
from .domain import PlanarArcRetopologyContext, SeamSmoothingPolicy
from .mesh import (
    boundary_loops,
    triangulate_ordered_loop_3d,
)
from .planar_arc import PlanarArcError, CurveClarityRequired, build_planar_arc_boundary
from .reporting import runtime_log
from .surface_quality import (
    SOURCE_NORMAL_MIN_SHAPE_QUALITY as _SOURCE_NORMAL_MIN_SHAPE_QUALITY,
    build_edge_face_map as _build_edge_face_map,
    directed_edge_topology_issues as _directed_edge_topology_issues,
    face_edge_keys as _face_edge_keys,
    replacement_preserves_winding as _replacement_preserves_winding,
    sparse_local_inversion_audit as _sparse_local_inversion_audit,
    triangle_minimum_angles_degrees as _triangle_minimum_angles_degrees,
    triangle_shape_quality as _triangle_shape_quality,
)


def _visual_displacement_advisory(
    validation_mode: str,
    displacement: np.ndarray,
    policy: SeamSmoothingPolicy,
) -> tuple[bool, float, float]:
    """Classify broad coverage by physical displacement, never topology."""
    values = np.asarray(displacement, dtype=np.float64)
    maximum = float(values.max(initial=0.0))
    p95 = float(np.percentile(values, 95)) if len(values) else 0.0
    accepted = bool(
        str(validation_mode) == "advisory"
        and maximum <= policy.maximum_displacement_mm + 1e-12
        and p95 <= policy.p95_displacement_mm + 1e-12
    )
    return accepted, maximum, p95


def _select_visible_boundary_target(
    source: np.ndarray,
    proposed: np.ndarray,
    visible_band_mm: float,
    maximum_visible_offset_mm: float,
) -> tuple[np.ndarray, dict]:
    """Choose the visible rim without discarding the manufacturing target.

    Motion larger than the safe visible fraction of the transition band must
    not be applied to the source surface.  The fitted target is nevertheless
    retained in the record so generated inward walls and caps can use it as
    their planning boundary.
    """
    source_points = np.asarray(source, dtype=np.float64)
    proposed_points = np.asarray(proposed, dtype=np.float64)
    if proposed_points.shape != source_points.shape:
        raise PlanarArcError("fitted boundary target does not match source rim")
    displacement = np.linalg.norm(proposed_points - source_points, axis=1)
    maximum = float(displacement.max(initial=0.0))
    safe_offset = float(maximum_visible_offset_mm)
    if safe_offset <= 0.0 or safe_offset > float(visible_band_mm) + 1e-12:
        raise ValueError("maximum visible offset must be inside the transition band")
    preserve_source = bool(maximum > safe_offset + 1e-12)
    return (
        source_points.copy() if preserve_source else proposed_points.copy(),
        {
            "large_displacement_source_boundary_preserved": preserve_source,
            "requested_maximum_target_displacement_mm": maximum,
            "visible_transition_band_mm": float(visible_band_mm),
            "maximum_visible_boundary_offset_mm": safe_offset,
            "large_displacement_strategy": (
                "generated_inward_wall_from_immutable_source_ring"
                if preserve_source
                else "visible_source_band_deformation"
            ),
            # Keep this JSON-compatible because retopology records are also
            # emitted as diagnostics.  Consumers convert it back to float64.
            "generated_inward_boundary_points": proposed_points.tolist(),
        },
    )


def generated_geometry_boundary_vertices(
    visible_vertices: np.ndarray,
    loops: list[list[int]],
    records: list[dict],
) -> np.ndarray:
    """Overlay retained fitted rings for hidden-geometry planning only."""
    planned = np.asarray(visible_vertices, dtype=np.float64).copy()
    if len(loops) != len(records):
        raise PlanarArcError("retopology records do not match boundary loops")
    for loop, record in zip(loops, records):
        if not record.get("large_displacement_source_boundary_preserved", False):
            continue
        loop_ids = np.asarray(loop, dtype=np.int64)
        target = np.asarray(
            record.get("generated_inward_boundary_points"), dtype=np.float64
        )
        if target.shape != (len(loop_ids), 3):
            raise PlanarArcError("generated inward target does not match boundary loop")
        planned[loop_ids] = target
    return planned


def _replace_faces_in_edge_map(
    faces: np.ndarray,
    face_ids: list[int] | tuple[int, ...] | np.ndarray,
    proposed_faces: np.ndarray,
    edge_faces: dict[tuple[int, int], list[int]],
) -> None:
    """Replace a local face patch while incrementally updating edge ownership."""

    ordered_ids = np.asarray(face_ids, dtype=np.int64)
    proposed = np.asarray(proposed_faces, dtype=np.int64)
    if proposed.shape != (len(ordered_ids), 3):
        raise ValueError("replacement face patch does not match face ids")

    for face_id in ordered_ids:
        for edge in _face_edge_keys(faces[int(face_id)]):
            owners = edge_faces[edge]
            owners.remove(int(face_id))
            if not owners:
                del edge_faces[edge]

    faces[ordered_ids] = proposed

    for face_id in ordered_ids:
        for edge in _face_edge_keys(faces[int(face_id)]):
            owners = edge_faces.setdefault(edge, [])
            owners.append(int(face_id))
            owners.sort()


def _locally_inverted_face_mask(
    source_points: np.ndarray,
    result_points: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Return reliable folds opposed to their current one-ring.

    A nearly collinear source triangle has an unstable normal: a harmless
    displacement or a new local diagonal can rotate that normal arbitrarily.
    Such faces remain subject to final degeneracy, stretch, topology, and
    intersection audits, but their source-face normal cannot veto the band.
    """

    mesh_faces = np.asarray(faces, dtype=np.int64)
    source_triangles = np.asarray(source_points, dtype=np.float64)[mesh_faces]
    result_triangles = np.asarray(result_points, dtype=np.float64)[mesh_faces]
    source_normals = np.cross(
        source_triangles[:, 1] - source_triangles[:, 0],
        source_triangles[:, 2] - source_triangles[:, 0],
    )
    result_normals = np.cross(
        result_triangles[:, 1] - result_triangles[:, 0],
        result_triangles[:, 2] - result_triangles[:, 0],
    )
    source_lengths = np.linalg.norm(source_normals, axis=1)
    result_lengths = np.linalg.norm(result_normals, axis=1)
    source_quality = _triangle_shape_quality(source_triangles)
    comparable = (source_lengths > 1e-15) & (result_lengths > 1e-12)
    source_result_cosine = np.ones(len(mesh_faces), dtype=np.float64)
    source_result_cosine[comparable] = np.einsum(
        "ij,ij->i",
        source_normals[comparable],
        result_normals[comparable],
    ) / (source_lengths[comparable] * result_lengths[comparable])

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(mesh_faces):
        for left, right in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge_faces.setdefault(
                tuple(sorted((int(left), int(right)))), []
            ).append(int(face_id))

    inverted = np.zeros(len(mesh_faces), dtype=bool)
    reliable_sign_change = (
        (source_result_cosine < -1e-8)
        & (source_quality >= _SOURCE_NORMAL_MIN_SHAPE_QUALITY)
    )
    for face_id in np.flatnonzero(reliable_sign_change):
        face = mesh_faces[int(face_id)]
        neighbor_ids = {
            neighbor
            for left, right in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            )
            for neighbor in edge_faces.get(
                tuple(sorted((int(left), int(right)))), ()
            )
            if int(neighbor) != int(face_id)
        }
        valid_neighbors = 0
        opposed = 0
        for neighbor_id in neighbor_ids:
            denominator = (
                result_lengths[int(face_id)] * result_lengths[int(neighbor_id)]
            )
            if denominator <= 1e-24:
                continue
            valid_neighbors += 1
            cosine = float(
                np.dot(
                    result_normals[int(face_id)],
                    result_normals[int(neighbor_id)],
                )
                / denominator
            )
            if cosine < -0.05:
                opposed += 1
        inverted[int(face_id)] = bool(
            valid_neighbors == 0 or opposed / valid_neighbors >= 2.0 / 3.0
        )
    return inverted


def _untangle_interior_surface_vertices(
    source_points: np.ndarray,
    result_points: np.ndarray,
    faces: np.ndarray,
    boundary_ids: np.ndarray,
    *,
    maximum_iterations: int = 32,
) -> tuple[np.ndarray, int, float]:
    """Remove residual folds by minimally retracting only interior vertices.

    The smoothed manufacturing seam is Dirichlet-pinned.  A candidate is
    accepted only when it strictly reduces the audited fold count, keeps every
    incident triangle non-degenerate, and remains inside the normal edge-
    stretch limit.  This repairs the displacement field instead of relaxing a
    topology or quality threshold.
    """

    source = np.asarray(source_points, dtype=np.float64)
    mesh_faces = np.asarray(faces, dtype=np.int64)
    result = np.asarray(result_points, dtype=np.float64).copy()
    initial_result = result.copy()
    boundary_set = set(int(value) for value in np.asarray(boundary_ids, dtype=np.int64))

    source_triangles = source[mesh_faces]
    source_normals = np.cross(
        source_triangles[:, 1] - source_triangles[:, 0],
        source_triangles[:, 2] - source_triangles[:, 0],
    )
    source_normal_lengths = np.linalg.norm(source_normals, axis=1)
    source_shape_quality = _triangle_shape_quality(source_triangles)

    vertex_faces: dict[int, list[int]] = {}
    adjacency: dict[int, set[int]] = {}
    for face_id, face in enumerate(mesh_faces):
        for vertex_id in face:
            vertex_faces.setdefault(int(vertex_id), []).append(int(face_id))
        for left, right in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            left_id = int(left)
            right_id = int(right)
            adjacency.setdefault(left_id, set()).add(right_id)
            adjacency.setdefault(right_id, set()).add(left_id)

    edge_faces = _build_edge_face_map(mesh_faces)
    face_neighbors: list[set[int]] = [set() for _ in range(len(mesh_faces))]
    for owners in edge_faces.values():
        if len(owners) < 2:
            continue
        for face_id in owners:
            face_neighbors[int(face_id)].update(
                int(neighbor_id)
                for neighbor_id in owners
                if int(neighbor_id) != int(face_id)
            )

    def affected_faces(changed_vertices: tuple[int, ...]) -> np.ndarray:
        incident = {
            int(face_id)
            for vertex_id in changed_vertices
            for face_id in vertex_faces.get(int(vertex_id), ())
        }
        affected = set(incident)
        for face_id in incident:
            affected.update(face_neighbors[int(face_id)])
        return np.asarray(sorted(affected), dtype=np.int64)

    def triangles_with_positions(
        face_ids: np.ndarray,
        positions: dict[int, np.ndarray],
    ) -> np.ndarray:
        selected_faces = mesh_faces[np.asarray(face_ids, dtype=np.int64)]
        triangles = result[selected_faces].copy()
        for vertex_id, position in positions.items():
            triangles[selected_faces == int(vertex_id)] = np.asarray(
                position, dtype=np.float64
            )
        return triangles

    def inverted_for_faces(
        face_ids: np.ndarray,
        positions: dict[int, np.ndarray],
    ) -> np.ndarray:
        requested = np.asarray(face_ids, dtype=np.int64)
        needed = set(int(face_id) for face_id in requested)
        for face_id in requested:
            needed.update(face_neighbors[int(face_id)])
        needed_ids = np.asarray(sorted(needed), dtype=np.int64)
        triangles = triangles_with_positions(needed_ids, positions)
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        lengths = np.linalg.norm(normals, axis=1)
        local_index = {
            int(face_id): index for index, face_id in enumerate(needed_ids)
        }
        inverted = np.zeros(len(requested), dtype=bool)
        for output_index, face_id_value in enumerate(requested):
            face_id = int(face_id_value)
            normal_index = local_index[face_id]
            source_length = float(source_normal_lengths[face_id])
            result_length = float(lengths[normal_index])
            if (
                source_length <= 1e-15
                or result_length <= 1e-12
                or float(source_shape_quality[face_id])
                < _SOURCE_NORMAL_MIN_SHAPE_QUALITY
            ):
                continue
            source_result_cosine = float(
                np.dot(source_normals[face_id], normals[normal_index])
                / (source_length * result_length)
            )
            if source_result_cosine >= -1e-8:
                continue
            valid_neighbors = 0
            opposed = 0
            for neighbor_id in face_neighbors[face_id]:
                neighbor_index = local_index[int(neighbor_id)]
                denominator = result_length * float(lengths[neighbor_index])
                if denominator <= 1e-24:
                    continue
                valid_neighbors += 1
                cosine = float(
                    np.dot(normals[normal_index], normals[neighbor_index])
                    / denominator
                )
                if cosine < -0.05:
                    opposed += 1
            inverted[output_index] = bool(
                valid_neighbors == 0
                or opposed / valid_neighbors >= 2.0 / 3.0
            )
        return inverted

    def candidate_is_geometrically_safe(
        changed_vertices: tuple[int, ...],
        positions: dict[int, np.ndarray],
    ) -> bool:
        incident_ids = sorted(
            {
                face_id
                for vertex_id in changed_vertices
                for face_id in vertex_faces.get(int(vertex_id), ())
            }
        )
        if not incident_ids:
            return False
        incident = mesh_faces[np.asarray(incident_ids, dtype=np.int64)]
        source_triangles = source[incident]
        candidate_triangles = triangles_with_positions(
            np.asarray(incident_ids, dtype=np.int64), positions
        )
        normals = np.cross(
            candidate_triangles[:, 1] - candidate_triangles[:, 0],
            candidate_triangles[:, 2] - candidate_triangles[:, 0],
        )
        if np.any(np.linalg.norm(normals, axis=1) <= 1e-12):
            return False
        source_edges = np.stack(
            (
                np.linalg.norm(source_triangles[:, 1] - source_triangles[:, 0], axis=1),
                np.linalg.norm(source_triangles[:, 2] - source_triangles[:, 1], axis=1),
                np.linalg.norm(source_triangles[:, 0] - source_triangles[:, 2], axis=1),
            ),
            axis=1,
        )
        candidate_edges = np.stack(
            (
                np.linalg.norm(candidate_triangles[:, 1] - candidate_triangles[:, 0], axis=1),
                np.linalg.norm(candidate_triangles[:, 2] - candidate_triangles[:, 1], axis=1),
                np.linalg.norm(candidate_triangles[:, 0] - candidate_triangles[:, 2], axis=1),
            ),
            axis=1,
        )
        return bool(
            np.max(candidate_edges / np.maximum(source_edges, 0.02)) <= 8.0 + 1e-9
        )

    def targets_for(vertex_id: int, current_result: np.ndarray) -> tuple[np.ndarray, ...]:
        neighbors = sorted(adjacency.get(int(vertex_id), ()))
        if not neighbors:
            return (source[int(vertex_id)],)
        neighbor_ids = np.asarray(neighbors, dtype=np.int64)
        neighbor_displacement = np.mean(
            current_result[neighbor_ids] - source[neighbor_ids], axis=0
        )
        return (
            source[int(vertex_id)] + neighbor_displacement,
            source[int(vertex_id)],
        )

    repair_count = 0
    fractions = (0.125, 0.25, 0.5, 0.75, 1.0)
    inverted = _locally_inverted_face_mask(source, result, mesh_faces)
    for _ in range(maximum_iterations):
        current_count = int(np.count_nonzero(inverted))
        if current_count == 0:
            break

        best: tuple[
            int,
            float,
            dict[int, np.ndarray],
            np.ndarray,
            np.ndarray,
        ] | None = None
        suspect_ids = np.flatnonzero(inverted)
        movable_vertices = sorted(
            {
                int(vertex_id)
                for face_id in suspect_ids
                for vertex_id in mesh_faces[int(face_id)]
                if int(vertex_id) not in boundary_set
            }
        )
        for vertex_id in movable_vertices:
            current = result[int(vertex_id)].copy()
            for target in targets_for(int(vertex_id), result):
                for fraction in fractions:
                    positions = {
                        int(vertex_id): current + fraction * (target - current)
                    }
                    changed = (int(vertex_id),)
                    if not candidate_is_geometrically_safe(changed, positions):
                        continue
                    affected = affected_faces(changed)
                    candidate_mask = inverted_for_faces(affected, positions)
                    candidate_count = (
                        current_count
                        - int(np.count_nonzero(inverted[affected]))
                        + int(np.count_nonzero(candidate_mask))
                    )
                    if candidate_count >= current_count:
                        continue
                    correction = float(
                        np.linalg.norm(positions[int(vertex_id)] - current)
                    )
                    score = (candidate_count, correction)
                    if best is None or score < (best[0], best[1]):
                        best = (
                            candidate_count,
                            correction,
                            positions,
                            affected,
                            candidate_mask,
                        )

        # Some folds straddle two displaced interior vertices and neither can
        # cross the orientation barrier alone.  Retract that pair together,
        # under the same strict acceptance checks.
        if best is None:
            for face_id in suspect_ids:
                movable = tuple(
                    int(value)
                    for value in mesh_faces[int(face_id)]
                    if int(value) not in boundary_set
                )
                if len(movable) < 2:
                    continue
                pair = movable[:2]
                pair_targets = tuple(targets_for(value, result)[0] for value in pair)
                for fraction in fractions:
                    positions = {
                        int(vertex_id): result[int(vertex_id)]
                        + fraction * (target - result[int(vertex_id)])
                        for vertex_id, target in zip(pair, pair_targets)
                    }
                    if not candidate_is_geometrically_safe(pair, positions):
                        continue
                    affected = affected_faces(pair)
                    candidate_mask = inverted_for_faces(affected, positions)
                    candidate_count = (
                        current_count
                        - int(np.count_nonzero(inverted[affected]))
                        + int(np.count_nonzero(candidate_mask))
                    )
                    if candidate_count >= current_count:
                        continue
                    correction = float(
                        max(
                            np.linalg.norm(
                                positions[int(vertex_id)] - result[int(vertex_id)]
                            )
                            for vertex_id in pair
                        )
                    )
                    score = (candidate_count, correction)
                    if best is None or score < (best[0], best[1]):
                        best = (
                            candidate_count,
                            correction,
                            positions,
                            affected,
                            candidate_mask,
                        )
        if best is None:
            break
        for vertex_id, position in best[2].items():
            result[int(vertex_id)] = position
        inverted[best[3]] = best[4]
        repair_count += 1

    maximum_correction = float(
        np.linalg.norm(result - initial_result, axis=1).max(initial=0.0)
    )
    return result, repair_count, maximum_correction


def _repair_flipped_boundary_ears(
    source_points: np.ndarray,
    result_points: np.ndarray,
    faces: np.ndarray,
    boundary_ids: np.ndarray,
    candidate_face_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Repair visible-source triangles folded by moving the shared seam.

    This runs before generated side walls and caps exist: those faces must use
    the final, audited visible boundary.  It repairs the source-side annulus,
    not a missing split cap, by changing only a local quad diagonal.
    """

    repaired = np.asarray(faces, dtype=np.int64).copy()
    repair_count = 0
    edge_faces = _build_edge_face_map(repaired)
    source_triangles = source_points[repaired]
    result_triangles = result_points[repaired]
    source_normals = np.cross(
        source_triangles[:, 1] - source_triangles[:, 0],
        source_triangles[:, 2] - source_triangles[:, 0],
    )
    result_normals = np.cross(
        result_triangles[:, 1] - result_triangles[:, 0],
        result_triangles[:, 2] - result_triangles[:, 0],
    )
    source_lengths = np.linalg.norm(source_normals, axis=1)
    result_lengths = np.linalg.norm(result_normals, axis=1)
    source_shape_quality = _triangle_shape_quality(source_triangles)

    def neighbor_face_ids(face_id: int) -> set[int]:
        face = repaired[int(face_id)]
        return {
            int(neighbor)
            for edge in _face_edge_keys(face)
            for neighbor in edge_faces.get(edge, ())
            if int(neighbor) != int(face_id)
        }

    def face_is_suspect(face_id: int) -> bool:
        if (
            float(source_shape_quality[int(face_id)])
            < _SOURCE_NORMAL_MIN_SHAPE_QUALITY
        ):
            return False
        denominator = source_lengths[int(face_id)] * result_lengths[int(face_id)]
        if denominator <= 1e-24:
            return False
        source_result_cosine = float(
            np.dot(source_normals[int(face_id)], result_normals[int(face_id)])
            / denominator
        )
        if source_result_cosine >= -1e-8:
            return False
        opposed = 0
        valid_neighbors = 0
        for neighbor_id in neighbor_face_ids(int(face_id)):
            local_denominator = (
                result_lengths[int(face_id)] * result_lengths[int(neighbor_id)]
            )
            if local_denominator <= 1e-24:
                continue
            valid_neighbors += 1
            if (
                float(
                    np.dot(
                        result_normals[int(face_id)],
                        result_normals[int(neighbor_id)],
                    )
                    / local_denominator
                )
                < -0.05
            ):
                opposed += 1
        return bool(
            valid_neighbors == 0 or opposed / valid_neighbors >= 2.0 / 3.0
        )

    candidate_faces = (
        np.arange(len(repaired), dtype=np.int64)
        if candidate_face_mask is None
        else np.flatnonzero(np.asarray(candidate_face_mask, dtype=bool))
    )
    suspects = {
        int(face_id)
        for face_id in candidate_faces
        if face_is_suspect(int(face_id))
    }
    seen_replacements: set[
        tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]
    ] = set()

    def replacement_key(
        face_ids: list[int] | tuple[int, ...] | np.ndarray,
        proposed: np.ndarray,
    ) -> tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]:
        return (
            tuple(sorted(int(value) for value in face_ids)),
            tuple(
                sorted(
                    tuple(sorted(int(vertex_id) for vertex_id in face))
                    for face in np.asarray(proposed, dtype=np.int64)
                )
            ),
        )

    def apply_replacement(
        face_ids: list[int] | tuple[int, ...] | np.ndarray,
        proposed: np.ndarray,
    ) -> None:
        changed_ids = np.asarray(face_ids, dtype=np.int64)
        affected = set(int(value) for value in changed_ids)
        for face_id in changed_ids:
            affected.update(neighbor_face_ids(int(face_id)))
        _replace_faces_in_edge_map(
            repaired,
            changed_ids,
            proposed,
            edge_faces,
        )
        source_triangles = source_points[repaired[changed_ids]]
        result_triangles = result_points[repaired[changed_ids]]
        source_normals[changed_ids] = np.cross(
            source_triangles[:, 1] - source_triangles[:, 0],
            source_triangles[:, 2] - source_triangles[:, 0],
        )
        result_normals[changed_ids] = np.cross(
            result_triangles[:, 1] - result_triangles[:, 0],
            result_triangles[:, 2] - result_triangles[:, 0],
        )
        source_lengths[changed_ids] = np.linalg.norm(
            source_normals[changed_ids], axis=1
        )
        result_lengths[changed_ids] = np.linalg.norm(
            result_normals[changed_ids], axis=1
        )
        source_shape_quality[changed_ids] = _triangle_shape_quality(
            source_triangles
        )
        for face_id in changed_ids:
            affected.update(neighbor_face_ids(int(face_id)))
        suspects.difference_update(affected)
        suspects.update(
            int(face_id)
            for face_id in affected
            if face_is_suspect(int(face_id))
        )

    # Every accepted replacement updates only a local topology patch.  Bound
    # work by the actual suspect count rather than the entire boundary: dense
    # vendor-painted rims can have thousands of vertices but only a localized
    # folded patch.  Remaining defects are still rejected by the quality audit
    # below instead of spending minutes exploring thousands of local remeshes.
    repair_budget = min(512, max(64, 2 * int(len(suspects))))
    for _ in range(repair_budget):
        if not suspects:
            break
        changed = False
        for face_id in sorted(suspects):
            face = repaired[int(face_id)]
            candidates: list[tuple[float, int, np.ndarray]] = []
            for left, right in (
                (int(face[0]), int(face[1])),
                (int(face[1]), int(face[2])),
                (int(face[2]), int(face[0])),
            ):
                incident = edge_faces.get(tuple(sorted((left, right))), [])
                if len(incident) != 2:
                    continue
                neighbor_id = next(
                    value for value in incident if int(value) != face_id
                )
                neighbor = repaired[int(neighbor_id)]
                source_opposite = next(
                    int(value) for value in face if int(value) not in (left, right)
                )
                neighbor_opposite = next(
                    int(value)
                    for value in neighbor
                    if int(value) not in (left, right)
                )
                diagonal = tuple(sorted((source_opposite, neighbor_opposite)))
                if diagonal in edge_faces:
                    continue
                proposed = np.asarray(
                    (
                        [source_opposite, left, neighbor_opposite],
                        [source_opposite, neighbor_opposite, right],
                    ),
                    dtype=np.int64,
                )
                if not _replacement_preserves_winding(
                    repaired,
                    edge_faces,
                    (face_id, neighbor_id),
                    proposed,
                ):
                    continue
                source_proposed = source_points[proposed]
                result_proposed = result_points[proposed]
                source_proposed_normals = np.cross(
                    source_proposed[:, 1] - source_proposed[:, 0],
                    source_proposed[:, 2] - source_proposed[:, 0],
                )
                result_proposed_normals = np.cross(
                    result_proposed[:, 1] - result_proposed[:, 0],
                    result_proposed[:, 2] - result_proposed[:, 0],
                )
                source_area = np.linalg.norm(source_proposed_normals, axis=1)
                result_area = np.linalg.norm(result_proposed_normals, axis=1)
                if float(source_area.min()) <= 1e-12 or float(result_area.min()) <= 1e-12:
                    continue
                source_result_cosine = np.einsum(
                    "ij,ij->i", source_proposed_normals, result_proposed_normals
                ) / (source_area * result_area)
                result_pair_cosine = float(
                    np.dot(result_proposed_normals[0], result_proposed_normals[1])
                    / (result_area[0] * result_area[1])
                )
                if result_pair_cosine < 0.05:
                    continue
                if replacement_key((face_id, neighbor_id), proposed) in seen_replacements:
                    continue
                score = result_pair_cosine
                candidates.append((score, int(neighbor_id), proposed))
            if not candidates:
                # One edge flip may be insufficient when two adjacent ears
                # cross together.  Retriangulate the smallest one- or two-ring
                # face patch on the final surface, preserving its exact outer
                # boundary and face count.
                patch_ids = {int(face_id)}
                replacement: tuple[list[int], np.ndarray] | None = None
                for _patch_ring in range(6):
                    expanded = set(patch_ids)
                    for patch_face_id in patch_ids:
                        patch_face = repaired[int(patch_face_id)]
                        for left, right in (
                            (patch_face[0], patch_face[1]),
                            (patch_face[1], patch_face[2]),
                            (patch_face[2], patch_face[0]),
                        ):
                            expanded.update(
                                edge_faces.get(
                                    tuple(sorted((int(left), int(right)))), ()
                                )
                            )
                    patch_ids = expanded
                    ordered_patch_ids = sorted(int(value) for value in patch_ids)
                    patch_faces = repaired[ordered_patch_ids]
                    patch_loops = boundary_loops(patch_faces)
                    if len(patch_loops) != 1:
                        continue
                    patch_loop = [int(value) for value in patch_loops[0]]
                    if len(patch_loop) - 2 != len(ordered_patch_ids):
                        continue
                    fallback_normal = np.sum(
                        source_normals[ordered_patch_ids], axis=0
                    )
                    local_triangles, _normal, _record = triangulate_ordered_loop_3d(
                        result_points[np.asarray(patch_loop, dtype=np.int64)],
                        fallback_normal,
                    )
                    if len(local_triangles) != len(ordered_patch_ids):
                        continue
                    proposed = np.asarray(
                        [
                            [patch_loop[int(index)] for index in triangle]
                            for triangle in local_triangles
                        ],
                        dtype=np.int64,
                    )
                    proposed_edges: dict[tuple[int, int], int] = {}
                    for proposed_face in proposed:
                        for left, right in (
                            (proposed_face[0], proposed_face[1]),
                            (proposed_face[1], proposed_face[2]),
                            (proposed_face[2], proposed_face[0]),
                        ):
                            key = tuple(sorted((int(left), int(right))))
                            proposed_edges[key] = proposed_edges.get(key, 0) + 1
                    patch_id_set = set(ordered_patch_ids)
                    if any(
                        count == 2
                        and any(
                            int(owner) not in patch_id_set
                            for owner in edge_faces.get(edge, ())
                        )
                        for edge, count in proposed_edges.items()
                    ):
                        continue
                    # The source patch is precisely what is being repaired, so
                    # its per-triangle normals are diagnostic rather than an
                    # immutable orientation constraint.  Choose the global
                    # winding that is coherent with the untouched result-side
                    # neighbours, then require the whole replacement patch to
                    # be locally coherent on the final surface.
                    orientation_candidates = (
                        proposed,
                        proposed[:, [0, 2, 1]],
                    )
                    best_oriented: np.ndarray | None = None
                    best_coherence = -np.inf
                    for oriented in orientation_candidates:
                        if not _replacement_preserves_winding(
                            repaired,
                            edge_faces,
                            ordered_patch_ids,
                            oriented,
                        ):
                            continue
                        result_proposed = result_points[oriented]
                        proposed_normals = np.cross(
                            result_proposed[:, 1] - result_proposed[:, 0],
                            result_proposed[:, 2] - result_proposed[:, 0],
                        )
                        proposed_lengths = np.linalg.norm(proposed_normals, axis=1)
                        if float(proposed_lengths.min()) <= 1e-12:
                            continue
                        oriented_edge_faces: dict[tuple[int, int], list[int]] = {}
                        for proposed_face_id, proposed_face in enumerate(oriented):
                            for left, right in (
                                (proposed_face[0], proposed_face[1]),
                                (proposed_face[1], proposed_face[2]),
                                (proposed_face[2], proposed_face[0]),
                            ):
                                oriented_edge_faces.setdefault(
                                    tuple(sorted((int(left), int(right)))), []
                                ).append(int(proposed_face_id))
                        coherence: list[float] = []
                        for edge, local_owners in oriented_edge_faces.items():
                            if len(local_owners) == 2:
                                left_id, right_id = local_owners
                                coherence.append(
                                    float(
                                        np.dot(
                                            proposed_normals[left_id],
                                            proposed_normals[right_id],
                                        )
                                        / (
                                            proposed_lengths[left_id]
                                            * proposed_lengths[right_id]
                                        )
                                    )
                                )
                                continue
                            outside_owners = [
                                int(owner)
                                for owner in edge_faces.get(edge, ())
                                if int(owner) not in patch_id_set
                            ]
                            if len(outside_owners) != 1:
                                continue
                            local_id = int(local_owners[0])
                            outside_id = int(outside_owners[0])
                            denominator = (
                                proposed_lengths[local_id]
                                * result_lengths[outside_id]
                            )
                            if denominator <= 1e-24:
                                coherence.append(-1.0)
                                continue
                            coherence.append(
                                float(
                                    np.dot(
                                        proposed_normals[local_id],
                                        result_normals[outside_id],
                                    )
                                    / denominator
                                )
                            )
                        minimum_coherence = min(coherence, default=1.0)
                        if minimum_coherence > best_coherence:
                            best_coherence = minimum_coherence
                            best_oriented = oriented
                    if best_oriented is None or best_coherence < -0.05:
                        continue
                    if (
                        replacement_key(ordered_patch_ids, best_oriented)
                        in seen_replacements
                    ):
                        continue
                    replacement = (ordered_patch_ids, best_oriented)
                    break
                if replacement is None:
                    continue
                ordered_patch_ids, proposed = replacement
                seen_replacements.add(replacement_key(ordered_patch_ids, proposed))
                apply_replacement(ordered_patch_ids, proposed)
                repair_count += 1
                changed = True
                break
            _, neighbor_id, proposed = max(candidates, key=lambda item: item[0])
            seen_replacements.add(replacement_key((face_id, neighbor_id), proposed))
            apply_replacement((face_id, neighbor_id), proposed)
            repair_count += 1
            changed = True
            break
        if not changed:
            break
    return repaired, int(repair_count)


def _surface_band_deformation(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    boundary_ids: np.ndarray,
    target_boundary_points: np.ndarray,
    band_width_mm: float,
    failure_sink: Callable[[dict[str, Any]], None] | None = None,
    minimum_sparse_inversion_angle_degrees: float = 3.0,
    validation_mode: str = "strict",
    smoothing_policy: SeamSmoothingPolicy | None = None,
) -> tuple[np.ndarray, dict]:
    """Move the real visible seam and diffuse its displacement into its patch.

    The planar-arc target is the manufacturing boundary, not a hidden guide.
    Its displacement is blended over a topology-connected surface band so the
    source triangles beside the seam do not become a sawtooth collar.  Nearest
    boundary interpolation is evaluated only on the connected candidate band;
    a nearby back face on a thin model therefore cannot move merely because it
    is close in Euclidean space.
    """

    policy = smoothing_policy or SeamSmoothingPolicy.named("print-balanced")
    points = np.asarray(source_vertices, dtype=np.float64)
    faces = np.asarray(source_faces, dtype=np.int64)
    original_faces = faces.copy()
    original_over_shared_edges, original_inconsistent_edges = (
        _directed_edge_topology_issues(original_faces)
    )
    boundary = np.asarray(boundary_ids, dtype=np.int64)
    targets = np.asarray(target_boundary_points, dtype=np.float64)
    if targets.shape != (len(boundary), 3):
        raise PlanarArcError("visible boundary target does not match source ids")
    if len(boundary) != len(np.unique(boundary)):
        raise PlanarArcError("visible boundary contains repeated local ids")
    if common.cKDTree is None:
        raise RuntimeError("scipy dependency is not loaded")

    band = max(float(band_width_mm), 1e-6)
    boundary_source = points[boundary]
    boundary_displacement = targets - boundary_source
    neighbor_count = min(4, len(boundary))
    distance, nearest = common.cKDTree(boundary_source).query(
        points,
        k=neighbor_count,
        workers=1,
    )
    if neighbor_count == 1:
        distance = np.asarray(distance, dtype=np.float64)[:, None]
        nearest = np.asarray(nearest, dtype=np.int64)[:, None]
    else:
        distance = np.asarray(distance, dtype=np.float64)
        nearest = np.asarray(nearest, dtype=np.int64)

    candidate_mask = distance[:, 0] <= band + 1e-12
    candidate_mask[boundary] = True
    edge_array = np.vstack(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        )
    )
    edge_array = edge_array[
        candidate_mask[edge_array[:, 0]]
        & candidate_mask[edge_array[:, 1]]
    ]
    adjacency: dict[int, set[int]] = {}
    for left, right in edge_array:
        left_id = int(left)
        right_id = int(right)
        adjacency.setdefault(left_id, set()).add(right_id)
        adjacency.setdefault(right_id, set()).add(left_id)

    active_mask = np.zeros(len(points), dtype=bool)
    stack = [(int(value), 0) for value in boundary]
    active_mask[boundary] = True
    while stack:
        current, layer = stack.pop()
        if layer >= policy.maximum_topology_layers:
            continue
        for neighbor in adjacency.get(current, ()):
            if active_mask[int(neighbor)]:
                continue
            active_mask[int(neighbor)] = True
            stack.append((int(neighbor), layer + 1))

    safe_distance = np.maximum(distance, 0.02)
    weights = 1.0 / (safe_distance * safe_distance)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-24)
    interpolated = np.sum(
        boundary_displacement[nearest] * weights[:, :, None],
        axis=1,
    )
    # Relax the displacement *direction field* over mesh connectivity before
    # applying the geometric C2 falloff.  Nearest-boundary interpolation alone
    # can assign noticeably different tangential motion to the two interior
    # vertices of one tiny vendor triangle and flip that face even though the
    # target seam is valid.  Boundary values stay Dirichlet-pinned; inactive
    # sheets never enter the stencil.
    boundary_mask = np.zeros(len(points), dtype=bool)
    boundary_mask[boundary] = True
    relaxed = interpolated.copy()
    active_non_boundary = np.flatnonzero(active_mask & ~boundary_mask)
    relaxation_iterations = 48 if len(boundary) <= 300 else 192
    relaxation_vertex_ids: list[int] = []
    relaxation_neighbor_ids: list[int] = []
    relaxation_neighbor_counts: list[int] = []
    for vertex_id in active_non_boundary:
        neighbors = sorted(adjacency.get(int(vertex_id), ()))
        if not neighbors:
            continue
        relaxation_vertex_ids.append(int(vertex_id))
        relaxation_neighbor_ids.extend(int(value) for value in neighbors)
        relaxation_neighbor_counts.append(len(neighbors))
    relaxation_vertices = np.asarray(relaxation_vertex_ids, dtype=np.int64)
    relaxation_neighbors = np.asarray(relaxation_neighbor_ids, dtype=np.int64)
    relaxation_counts = np.asarray(relaxation_neighbor_counts, dtype=np.float64)
    relaxation_offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(
                np.asarray(relaxation_neighbor_counts[:-1], dtype=np.int64)
            ),
        )
    )
    relaxation_started_at = time.perf_counter()
    for _ in range(relaxation_iterations):
        updated = relaxed.copy()
        if len(relaxation_vertices):
            neighbor_sums = np.add.reduceat(
                relaxed[relaxation_neighbors],
                relaxation_offsets,
                axis=0,
            )
            neighbor_means = neighbor_sums / relaxation_counts[:, None]
            updated[relaxation_vertices] = (
                0.35 * relaxed[relaxation_vertices]
                + 0.65 * neighbor_means
            )
        updated[boundary] = boundary_displacement
        relaxed = updated
    relaxation_seconds = float(time.perf_counter() - relaxation_started_at)
    runtime_log(
        "几何性能",
        "surface_band_relaxation_done",
        "可见表面带平滑完成",
        duration_seconds=round(relaxation_seconds, 3),
        iterations=int(relaxation_iterations),
        active_vertices=int(len(relaxation_vertices)),
    )
    normalized_distance = np.clip(distance[:, 0] / band, 0.0, 1.0)
    remaining = 1.0 - normalized_distance
    fade = remaining * remaining * (3.0 - 2.0 * remaining)
    applied = relaxed * fade[:, None]
    applied[~active_mask] = 0.0
    applied[boundary] = boundary_displacement

    result = points + applied
    result[boundary] = targets
    boundary_repair_started_at = time.perf_counter()
    boundary_repair_candidate_mask = np.any(active_mask[faces], axis=1)
    runtime_log(
        "几何性能",
        "surface_band_boundary_repair_start",
        "开始检查并修复边界耳片",
        candidate_faces=int(np.count_nonzero(boundary_repair_candidate_mask)),
        boundary_vertices=int(len(boundary)),
    )
    if np.count_nonzero(boundary_repair_candidate_mask) > 50_000:
        # Local ear search is combinatorial on highly fragmented paint bands.
        # Do not mutate a huge source band speculatively; the vectorized
        # inversion/degeneracy audit immediately below remains blocking.
        boundary_ear_repairs = 0
        runtime_log(
            "几何性能", "surface_band_boundary_repair_bounded",
            "候选表面带过大，跳过组合式耳片搜索并交由向量化质量审核",
            candidate_faces=int(np.count_nonzero(boundary_repair_candidate_mask)),
            threshold_faces=50_000,
        )
    else:
        faces, boundary_ear_repairs = _repair_flipped_boundary_ears(
            points,
            result,
            faces,
            boundary,
            candidate_face_mask=boundary_repair_candidate_mask,
        )
    boundary_repair_seconds = float(
        time.perf_counter() - boundary_repair_started_at
    )
    runtime_log(
        "几何性能",
        "surface_band_boundary_repair_done",
        "边界耳片修复完成",
        duration_seconds=round(boundary_repair_seconds, 3),
        repairs=int(boundary_ear_repairs),
    )

    result_triangles_after_boundary_repair = result[faces]
    result_areas_after_boundary_repair = np.linalg.norm(
        np.cross(
            result_triangles_after_boundary_repair[:, 1]
            - result_triangles_after_boundary_repair[:, 0],
            result_triangles_after_boundary_repair[:, 2]
            - result_triangles_after_boundary_repair[:, 0],
        ),
        axis=1,
    )
    locked_boundary_face_mask = np.all(boundary_mask[faces], axis=1)
    locked_boundary_degenerate_face_count = int(
        np.count_nonzero(
            locked_boundary_face_mask
            & (result_areas_after_boundary_repair <= 1e-12)
        )
    )
    untangle_started_at = time.perf_counter()
    if np.count_nonzero(boundary_repair_candidate_mask) > 50_000:
        interior_untangle_repairs = 0
        maximum_interior_correction = 0.0
        untangle_skipped_reason = "large_band_vectorized_audit"
    elif locked_boundary_degenerate_face_count:
        interior_untangle_repairs = 0
        maximum_interior_correction = 0.0
        untangle_skipped_reason = "locked_boundary_degenerate_faces"
    else:
        result, interior_untangle_repairs, maximum_interior_correction = (
            _untangle_interior_surface_vertices(
                points,
                result,
                faces,
                boundary,
            )
        )
        untangle_skipped_reason = None
    untangle_seconds = float(time.perf_counter() - untangle_started_at)
    runtime_log(
        "几何性能",
        "surface_band_untangle_done",
        (
            "内部解缠因固定边界退化面而提前结束"
            if untangle_skipped_reason
            else "内部解缠完成"
        ),
        duration_seconds=round(untangle_seconds, 3),
        repairs=int(interior_untangle_repairs),
        skipped_reason=untangle_skipped_reason,
        locked_boundary_degenerate_faces=(
            locked_boundary_degenerate_face_count
        ),
    )
    np.asarray(source_faces, dtype=np.int64)[...] = faces
    result_over_shared_edges, result_inconsistent_edges = (
        _directed_edge_topology_issues(faces)
    )
    introduced_over_shared_edges = (
        result_over_shared_edges - original_over_shared_edges
    )
    introduced_inconsistent_edges = (
        result_inconsistent_edges - original_inconsistent_edges
    )
    applied = result - points
    moved_mask = np.linalg.norm(applied, axis=1) > 1e-12
    affected_face_mask = np.any(moved_mask[faces], axis=1)
    affected_faces = faces[affected_face_mask]
    if len(affected_faces):
        source_triangles = points[affected_faces]
        result_triangles = result[affected_faces]
        source_normals = np.cross(
            source_triangles[:, 1] - source_triangles[:, 0],
            source_triangles[:, 2] - source_triangles[:, 0],
        )
        result_normals = np.cross(
            result_triangles[:, 1] - result_triangles[:, 0],
            result_triangles[:, 2] - result_triangles[:, 0],
        )
        source_areas = np.linalg.norm(source_normals, axis=1)
        result_areas = np.linalg.norm(result_normals, axis=1)
        source_shape_quality = _triangle_shape_quality(source_triangles)
        source_minimum_angles = _triangle_minimum_angles_degrees(
            source_triangles
        )
        result_minimum_angles = _triangle_minimum_angles_degrees(
            result_triangles
        )
        valid_source = source_areas > 1e-15
        valid_result = result_areas > 1e-12
        comparable = valid_source & valid_result
        cosine = np.ones(len(affected_faces), dtype=np.float64)
        cosine[comparable] = np.einsum(
            "ij,ij->i",
            source_normals[comparable],
            result_normals[comparable],
        ) / (source_areas[comparable] * result_areas[comparable])
        source_edges = np.stack(
            (
                np.linalg.norm(source_triangles[:, 1] - source_triangles[:, 0], axis=1),
                np.linalg.norm(source_triangles[:, 2] - source_triangles[:, 1], axis=1),
                np.linalg.norm(source_triangles[:, 0] - source_triangles[:, 2], axis=1),
            ),
            axis=1,
        )
        result_edges = np.stack(
            (
                np.linalg.norm(result_triangles[:, 1] - result_triangles[:, 0], axis=1),
                np.linalg.norm(result_triangles[:, 2] - result_triangles[:, 1], axis=1),
                np.linalg.norm(result_triangles[:, 0] - result_triangles[:, 2], axis=1),
            ),
            axis=1,
        )
        edge_stretch = result_edges / np.maximum(source_edges, 0.02)
        degenerate_faces = int(np.count_nonzero(~valid_result))
        source_normal_sign_changes = cosine < -1e-8
        affected_global_ids = np.flatnonzero(affected_face_mask)
        degenerate_face_ids = affected_global_ids[~valid_result]
        overstretched_face_ids = affected_global_ids[
            np.any(edge_stretch > 8.0 + 1e-9, axis=1)
        ]
        locally_inverted = _locally_inverted_face_mask(
            points,
            result,
            faces,
        )
        locally_inverted &= affected_face_mask
        reversed_faces = int(np.count_nonzero(locally_inverted))
        inverted_affected_angles = result_minimum_angles[
            locally_inverted[affected_face_mask]
        ]
        sparse_inversion_audit = _sparse_local_inversion_audit(
            faces,
            locally_inverted,
            audited_face_count=len(affected_faces),
            result_minimum_angles_degrees=inverted_affected_angles,
            minimum_result_angle_degrees=(
                policy.minimum_reversed_angle_degrees
            ),
            maximum_ratio=policy.maximum_introduced_reversed_ratio,
            maximum_edge_connected_cluster=policy.maximum_reversed_cluster_faces,
        )
        boundary_set = set(int(value) for value in boundary)
        inverted_face_ids = np.flatnonzero(locally_inverted)
        inverted_faces_touching_boundary = int(
            sum(
                any(int(vertex_id) in boundary_set for vertex_id in faces[int(face_id)])
                for face_id in inverted_face_ids
            )
        )
        if (
            inverted_faces_touching_boundary
            and not policy.allow_sparse_seam_reversals
        ):
            sparse_inversion_audit = {
                **sparse_inversion_audit,
                "accepted": False,
                "rejected_reason": "introduced_inversion_touches_final_seam",
            }
        user_reviewed_advisory = str(validation_mode) == "advisory"
        normal_change_advisory_accepted = bool(
            user_reviewed_advisory
            and reversed_faces > 0
            and not sparse_inversion_audit["accepted"]
        )
        blocking_reversed_faces = (
            0
            if sparse_inversion_audit["accepted"]
            or normal_change_advisory_accepted
            else reversed_faces
        )
        locally_inverted_samples = [
            {
                "face_index": int(face_id),
                "vertex_ids": [int(value) for value in faces[int(face_id)]],
                "boundary_vertex_count": int(
                    sum(
                        int(value) in boundary_set
                        for value in faces[int(face_id)]
                    )
                ),
                "source_result_normal_cosine": float(
                    cosine[
                        int(np.searchsorted(affected_global_ids, int(face_id)))
                    ]
                ),
                "source_triangle_shape_quality": float(
                    source_shape_quality[
                        int(np.searchsorted(affected_global_ids, int(face_id)))
                    ]
                ),
                "source_minimum_angle_degrees": float(
                    source_minimum_angles[
                        int(np.searchsorted(affected_global_ids, int(face_id)))
                    ]
                ),
                "result_minimum_angle_degrees": float(
                    result_minimum_angles[
                        int(np.searchsorted(affected_global_ids, int(face_id)))
                    ]
                ),
            }
            for face_id in np.flatnonzero(locally_inverted)[:8]
        ]
        unstable_source_normal_changes = (
            source_normal_sign_changes
            & (source_shape_quality < _SOURCE_NORMAL_MIN_SHAPE_QUALITY)
        )
        unstable_source_normal_samples = [
            {
                "face_index": int(affected_global_ids[int(local_id)]),
                "vertex_ids": [
                    int(value)
                    for value in faces[int(affected_global_ids[int(local_id)])]
                ],
                "source_result_normal_cosine": float(cosine[int(local_id)]),
                "source_triangle_shape_quality": float(
                    source_shape_quality[int(local_id)]
                ),
                "source_minimum_angle_degrees": float(
                    source_minimum_angles[int(local_id)]
                ),
                "result_minimum_angle_degrees": float(
                    result_minimum_angles[int(local_id)]
                ),
                "classification": "advisory_unstable_source_normal",
            }
            for local_id in np.flatnonzero(unstable_source_normal_changes)[:8]
        ]
        source_normal_sign_change_count = int(
            np.count_nonzero(source_normal_sign_changes)
        )
        near_orthogonal_faces = int(
            np.count_nonzero((cosine >= -1e-8) & (cosine < 0.05))
        )
        minimum_normal_cosine = float(cosine.min())
        maximum_edge_stretch = float(edge_stretch.max())
    else:
        degenerate_faces = 0
        reversed_faces = 0
        near_orthogonal_faces = 0
        source_normal_sign_change_count = 0
        unstable_source_normal_changes = np.zeros(0, dtype=bool)
        unstable_source_normal_samples = []
        locally_inverted_samples = []
        minimum_normal_cosine = 1.0
        maximum_edge_stretch = 1.0
        affected_global_ids = np.empty(0, dtype=np.int64)
        degenerate_face_ids = np.empty(0, dtype=np.int64)
        overstretched_face_ids = np.empty(0, dtype=np.int64)
        locally_inverted = np.zeros(len(faces), dtype=bool)
        blocking_reversed_faces = 0
        normal_change_advisory_accepted = False
        sparse_inversion_audit = _sparse_local_inversion_audit(
            faces,
            locally_inverted,
            audited_face_count=0,
            result_minimum_angles_degrees=np.empty(0, dtype=np.float64),
            minimum_result_angle_degrees=(
                policy.minimum_reversed_angle_degrees
            ),
            maximum_ratio=policy.maximum_introduced_reversed_ratio,
            maximum_edge_connected_cluster=policy.maximum_reversed_cluster_faces,
        )
        inverted_faces_touching_boundary = 0

    boundary_error = float(
        np.linalg.norm(result[boundary] - targets, axis=1).max(initial=0.0)
    )
    source_triangle_double_areas = np.linalg.norm(
        np.cross(
            points[faces][:, 1] - points[faces][:, 0],
            points[faces][:, 2] - points[faces][:, 0],
        ),
        axis=1,
    )
    affected_area_ratio = float(
        source_triangle_double_areas[affected_face_mask].sum()
        / max(float(source_triangle_double_areas.sum()), 1e-24)
    )
    affected_area_within_budget = bool(
        len(faces) < 1000
        or affected_area_ratio <= policy.maximum_affected_area_ratio + 1e-12
    )
    displacement = np.linalg.norm(result - points, axis=1)
    # Coverage is not visual severity: a smooth sub-nozzle displacement can
    # touch a broad finely tessellated band without producing a visible ridge.
    # Only explicit advisory mode may replace coverage budgets with the actual
    # displacement envelope; topology and printable-face gates remain blocking.
    (
        visual_extent_advisory_accepted,
        maximum_displacement,
        p95_displacement,
    ) = _visual_displacement_advisory(
        validation_mode,
        displacement,
        policy,
    )
    # The seam itself is the requested interface replacement.  The collateral
    # source budget counts only vertices reached beyond that interface.
    moved_vertex_mask = displacement > 1e-12
    moved_vertex_mask[boundary] = False
    affected_vertex_count = int(np.count_nonzero(moved_vertex_mask))
    maximum_affected_vertices = min(
        int(policy.maximum_affected_vertices),
        int(np.floor(len(points) * policy.maximum_affected_vertex_ratio)),
    )
    affected_vertices_within_budget = bool(
        len(points) < 1000 or affected_vertex_count <= maximum_affected_vertices
    )
    maximum_allowed_edge_stretch = float(policy.maximum_edge_stretch_ratio)
    if str(validation_mode) == "advisory":
        maximum_allowed_edge_stretch = max(maximum_allowed_edge_stretch, 128.0)
    edge_stretch_advisory_accepted = bool(
        str(validation_mode) == "advisory"
        and maximum_edge_stretch > 8.0 + 1e-9
        and maximum_edge_stretch <= maximum_allowed_edge_stretch + 1e-9
    )
    valid = bool(
        boundary_error <= 1e-9
        and degenerate_faces == 0
        and blocking_reversed_faces == 0
        and not introduced_over_shared_edges
        and not introduced_inconsistent_edges
        and (affected_area_within_budget or visual_extent_advisory_accepted)
        and (affected_vertices_within_budget or visual_extent_advisory_accepted)
        and maximum_edge_stretch <= maximum_allowed_edge_stretch + 1e-9
    )
    quality = {
        "valid": valid,
        "strategy": "topology_connected_c2_visible_surface_band",
        "surface_band_validation_mode": str(validation_mode),
        "seam_smoothing_profile": policy.profile,
        "surface_band_affected_area_ratio": affected_area_ratio,
        "maximum_affected_area_ratio": policy.maximum_affected_area_ratio,
        "affected_area_within_budget": affected_area_within_budget,
        "visual_extent_advisory_accepted": visual_extent_advisory_accepted,
        "p95_vertex_displacement_mm": p95_displacement,
        "affected_vertex_count": affected_vertex_count,
        "maximum_affected_vertices": maximum_affected_vertices,
        "affected_vertices_within_budget": affected_vertices_within_budget,
        "maximum_topology_layers": policy.maximum_topology_layers,
        "introduced_inverted_faces_touching_boundary": inverted_faces_touching_boundary,
        "surface_band_width_mm": band,
        "visible_boundary_vertex_count": int(len(boundary)),
        "surface_band_candidate_vertices": int(np.count_nonzero(candidate_mask)),
        "surface_band_connected_vertices": int(np.count_nonzero(active_mask)),
        "surface_band_moved_vertices": int(np.count_nonzero(moved_mask)),
        "surface_band_affected_faces": int(len(affected_faces)),
        "surface_band_relaxation_iterations": int(relaxation_iterations),
        "surface_band_relaxation_seconds": relaxation_seconds,
        "surface_band_boundary_ear_edge_flips": int(boundary_ear_repairs),
        "surface_band_boundary_repair_seconds": boundary_repair_seconds,
        "surface_band_interior_untangle_repairs": int(interior_untangle_repairs),
        "surface_band_interior_untangle_seconds": untangle_seconds,
        "surface_band_untangle_skipped_reason": untangle_skipped_reason,
        "locked_boundary_degenerate_face_count": (
            locked_boundary_degenerate_face_count
        ),
        "maximum_interior_untangle_correction_mm": float(maximum_interior_correction),
        "boundary_match_error_mm": boundary_error,
        "maximum_vertex_displacement_mm": maximum_displacement,
        "degenerate_face_count": degenerate_faces,
        "reversed_face_count": reversed_faces,
        "blocking_reversed_face_count": int(blocking_reversed_faces),
        "normal_change_advisory_accepted": bool(
            normal_change_advisory_accepted
        ),
        "sparse_local_inversion_audit": sparse_inversion_audit,
        "source_normal_sign_change_count": source_normal_sign_change_count,
        "unstable_source_normal_sign_change_count": int(
            np.count_nonzero(unstable_source_normal_changes)
        ),
        "unstable_source_normal_samples": unstable_source_normal_samples,
        "locally_inverted_face_samples": locally_inverted_samples,
        "result_over_shared_edge_count": int(len(result_over_shared_edges)),
        "result_inconsistent_shared_edge_count": int(
            len(result_inconsistent_edges)
        ),
        "introduced_over_shared_edge_count": int(
            len(introduced_over_shared_edges)
        ),
        "introduced_inconsistent_shared_edge_count": int(
            len(introduced_inconsistent_edges)
        ),
        "near_orthogonal_face_count": near_orthogonal_faces,
        "minimum_source_normal_cosine": minimum_normal_cosine,
        "maximum_edge_stretch_ratio": maximum_edge_stretch,
        "maximum_allowed_edge_stretch_ratio": float(
            maximum_allowed_edge_stretch
        ),
        "edge_stretch_advisory_accepted": bool(
            edge_stretch_advisory_accepted
        ),
    }
    if failure_sink is not None and not valid:
        try:
            failure_sink(
                {
                    "source_points": points,
                    "result_points": result,
                    "faces": faces,
                    "boundary_ids": boundary,
                    "target_boundary_points": targets,
                    "affected_face_ids": affected_global_ids,
                    "degenerate_face_ids": degenerate_face_ids,
                    "reversed_face_ids": np.flatnonzero(locally_inverted),
                    "overstretched_face_ids": overstretched_face_ids,
                    "quality": dict(quality),
                }
            )
        except Exception as exc:
            runtime_log(
                "失败诊断",
                "retopology_failure_artifact_error",
                "表面带失败图写出失败；质量阻断保持不变",
                error=str(exc),
            )
    return result, quality


class InterfaceRetopologyService:
    """Create one shared, source-id aligned target for both sides of a seam."""

    @staticmethod
    def retopologize_loop(
        points: np.ndarray,
        source_vertex_ids: np.ndarray,
        context: PlanarArcRetopologyContext,
    ) -> tuple[np.ndarray, dict]:
        policy = context.config.smoothing_policy
        loop_points = np.asarray(points, dtype=np.float64)
        loop_diagonal = float(np.linalg.norm(np.ptp(loop_points, axis=0)))
        maximum_target_offset = min(
            float(policy.maximum_displacement_mm),
            max(
                min(2.0, float(policy.maximum_displacement_mm)),
                loop_diagonal * float(policy.maximum_bbox_diagonal_ratio),
            ),
        )
        try:
            target = build_planar_arc_boundary(
                points,
                source_vertex_ids,
                target_samples=context.config.target_samples,
                smooth_passes=context.config.smooth_passes,
                maximum_target_offset_mm=maximum_target_offset,
                maximum_p95_target_offset_mm=policy.p95_displacement_mm,
            )
        except CurveClarityRequired as exc:
            if context.config.surface_band_validation == "advisory":
                # A projected crossing does not prove that the original 3-D
                # source seam self-intersects.  Dense painted models often
                # contain folded/steep seams whose planar fit creates the
                # crossing.  In reviewed advisory mode the safest completion
                # is therefore no geometric edit at all: keep the exact source
                # ring and let cap, topology, Boolean and visual audits remain
                # authoritative.
                proposal_record = dict(exc.proposal.record)
                proposal_record.update(
                    status="source_curve_preserved",
                    source_vertices=int(len(loop_points)),
                    target_samples=int(len(loop_points)),
                    maximum_target_offset_mm=0.0,
                    p95_target_offset_mm=0.0,
                    rms_target_offset_mm=0.0,
                    ambiguous_planar_fit_skipped=True,
                    topology_change=False,
                    requires_user_confirmation=False,
                )
                runtime_log(
                    "planar-arc-retopology",
                    "ambiguous_fit_source_curve_preserved",
                    "投影拟合产生交叉；advisory 模式保留原始三维边界并继续严格几何审核",
                    source_vertices=len(loop_points),
                    projected_crossings=proposal_record.get(
                        "projected_crossings_before", 0),
                )
                return loop_points.copy(), proposal_record
            if context.curve_review_sink is not None:
                context.curve_review_sink(exc)
            raise
        record = dict(target.record)
        record.update(
            {
                "retopology_band_mm": float(context.config.retopology_band_mm),
                "maximum_band_fraction": float(context.config.maximum_band_fraction),
                "target_slope_degrees": float(context.config.target_slope_degrees),
                "minimum_slope_degrees": float(context.config.minimum_slope_degrees),
                "maximum_slope_degrees": float(context.config.maximum_slope_degrees),
                "seam_smoothing_profile": policy.profile,
                "maximum_profile_target_offset_mm": maximum_target_offset,
            }
        )
        return target.target_points, record

    @staticmethod
    def retopologize_local_loops(
        local_vertices: np.ndarray,
        loops: list[list[int]],
        global_vertex_ids: np.ndarray,
        context: PlanarArcRetopologyContext,
        local_faces: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        if context.active_layer_seam is not None:
            if context.config != context.active_layer_seam.config:
                raise PlanarArcError('Layer seam settings changed after planning')
            return context.active_layer_seam.apply(
                local_vertices, local_faces, global_vertex_ids, loops)
        source = np.asarray(local_vertices, dtype=np.float64)
        result = source.copy()
        global_ids = np.asarray(global_vertex_ids, dtype=np.int64)
        runtime_log(
            "planar-arc-retopology",
            "visible_surface_conformance_start",
            "先一致化并审计可见源表面；侧壁和封口面将在边界通过后生成",
            local_faces=0 if local_faces is None else int(len(local_faces)),
            boundary_loops=int(len(loops)),
            generated_split_faces=0,
        )
        records: list[dict] = []
        local_boundary_ids: list[int] = []
        target_boundary_points: list[np.ndarray] = []
        for loop_index, loop in enumerate(loops):
            local_indices = np.asarray(loop, dtype=np.int64)
            if len(np.unique(local_indices)) != len(local_indices):
                raise PlanarArcError("retopology boundary contains repeated vertices")
            target, record = InterfaceRetopologyService.retopologize_loop(
                source[local_indices],
                global_ids[local_indices],
                context,
            )
            target, displacement_strategy = _select_visible_boundary_target(
                source[local_indices],
                target,
                context.config.retopology_band_mm,
                context.config.maximum_safe_target_offset_mm,
            )
            record.update(displacement_strategy)
            local_boundary_ids.extend(int(value) for value in local_indices)
            target_boundary_points.extend(
                np.asarray(point, dtype=np.float64) for point in target
            )
            records.append({"loop_index": int(loop_index), **record})

        if local_boundary_ids:
            boundary_array = np.asarray(local_boundary_ids, dtype=np.int64)
            targets = np.asarray(target_boundary_points, dtype=np.float64)
            unique_boundary_ids, boundary_counts = np.unique(
                boundary_array,
                return_counts=True,
            )
            shared_boundary_ids = unique_boundary_ids[boundary_counts > 1]
            shared_vertex_advisory = bool(len(shared_boundary_ids))
            maximum_shared_target_disagreement = 0.0
            if shared_vertex_advisory:
                loop_edge_sets: list[set[tuple[int, int]]] = []
                for loop in loops:
                    loop_edge_sets.append(
                        {
                            tuple(sorted((int(loop[index]), int(loop[(index + 1) % len(loop)]))))
                            for index in range(len(loop))
                        }
                    )
                shared_edges = set()
                for left_index, left_edges in enumerate(loop_edge_sets):
                    for right_edges in loop_edge_sets[left_index + 1 :]:
                        shared_edges.update(left_edges & right_edges)
                if shared_edges:
                    raise PlanarArcError(
                        "visible retopology loops share a boundary edge"
                    )
                if context.config.surface_band_validation != "advisory":
                    raise PlanarArcError(
                        "visible retopology loops share a boundary vertex"
                    )

                # Touching closed loops form a valid figure-eight junction but
                # their independently smoothed targets can disagree at the
                # shared source id. Keep that junction locked to its original
                # point, then pass each remaining boundary vertex exactly once
                # to the surface-band solver.
                first_positions: dict[int, int] = {}
                unique_order: list[int] = []
                for position, vertex_id in enumerate(boundary_array):
                    value = int(vertex_id)
                    if value not in first_positions:
                        first_positions[value] = int(position)
                        unique_order.append(value)
                for vertex_id in shared_boundary_ids:
                    positions = np.flatnonzero(boundary_array == int(vertex_id))
                    disagreement = np.linalg.norm(
                        targets[positions] - source[int(vertex_id)],
                        axis=1,
                    )
                    maximum_shared_target_disagreement = max(
                        maximum_shared_target_disagreement,
                        float(disagreement.max(initial=0.0)),
                    )
                shared_boundary_set = set(shared_boundary_ids.astype(int).tolist())
                boundary_array = np.asarray(unique_order, dtype=np.int64)
                targets = np.asarray(
                    [
                        (
                            source[vertex_id]
                            if vertex_id in shared_boundary_set
                            else targets[first_positions[vertex_id]]
                        )
                        for vertex_id in unique_order
                    ],
                    dtype=np.float64,
                )
            if local_faces is None:
                result[boundary_array] = targets
                band_quality = {
                    "valid": True,
                    "strategy": "boundary_only_without_surface_faces",
                    "surface_band_width_mm": float(
                        context.config.retopology_band_mm
                    ),
                    "surface_band_moved_vertices": int(len(boundary_array)),
                    "surface_band_affected_faces": 0,
                    "boundary_match_error_mm": 0.0,
                    "degenerate_face_count": 0,
                    "reversed_face_count": 0,
                    "minimum_source_normal_cosine": 1.0,
                    "maximum_edge_stretch_ratio": 1.0,
                }
            else:
                result, band_quality = _surface_band_deformation(
                    source,
                    np.asarray(local_faces, dtype=np.int64),
                    boundary_array,
                    targets,
                    context.config.retopology_band_mm,
                    failure_sink=context.failure_sink,
                    minimum_sparse_inversion_angle_degrees=(
                        1.0
                        if context.config.surface_band_validation == "advisory"
                        else 3.0
                    ),
                    validation_mode=context.config.surface_band_validation,
                    smoothing_policy=context.config.smoothing_policy,
                )
            if not bool(band_quality["valid"]):
                raise PlanarArcError(
                    "visible planar-arc surface band failed quality audit: "
                    f"degenerate_faces={band_quality['degenerate_face_count']}, "
                    f"reversed_faces={band_quality['reversed_face_count']}, "
                    f"boundary_vertices={band_quality.get('visible_boundary_vertex_count')}, "
                    f"edge_flips={band_quality.get('surface_band_boundary_ear_edge_flips')}, "
                    f"affected_area_ratio={band_quality.get('surface_band_affected_area_ratio')}, "
                    f"affected_area_limit={band_quality.get('maximum_affected_area_ratio')}, "
                    f"sparse_audit={band_quality.get('sparse_local_inversion_audit')}, "
                    f"samples={band_quality.get('locally_inverted_face_samples')}, "
                    "maximum_edge_stretch_ratio="
                    f"{band_quality['maximum_edge_stretch_ratio']:.6f}"
                )
            for record in records:
                minimal_loop_preserved = bool(
                    record.get("minimal_loop_preserved", False)
                )
                record.update(
                    {
                        "shared_boundary_vertex_advisory_accepted": bool(
                            shared_vertex_advisory
                        ),
                        "shared_boundary_vertex_count": int(
                            len(shared_boundary_ids)
                        ),
                        "shared_boundary_junction_strategy": (
                            "lock_to_original_source_position"
                            if shared_vertex_advisory
                            else "none"
                        ),
                        "maximum_shared_target_disagreement_mm": float(
                            maximum_shared_target_disagreement
                        ),
                        "visible_top_source_preserved": minimal_loop_preserved,
                        "visible_boundary_retopologized": not minimal_loop_preserved,
                        "interface_retopology_surface_role": (
                            "shared_visible_boundary_and_surface_band"
                        ),
                        "visible_surface_band_quality": dict(band_quality),
                    }
                )
        return result, records
