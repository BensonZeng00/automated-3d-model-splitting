from __future__ import annotations

import math

import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse import coo_matrix, eye, vstack

from .domain import BoundaryFairingConfig, BoundaryFairingContext
from .mesh import taubin_smooth_loop


_EPSILON = 1e-10


def _canonical_order(source_vertex_ids: np.ndarray) -> np.ndarray:
    """Return an orientation-independent order for one closed source loop."""
    ids = np.asarray(source_vertex_ids, dtype=np.int64)
    if len(ids) < 2:
        return np.arange(len(ids), dtype=np.int64)
    start = int(np.argmin(ids))
    forward = np.array([(start + offset) % len(ids) for offset in range(len(ids))], dtype=np.int64)
    reverse = np.array([(start - offset) % len(ids) for offset in range(len(ids))], dtype=np.int64)
    return forward if tuple(ids[forward]) <= tuple(ids[reverse]) else reverse


def _unit_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(values, axis=1)
    valid = lengths > _EPSILON
    result = np.zeros_like(values, dtype=np.float64)
    result[valid] = values[valid] / lengths[valid, None]
    return result, valid


def _loop_geometry(points: np.ndarray, surface_normals: np.ndarray) -> dict[str, np.ndarray]:
    previous_edges = points - np.roll(points, 1, axis=0)
    next_edges = np.roll(points, -1, axis=0) - points
    incoming, valid_previous = _unit_rows(previous_edges)
    outgoing, valid_next = _unit_rows(next_edges)
    tangents, valid_tangent = _unit_rows(incoming + outgoing)
    normals, valid_normal = _unit_rows(surface_normals)
    conormals, valid_conormal = _unit_rows(np.cross(normals, tangents))
    turning = np.degrees(
        np.arccos(np.clip(np.einsum("ij,ij->i", incoming, outgoing), -1.0, 1.0))
    )
    return {
        "previous_lengths": np.linalg.norm(previous_edges, axis=1),
        "next_lengths": np.linalg.norm(next_edges, axis=1),
        "conormals": conormals,
        "turning_degrees": turning,
        "valid": valid_previous & valid_next & valid_tangent & valid_normal & valid_conormal,
        "incoming": incoming,
        "outgoing": outgoing,
    }


def _fairness_system(
    points: np.ndarray,
    conormals: np.ndarray,
    previous_lengths: np.ndarray,
    next_lengths: np.ndarray,
    free_indices: np.ndarray,
    config: BoundaryFairingConfig,
):
    count = len(points)
    free_column = {int(vertex): column for column, vertex in enumerate(free_indices)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    target = np.zeros(count * 3, dtype=np.float64)
    fairness_scale = max(float(config.radius_mm), _EPSILON) ** 2

    for index in range(count):
        if index not in free_column:
            continue
        previous = (index - 1) % count
        following = (index + 1) % count
        h_previous = max(float(previous_lengths[index]), _EPSILON)
        h_next = max(float(next_lengths[index]), _EPSILON)
        previous_coefficient = 2.0 / (h_previous * (h_previous + h_next))
        next_coefficient = 2.0 / (h_next * (h_previous + h_next))
        center_coefficient = -(previous_coefficient + next_coefficient)
        coefficients = (
            (previous, previous_coefficient),
            (index, center_coefficient),
            (following, next_coefficient),
        )
        base_curvature = sum(coefficient * points[vertex] for vertex, coefficient in coefficients)
        for axis in range(3):
            row = index * 3 + axis
            target[row] = -fairness_scale * float(base_curvature[axis])
            for vertex, coefficient in coefficients:
                column = free_column.get(int(vertex))
                if column is not None:
                    rows.append(row)
                    columns.append(column)
                    values.append(fairness_scale * coefficient * float(conormals[vertex, axis]))

    matrix = coo_matrix((values, (rows, columns)), shape=(count * 3, len(free_indices))).tocsr()
    fidelity = math.sqrt(max(float(config.fidelity_weight), _EPSILON))
    matrix = vstack((matrix, eye(len(free_indices), format="csr") * fidelity), format="csr")
    target = np.concatenate((target, np.zeros(len(free_indices), dtype=np.float64)))
    return matrix, target


def _add_centroid_constraint(
    matrix,
    target: np.ndarray,
    conormals: np.ndarray,
    free_indices: np.ndarray,
):
    """Prevent rigid drift without preserving high-frequency noisy perimeter."""
    centroid = conormals[free_indices].T / max(len(conormals), 1)
    matrix = vstack((matrix, coo_matrix(centroid * 1_000.0).tocsr()), format="csr")
    target = np.concatenate((target, np.zeros(3, dtype=np.float64)))
    return matrix, target


def _add_extent_anchors(points: np.ndarray, locked: np.ndarray) -> np.ndarray:
    """Pin broad loop extents when no semantic corners anchor the outline."""
    if int(np.count_nonzero(locked)) >= 3:
        return locked
    centered = points - points.mean(axis=0)
    _u, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    for axis, singular_value in zip(axes[:2], singular_values[:2]):
        if float(singular_value) <= _EPSILON:
            continue
        projection = centered @ axis
        locked[int(np.argmin(projection))] = True
        locked[int(np.argmax(projection))] = True
    return locked


def _loop_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum())


def fair_boundary_loop(
    points: np.ndarray,
    surface_normals: np.ndarray,
    source_vertex_ids: np.ndarray,
    config: BoundaryFairingConfig,
) -> tuple[np.ndarray, dict]:
    original = np.asarray(points, dtype=np.float64)
    if config.mode == "off" or len(original) < 4:
        return original.copy(), {"mode": config.mode, "status": "unchanged", "vertices": int(len(original))}
    if config.mode == "taubin":
        result = taubin_smooth_loop(
            original,
            config.legacy_iterations,
            config.legacy_lambda,
            config.legacy_mu,
        )
        displacement = np.linalg.norm(result - original, axis=1)
        return result, {
            "mode": "taubin",
            "status": "legacy_compatibility",
            "vertices": int(len(original)),
            "iterations": int(config.legacy_iterations),
            "maximum_displacement_mm": float(displacement.max(initial=0.0)),
        }

    order = _canonical_order(np.asarray(source_vertex_ids, dtype=np.int64))
    canonical_points = original[order]
    canonical_normals = np.asarray(surface_normals, dtype=np.float64)[order]
    geometry = _loop_geometry(canonical_points, canonical_normals)
    locked = (~geometry["valid"]) | (
        geometry["turning_degrees"] >= float(config.feature_angle_degrees)
    )
    locked = _add_extent_anchors(canonical_points, locked)
    free_indices = np.flatnonzero(~locked)
    if len(free_indices) == 0 or config.max_displacement_mm <= 0.0:
        return original.copy(), {
            "mode": "constrained-arc-length",
            "status": "all_features_locked",
            "vertices": int(len(original)),
            "locked_feature_vertices": int(np.count_nonzero(locked)),
        }

    matrix, target = _fairness_system(
        canonical_points,
        geometry["conormals"],
        geometry["previous_lengths"],
        geometry["next_lengths"],
        free_indices,
        config,
    )
    matrix, target = _add_centroid_constraint(
        matrix,
        target,
        geometry["conormals"],
        free_indices,
    )
    limit = max(float(config.max_displacement_mm), 0.0)
    solution = lsq_linear(matrix, target, bounds=(-limit, limit), lsmr_tol="auto", max_iter=100)
    if not solution.success:
        return original.copy(), {
            "mode": "constrained-arc-length",
            "status": "solver_failed",
            "solver_message": str(solution.message),
            "vertices": int(len(original)),
        }

    canonical_result = canonical_points.copy()
    canonical_result[free_indices] += (
        solution.x[:, None] * geometry["conormals"][free_indices]
    )
    result = np.empty_like(canonical_result)
    result[order] = canonical_result
    displacement = np.linalg.norm(result - original, axis=1)
    length_before = _loop_length(original)
    length_after = _loop_length(result)
    return result, {
        "mode": "constrained-arc-length",
        "status": "solved",
        "vertices": int(len(original)),
        "movable_vertices": int(len(free_indices)),
        "locked_feature_vertices": int(np.count_nonzero(locked)),
        "feature_angle_degrees": float(config.feature_angle_degrees),
        "fairing_radius_mm": float(config.radius_mm),
        "displacement_limit_mm": limit,
        "maximum_displacement_mm": float(displacement.max(initial=0.0)),
        "rms_displacement_mm": float(np.sqrt(np.mean(displacement * displacement))),
        "boundary_length_before_mm": length_before,
        "boundary_length_after_mm": length_after,
        "boundary_length_change_ratio": float(
            (length_after - length_before) / max(length_before, _EPSILON)
        ),
        "vertices_at_displacement_limit": int(np.count_nonzero(displacement >= limit - 1e-8)),
        "solver_iterations": int(solution.nit),
        "solver_cost": float(solution.cost),
    }


def fair_local_boundary_loops(
    local_vertices: np.ndarray,
    loops: list[list[int]],
    global_vertex_ids: np.ndarray,
    context: BoundaryFairingContext,
) -> tuple[np.ndarray, list[dict]]:
    result = np.asarray(local_vertices, dtype=np.float64).copy()
    records: list[dict] = []
    for loop_index, loop in enumerate(loops):
        local_indices = np.asarray(loop, dtype=np.int64)
        source_ids = np.asarray(global_vertex_ids, dtype=np.int64)[local_indices]
        smoothed, record = fair_boundary_loop(
            result[local_indices],
            np.asarray(context.source_surface_normals, dtype=np.float64)[source_ids],
            source_ids,
            context.config,
        )
        maximum_displacement = float(record.get("maximum_displacement_mm", 0.0))
        if (
            context.config.mode == "constrained"
            and maximum_displacement > float(context.config.max_displacement_mm) + 1e-8
        ):
            raise ValueError("Boundary fairing exceeded its hard displacement limit")
        result[local_indices] = smoothed
        records.append({"loop_index": int(loop_index), **record})
    return result, records


class BoundaryFairingService:
    """Apply deterministic, source-id-aligned fairing to cut boundary loops."""

    fair_loop = staticmethod(fair_boundary_loop)
    fair_local_loops = staticmethod(fair_local_boundary_loops)
