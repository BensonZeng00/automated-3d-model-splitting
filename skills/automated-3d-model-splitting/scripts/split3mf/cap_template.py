from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from .cap_backoff import measured_backoff


@dataclass(frozen=True)
class AffineCapBackoffResult:
    template_points: np.ndarray
    boundary_points: np.ndarray
    distances: np.ndarray
    directions: np.ndarray
    thickness_record: dict
    attempts: int
    total_backoff_mm: float
    trace: tuple[dict, ...]


PatchDeformer = Callable[
    [np.ndarray, np.ndarray, tuple[int, ...], np.ndarray],
    tuple[np.ndarray, dict],
]


def _projection_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = np.asarray(normal, dtype=np.float64)
    unit /= max(float(np.linalg.norm(unit)), 1e-12)
    seed = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(seed, unit))) > 0.85:
        seed = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    first = np.cross(unit, seed)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = np.cross(unit, first)
    second /= max(float(np.linalg.norm(second)), 1e-12)
    return first, second


def refined_harmonic_heightfield_cap(
    *,
    boundary_points: np.ndarray,
    initial_faces: np.ndarray | list[tuple[int, int, int]],
    reference_normal: np.ndarray,
    maximum_projected_edge_mm: float = 1.0,
    maximum_refinement_rounds: int = 12,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a short-edge curved cap while preserving its rim exactly.

    The supplied faces must already be a valid non-crossing triangulation of
    the projected rim.  Only interior edges are refined, so the connector rim
    remains byte-for-byte compatible with its side wall.  A scalar harmonic
    solve then smooths height over the fixed valid 2-D topology.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    boundary = np.asarray(boundary_points, dtype=np.float64)
    faces = np.asarray(initial_faces, dtype=np.int64)
    normal = np.asarray(reference_normal, dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    if boundary.ndim != 2 or boundary.shape[1] != 3 or len(boundary) < 3:
        raise ValueError("heightfield cap boundary must be an Nx3 ring")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("heightfield cap requires an initial triangulation")
    if float(maximum_projected_edge_mm) <= 0.0:
        raise ValueError("heightfield cap edge limit must be positive")

    origin = boundary.mean(axis=0)
    axis_u, axis_v = _projection_basis(normal)
    centered = boundary - origin
    projected: list[np.ndarray] = [
        np.asarray([np.dot(point, axis_u), np.dot(point, axis_v)])
        for point in centered
    ]
    boundary_edges = {
        tuple(sorted((index, (index + 1) % len(boundary))))
        for index in range(len(boundary))
    }
    refined_faces = [tuple(int(value) for value in face) for face in faces]
    rounds = 0
    inserted_vertices = 0

    for round_index in range(int(maximum_refinement_rounds)):
        marked_edges: set[tuple[int, int]] = set()
        for face in refined_faces:
            for first, second in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            ):
                edge = tuple(sorted((int(first), int(second))))
                if edge in boundary_edges:
                    continue
                length = float(
                    np.linalg.norm(projected[edge[0]] - projected[edge[1]])
                )
                if length > float(maximum_projected_edge_mm) + 1e-9:
                    marked_edges.add(edge)
        if not marked_edges:
            break

        midpoint_ids: dict[tuple[int, int], int] = {}
        for edge in sorted(marked_edges):
            midpoint_ids[edge] = len(projected)
            projected.append(
                0.5 * (projected[edge[0]] + projected[edge[1]])
            )
        inserted_vertices += len(midpoint_ids)
        next_faces: list[tuple[int, int, int]] = []
        for a, b, c in refined_faces:
            m_ab = midpoint_ids.get(tuple(sorted((a, b))))
            m_bc = midpoint_ids.get(tuple(sorted((b, c))))
            m_ca = midpoint_ids.get(tuple(sorted((c, a))))
            marked = int(m_ab is not None) + int(m_bc is not None) + int(
                m_ca is not None
            )
            if marked == 0:
                next_faces.append((a, b, c))
            elif marked == 1:
                if m_ab is not None:
                    next_faces.extend(((a, m_ab, c), (m_ab, b, c)))
                elif m_bc is not None:
                    next_faces.extend(((b, m_bc, a), (m_bc, c, a)))
                else:
                    next_faces.extend(((c, m_ca, b), (m_ca, a, b)))
            elif marked == 2:
                if m_ab is not None and m_bc is not None:
                    next_faces.extend(
                        ((b, m_bc, m_ab), (a, m_ab, c), (m_ab, m_bc, c))
                    )
                elif m_bc is not None and m_ca is not None:
                    next_faces.extend(
                        ((c, m_ca, m_bc), (b, m_bc, a), (m_bc, m_ca, a))
                    )
                else:
                    next_faces.extend(
                        ((a, m_ab, m_ca), (c, m_ca, b), (m_ca, m_ab, b))
                    )
            else:
                next_faces.extend(
                    (
                        (a, m_ab, m_ca),
                        (m_ab, b, m_bc),
                        (m_ca, m_bc, c),
                        (m_ab, m_bc, m_ca),
                    )
                )
        refined_faces = next_faces
        rounds = round_index + 1

    projected_array = np.asarray(projected, dtype=np.float64)
    refined_face_array = np.asarray(refined_faces, dtype=np.int64)
    first_edges = (
        projected_array[refined_face_array[:, 1]]
        - projected_array[refined_face_array[:, 0]]
    )
    second_edges = (
        projected_array[refined_face_array[:, 2]]
        - projected_array[refined_face_array[:, 0]]
    )
    signed_area = (
        first_edges[:, 0] * second_edges[:, 1]
        - first_edges[:, 1] * second_edges[:, 0]
    )
    negative = signed_area < 0.0
    if np.any(negative):
        reversed_vertices = refined_face_array[negative][:, [0, 2, 1]]
        refined_face_array[negative] = reversed_vertices
        signed_area[negative] = -signed_area[negative]

    vertex_count = len(projected_array)
    boundary_count = len(boundary)
    neighbors: list[set[int]] = [set() for _ in range(vertex_count)]
    for a, b, c in refined_face_array:
        a, b, c = int(a), int(b), int(c)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    interior = np.arange(boundary_count, vertex_count, dtype=np.int64)
    heights = np.zeros(vertex_count, dtype=np.float64)
    heights[:boundary_count] = centered @ normal
    if len(interior):
        lookup = np.full(vertex_count, -1, dtype=np.int64)
        lookup[interior] = np.arange(len(interior), dtype=np.int64)
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        rhs = np.zeros(len(interior), dtype=np.float64)
        for row, vertex_id in enumerate(interior):
            adjacent = neighbors[int(vertex_id)]
            rows.append(row)
            columns.append(row)
            values.append(float(len(adjacent)))
            for neighbor_id in adjacent:
                if int(neighbor_id) < boundary_count:
                    rhs[row] += heights[int(neighbor_id)]
                else:
                    rows.append(row)
                    columns.append(int(lookup[int(neighbor_id)]))
                    values.append(-1.0)
        matrix = coo_matrix(
            (values, (rows, columns)),
            shape=(len(interior), len(interior)),
        ).tocsr()
        heights[interior] = spsolve(matrix, rhs)

    points = (
        origin[None, :]
        + projected_array[:, 0, None] * axis_u[None, :]
        + projected_array[:, 1, None] * axis_v[None, :]
        + heights[:, None] * normal[None, :]
    )
    points[:boundary_count] = boundary
    triangles = points[refined_face_array]
    triangle_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    doubled_areas = np.linalg.norm(triangle_normals, axis=1)
    normal_cosines = (triangle_normals @ normal) / np.maximum(
        doubled_areas, 1e-15
    )
    edge_lengths = np.stack(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    boundary_error = float(
        np.linalg.norm(points[:boundary_count] - boundary, axis=1).max()
    )
    degenerate_faces = int(np.count_nonzero(doubled_areas <= 1e-12))
    reversed_faces = int(np.count_nonzero(normal_cosines <= 1e-6))
    steep_faces = int(np.count_nonzero(normal_cosines <= 0.05))
    maximum_edge = float(edge_lengths.max())
    boundary_edge_lengths = np.linalg.norm(
        np.roll(boundary, -1, axis=0) - boundary,
        axis=1,
    )
    edge_limit = max(2.5, 4.0 * float(boundary_edge_lengths.max()))
    valid = bool(
        boundary_error <= 1e-9
        and degenerate_faces == 0
        and reversed_faces == 0
        and maximum_edge <= edge_limit + 1e-9
    )
    quality = {
        "valid": valid,
        "strategy": "refined_harmonic_heightfield",
        "vertex_count": int(len(points)),
        "face_count": int(len(refined_face_array)),
        "boundary_vertex_count": int(boundary_count),
        "inserted_vertex_count": int(inserted_vertices),
        "refinement_rounds": int(rounds),
        "boundary_match_error_mm": boundary_error,
        "degenerate_face_count": degenerate_faces,
        "reversed_face_count": reversed_faces,
        "steep_face_count": steep_faces,
        "minimum_reference_normal_cosine": float(normal_cosines.min()),
        "maximum_triangle_edge_mm": maximum_edge,
        "maximum_triangle_edge_limit_mm": edge_limit,
    }
    return points, refined_face_array, quality


def progressive_boundary_deformation(
    *,
    source_points: np.ndarray,
    source_faces: np.ndarray,
    boundary_indices: np.ndarray | tuple[int, ...] | list[int],
    target_boundary_points: np.ndarray,
    deform: PatchDeformer,
    initial_step_fraction: float = 0.25,
    minimum_step_fraction: float = 1.0 / 4096.0,
    maximum_attempts: int = 96,
) -> tuple[np.ndarray, dict]:
    """Reach an exact cap rim through quality-gated harmonic continuation.

    A single large boundary displacement can fold a dense source patch even
    when the final rim itself is valid.  Continuation follows the same linear
    boundary path in adaptive increments, accepting only deformation steps
    that retain the existing degeneration, orientation and edge-length gates.
    """
    points = np.asarray(source_points, dtype=np.float64)
    faces = np.asarray(source_faces, dtype=np.int64)
    boundary = np.asarray(boundary_indices, dtype=np.int64)
    targets = np.asarray(target_boundary_points, dtype=np.float64)
    if targets.shape != (len(boundary), 3):
        raise ValueError("progressive cap target rim does not match boundary ids")
    if not 0.0 < float(initial_step_fraction) <= 1.0:
        raise ValueError("progressive cap initial step must be in (0, 1]")
    if not 0.0 < float(minimum_step_fraction) <= 1.0:
        raise ValueError("progressive cap minimum step must be in (0, 1]")
    if int(maximum_attempts) <= 0:
        raise ValueError("progressive cap requires a positive attempt budget")

    source_boundary = points[boundary].copy()
    current = points.copy()
    progress_fraction = 0.0
    step_fraction = float(initial_step_fraction)
    accepted_steps = 0
    rejected_steps = 0
    trace: list[dict] = []
    minimum_step_cosine = 1.0
    maximum_step_edge_stretch = 1.0

    for attempt in range(1, int(maximum_attempts) + 1):
        if progress_fraction >= 1.0 - 1e-12:
            break
        next_fraction = min(progress_fraction + step_fraction, 1.0)
        intermediate_targets = source_boundary + next_fraction * (
            targets - source_boundary
        )
        candidate, step_quality = deform(
            current,
            faces,
            tuple(int(value) for value in boundary),
            intermediate_targets,
        )
        valid = bool(step_quality.get("valid", False))
        trace.append(
            {
                "attempt": attempt,
                "from_fraction": float(progress_fraction),
                "to_fraction": float(next_fraction),
                "step_fraction": float(next_fraction - progress_fraction),
                "valid": valid,
                "degenerate_face_count": int(
                    step_quality.get("degenerate_face_count", 0)
                ),
                "reversed_face_count": int(
                    step_quality.get("reversed_face_count", 0)
                ),
                "minimum_source_normal_cosine": float(
                    step_quality.get("minimum_source_normal_cosine", 1.0)
                ),
            }
        )
        if not valid:
            rejected_steps += 1
            step_fraction *= 0.5
            if step_fraction < float(minimum_step_fraction) - 1e-15:
                return current, {
                    "valid": False,
                    "strategy": "source_patch_progressive_harmonic_displacement",
                    "progress_fraction": float(progress_fraction),
                    "accepted_step_count": accepted_steps,
                    "rejected_step_count": rejected_steps,
                    "attempt_count": len(trace),
                    "minimum_step_fraction": float(minimum_step_fraction),
                    "failure_reason": "minimum_step_fraction_exhausted",
                    "last_rejected_quality": step_quality,
                    "trace": trace,
                }
            continue

        current = np.asarray(candidate, dtype=np.float64)
        progress_fraction = next_fraction
        accepted_steps += 1
        minimum_step_cosine = min(
            minimum_step_cosine,
            float(step_quality.get("minimum_source_normal_cosine", 1.0)),
        )
        maximum_step_edge_stretch = max(
            maximum_step_edge_stretch,
            float(step_quality.get("maximum_edge_stretch_ratio", 1.0)),
        )
        remaining = 1.0 - progress_fraction
        step_fraction = min(step_fraction * 1.5, remaining)

    boundary_error = float(
        np.linalg.norm(current[boundary] - targets, axis=1).max()
    )
    complete = bool(
        progress_fraction >= 1.0 - 1e-12 and boundary_error <= 1e-9
    )
    return current, {
        "valid": complete,
        "strategy": "source_patch_progressive_harmonic_displacement",
        "vertex_count": int(len(points)),
        "face_count": int(len(faces)),
        "boundary_vertex_count": int(len(boundary)),
        "boundary_match_error_mm": boundary_error,
        "progress_fraction": float(progress_fraction),
        "accepted_step_count": accepted_steps,
        "rejected_step_count": rejected_steps,
        "attempt_count": len(trace),
        "minimum_step_normal_cosine": float(minimum_step_cosine),
        "maximum_step_edge_stretch_ratio": float(maximum_step_edge_stretch),
        "failure_reason": None if complete else "attempt_budget_exhausted",
        "trace": trace,
    }


def fit_affine_cap_inside_parent(
    *,
    template_points: np.ndarray,
    boundary_indices: np.ndarray,
    fit_points: np.ndarray,
    inward_axis: np.ndarray,
    safety_limit: Callable[[np.ndarray, np.ndarray, float], tuple[float, dict]],
    maximum_depth_mm: float,
    minimum_depth_mm: float = 0.08,
    maximum_attempts: int = 24,
) -> AffineCapBackoffResult:
    """Move only a generated affine cap outward until it fits its parent.

    The source surface is immutable. Solve a bounded translation against the
    measured depth, then authoritatively remeasure its new ray directions.
    Fall back to half-headroom only when the measured interval is infeasible.
    """
    points = np.asarray(template_points, dtype=np.float64).copy()
    boundary_indices = np.asarray(boundary_indices, dtype=np.int64)
    fit_points = np.asarray(fit_points, dtype=np.float64)
    axis = np.asarray(inward_axis, dtype=np.float64)
    axis_length = float(np.linalg.norm(axis))
    if axis_length <= 1e-12:
        raise ValueError("affine source-patch cap has a zero inward axis")
    axis = axis / axis_length
    if maximum_attempts <= 0:
        raise ValueError("affine source-patch cap requires a positive attempt budget")

    trace: list[dict] = []
    total_backoff = 0.0
    last_state: tuple[np.ndarray, np.ndarray, np.ndarray, float, dict] | None = None
    # A backoff changes ray directions but not their origins or maximum finite
    # length.  Reuse only a conservative direction-independent broad phase;
    # every attempt still performs its own capsule rejection, exact triangle
    # intersections, and complete safety-limit policy.
    probe = getattr(safety_limit, "__self__", None)
    prepare_candidates = getattr(probe, "prepare_safety_limit_candidates", None)
    broad_phase_cache = (
        prepare_candidates(fit_points, float(maximum_depth_mm))
        if callable(prepare_candidates)
        else None
    )
    for attempt in range(1, int(maximum_attempts) + 1):
        boundary = points[boundary_indices]
        displacement = boundary - fit_points
        distances = np.linalg.norm(displacement, axis=1)
        minimum_distance = float(distances.min())
        if minimum_distance <= float(minimum_depth_mm) - 1e-9:
            raise ValueError(
                "affine source-patch cap has insufficient positive depth: "
                f"attempt={attempt}, minimum_depth_mm={minimum_distance:.6f}, "
                f"required_mm={float(minimum_depth_mm):.6f}, "
                f"total_backoff_mm={total_backoff:.6f}"
            )
        directions = displacement / distances[:, None]
        if broad_phase_cache is None:
            safe_maximum, thickness_record = safety_limit(
                fit_points,
                directions,
                float(maximum_depth_mm),
            )
        else:
            safe_maximum, thickness_record = probe.safety_limit(
                fit_points,
                directions,
                float(maximum_depth_mm),
                broad_phase_cache=broad_phase_cache,
            )
        maximum_distance = float(distances.max())
        excess = maximum_distance - float(safe_maximum)
        trace.append(
            {
                "attempt": attempt,
                "minimum_depth_mm": minimum_distance,
                "maximum_depth_mm": maximum_distance,
                "safe_maximum_depth_mm": float(safe_maximum),
                "excess_depth_mm": float(excess),
                "total_backoff_mm": float(total_backoff),
            }
        )
        last_state = (
            boundary,
            distances,
            directions,
            float(safe_maximum),
            dict(thickness_record),
        )
        if excess <= 1e-7:
            return AffineCapBackoffResult(
                template_points=points,
                boundary_points=boundary,
                distances=distances,
                directions=directions,
                thickness_record=dict(thickness_record),
                attempts=attempt,
                total_backoff_mm=float(total_backoff),
                trace=tuple(trace),
            )

        applied_backoff, method = measured_backoff(
            displacement, axis, float(safe_maximum), float(minimum_depth_mm), excess)
        trace[-1]["backoff_method"] = method
        trace[-1]["applied_backoff_mm"] = float(applied_backoff)
        if applied_backoff <= 1e-7:
            break
        points -= axis[None, :] * applied_backoff
        total_backoff += applied_backoff

    if last_state is None:
        raise ValueError("affine source-patch cap backoff produced no measurement")
    _boundary, distances, _directions, safe_maximum, _record = last_state
    raise ValueError(
        "affine source-patch cap could not fit inside parent thickness: "
        f"attempts={len(trace)}, minimum_depth_mm={float(distances.min()):.6f}, "
        f"maximum_depth_mm={float(distances.max()):.6f}, "
        f"safe_maximum_mm={safe_maximum:.6f}, "
        f"excess_mm={float(distances.max()) - safe_maximum:.6f}, "
        f"total_backoff_mm={total_backoff:.6f}"
    )
