"""Topology primitives dedicated to local connector interfaces.

The painted-mesh builder owns source geometry and colors.  This module owns
only two-dimensional connector topology: polygon-with-hole triangulation and
auditing.  Keeping it independent from :mod:`inward` makes the same algorithm
available to production builders, unit tests, and the synthetic harness.
"""

from __future__ import annotations

import collections
import time
from dataclasses import asdict, dataclass, replace

from .common import *
from .mesh import (
    point_in_poly,
    point_on_poly_boundary,
    segments_intersect_2d_strict,
    signed_area,
    triangulate_polygon_ear_clip,
)
from .reporting import runtime_log
from .annulus_projection import audit_projection


# The visibility-path solver is quadratic in ring size, but 1024 x 1024 is
# still comfortably bounded and covers production contours whose trimmed
# inset crosses a medial-axis event.  Larger vendor rings retain the linear
# ranked-seam and constrained-annulus paths.
MAXIMUM_VISIBILITY_PATH_RING_VERTICES = 1024

# Large vendor-painted contours can contain thousands of points.  The
# constrained polygon solver and the explicit visible-bridge fallback both
# become expensive at that scale, while the balanced strip and star-shaped
# zipper remain close to linear.  Their output is still accepted only after
# the same strict topology/geometry audits used by the slower solvers.
LARGE_ANNULUS_FAST_PATH_VERTICES = 1024
DENSE_VISIBLE_BRIDGE_FALLBACK_VERTICES = 256
USER_REVIEWED_MAXIMUM_HIDDEN_FANOUT = 64
USER_REVIEWED_MAXIMUM_HIDDEN_CROSS_EDGE_MM = 8.0


@dataclass(frozen=True)
class PatchTopologyAudit:
    """Immutable audit result for one triangulated annulus."""

    valid: bool
    face_count: int
    boundary_edge_count: int
    open_edge_count: int
    over_shared_edge_count: int
    degenerate_face_count: int
    covered_area: float
    expected_area: float
    reason: str = ""
    strategy: str = ""


@dataclass(frozen=True)
class RingStripAudit:
    """Quality evidence for one side strip between two closed rings."""

    valid: bool
    face_count: int
    boundary_edge_count: int
    open_edge_count: int
    over_shared_edge_count: int
    degenerate_face_count: int
    maximum_fanout: int
    maximum_cross_edge_mm: float
    median_cross_edge_mm: float
    invalid_bridge_count: int
    reason: str = ""
    strategy: str = "balanced_arc_length_bounded_fanout"
    worst_bridge_id: tuple[int, int] | None = None
    planar_transition_long_bridge_accepted: bool = False
    projection_bridge_passthrough_accepted: bool = False
    user_reviewed_broad_fanout_accepted: bool = False
    projection: dict | None = None


def _user_reviewed_broad_fanout_is_bounded(
    *,
    enabled: bool,
    maximum_fanout: int,
    maximum_cross_edge_mm: float,
    invalid_bridge_count: int,
) -> bool:
    """Bound a visually approved hidden fan without weakening topology."""

    return bool(
        enabled
        and int(invalid_bridge_count) == 0
        and int(maximum_fanout) <= USER_REVIEWED_MAXIMUM_HIDDEN_FANOUT
        and float(maximum_cross_edge_mm)
        <= USER_REVIEWED_MAXIMUM_HIDDEN_CROSS_EDGE_MM + 1e-12
    )


@dataclass(frozen=True)
class RingStripResult:
    """Faces plus the possibly locally refined second boundary ring."""

    faces: tuple[tuple[int, int, int], ...]
    audit: RingStripAudit
    inner_ids: tuple[int, ...]
    inner_points: np.ndarray


def _refine_ring_to_minimum_count(
    ids: list[int],
    points: np.ndarray,
    projected: np.ndarray,
    *,
    minimum_count: int,
    next_id: int,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Subdivide longest boundary segments without changing the polygon."""

    ids = [int(value) for value in ids]
    points = np.asarray(points, dtype=np.float64)
    projected = np.asarray(projected, dtype=np.float64)
    target = max(len(ids), int(minimum_count))
    if target == len(ids):
        return ids, points, projected
    lengths = np.linalg.norm(np.roll(projected, -1, axis=0) - projected, axis=1)
    subdivisions = np.ones(len(ids), dtype=np.int64)
    for _ in range(target - len(ids)):
        edge_index = int(np.argmax(lengths / subdivisions))
        subdivisions[edge_index] += 1
    refined_ids: list[int] = []
    refined_points: list[np.ndarray] = []
    refined_projected: list[np.ndarray] = []
    generated_id = int(next_id)
    for index, vertex_id in enumerate(ids):
        following = (index + 1) % len(ids)
        segment_count = int(subdivisions[index])
        refined_ids.append(int(vertex_id))
        refined_points.append(points[index])
        refined_projected.append(projected[index])
        for segment_index in range(1, segment_count):
            ratio = segment_index / segment_count
            refined_ids.append(int(generated_id))
            refined_points.append(
                points[index] * (1.0 - ratio) + points[following] * ratio
            )
            refined_projected.append(
                projected[index] * (1.0 - ratio) + projected[following] * ratio
            )
            generated_id += 1
    return (
        refined_ids,
        np.asarray(refined_points, dtype=np.float64),
        np.asarray(refined_projected, dtype=np.float64),
    )


def _refine_ring_to_reference_parameter_grid(
    ids: list[int],
    points: np.ndarray,
    projected: np.ndarray,
    reference_projected: np.ndarray,
    *,
    next_id: int,
    seam_rank: int = 0,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Subdivide a sparse ring onto a dense ring's cyclic parameter grid.

    Every original sparse-ring vertex is retained.  Additional vertices are
    interpolated only along its existing straight edges, while their cyclic
    indices follow the reference ring's normalized perimeter.  Adjacent
    backing layers can therefore use deterministic one-to-one strips without
    rounding off Clipper corners or inventing a new contour.
    """

    ids = [int(value) for value in ids]
    points = np.asarray(points, dtype=np.float64)
    projected = np.asarray(projected, dtype=np.float64)
    reference = np.asarray(reference_projected, dtype=np.float64)
    target_count = int(len(reference))
    if target_count < len(ids):
        raise ValueError("reference parameter grid is smaller than source ring")
    if target_count == len(ids):
        return ids, points, projected

    def parameter_assignment(ring: np.ndarray) -> np.ndarray:
        source_parameters = _normalized_ring_parameters(ring)[:-1]
        assigned = np.zeros(len(ring), dtype=np.int64)
        previous = 0
        for index in range(1, len(ring)):
            remaining = len(ring) - index
            ideal = int(np.rint(float(source_parameters[index]) * target_count))
            current = max(previous + 1, min(ideal, target_count - remaining))
            assigned[index] = int(current)
            previous = int(current)
        return assigned

    def sample_projected(ring: np.ndarray, assigned: np.ndarray) -> np.ndarray:
        sampled: list[np.ndarray] = []
        for index in range(len(ring)):
            following = (index + 1) % len(ring)
            start = int(assigned[index])
            end = int(assigned[index + 1]) if following else target_count
            segment_count = end - start
            for segment_index in range(segment_count):
                ratio = segment_index / segment_count
                sampled.append(
                    ring[index] * (1.0 - ratio) + ring[following] * ratio
                )
        return np.asarray(sampled, dtype=np.float64)

    # A nearest point is not a reliable cyclic anchor on a deep concavity: it
    # can pair opposite banks of the notch.  Evaluate every existing sparse
    # vertex as a seam and choose the phase whose complete parameter-grid
    # correspondence has the least bridge energy.  This is linear in the
    # dense-grid size for each sparse candidate and runs only when a ring has
    # already simplified by more than 1.5x.
    ranked_seams: list[tuple[tuple[float, float, int], int]] = []
    for seam in range(len(ids)):
        rotated_projected = np.concatenate(
            (projected[seam:], projected[:seam]), axis=0
        )
        candidate_assignment = parameter_assignment(rotated_projected)
        candidate = sample_projected(rotated_projected, candidate_assignment)
        squared = np.einsum("ij,ij->i", candidate - reference, candidate - reference)
        score = (float(np.mean(squared)), float(np.max(squared)), int(seam))
        ranked_seams.append((score, int(seam)))

    ranked_seams.sort(key=lambda item: item[0])
    selected_rank = min(max(int(seam_rank), 0), len(ranked_seams) - 1)
    best_seam = int(ranked_seams[selected_rank][1])

    ids = ids[best_seam:] + ids[:best_seam]
    points = np.concatenate((points[best_seam:], points[:best_seam]), axis=0)
    projected = np.concatenate(
        (projected[best_seam:], projected[:best_seam]), axis=0
    )
    assigned = parameter_assignment(projected)

    refined_ids: list[int] = []
    refined_points: list[np.ndarray] = []
    refined_projected: list[np.ndarray] = []
    generated_id = int(next_id)
    for index, vertex_id in enumerate(ids):
        following = (index + 1) % len(ids)
        start = int(assigned[index])
        end = int(assigned[index + 1]) if following else target_count
        segment_count = end - start
        if segment_count < 1:
            raise ValueError("reference parameter grid produced a collapsed segment")
        for segment_index in range(segment_count):
            ratio = segment_index / segment_count
            refined_ids.append(
                int(vertex_id) if segment_index == 0 else int(generated_id)
            )
            refined_points.append(
                points[index] * (1.0 - ratio) + points[following] * ratio
            )
            refined_projected.append(
                projected[index] * (1.0 - ratio) + projected[following] * ratio
            )
            if segment_index:
                generated_id += 1
    return (
        refined_ids,
        np.asarray(refined_points, dtype=np.float64),
        np.asarray(refined_projected, dtype=np.float64),
    )


def _audit_equal_count_ring_strip(
    outer_ids: list[int],
    outer_points: np.ndarray,
    outer_projected: np.ndarray,
    inner_ids: list[int],
    inner_points: np.ndarray,
    inner_projected: np.ndarray,
    *,
    maximum_fanout: int,
    allow_preconditioned_compact_fan: bool,
    strategy: str,
) -> RingStripResult:
    """Build and strictly audit an index-aligned equal-count ring strip."""

    if len(outer_ids) != len(inner_ids):
        raise ValueError("equal-count ring strip received mismatched boundaries")
    faces: list[tuple[int, int, int]] = []
    for index in range(len(outer_ids)):
        following = (index + 1) % len(outer_ids)
        faces.append(
            (
                int(outer_ids[index]),
                int(outer_ids[following]),
                int(inner_ids[index]),
            )
        )
        faces.append(
            (
                int(outer_ids[following]),
                int(inner_ids[following]),
                int(inner_ids[index]),
            )
        )
    point_by_id = {
        **{int(i): p for i, p in zip(outer_ids, outer_points)},
        **{int(i): p for i, p in zip(inner_ids, inner_points)},
    }
    projected_by_id = {
        **{int(i): p for i, p in zip(outer_ids, outer_projected)},
        **{int(i): p for i, p in zip(inner_ids, inner_projected)},
    }
    audit = _audit_ring_strip(
        faces,
        point_by_id,
        projected_by_id,
        outer_ids,
        inner_ids,
        outer_projected,
        inner_projected,
        maximum_fanout=int(maximum_fanout),
        allow_preconditioned_compact_fan=allow_preconditioned_compact_fan,
    )
    return RingStripResult(
        tuple(faces) if audit.valid else tuple(),
        replace(audit, strategy=strategy),
        tuple(int(value) for value in inner_ids),
        np.asarray(inner_points, dtype=np.float64),
    )


def _ring_edges(ring: list[int]) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(ring[index]), int(ring[(index + 1) % len(ring)]))))
        for index in range(len(ring))
    }


def _normalized_ring_parameters(points: np.ndarray) -> np.ndarray:
    ring = np.asarray(points, dtype=np.float64)
    lengths = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 1e-12:
        raise ValueError("connector strip ring has zero perimeter")
    return np.concatenate(([0.0], np.cumsum(lengths))) / perimeter


def _balanced_ring_strip_faces(
    outer_ids: list[int],
    outer_points: np.ndarray,
    inner_ids: list[int],
    inner_points: np.ndarray,
    *,
    maximum_consecutive_advances: int,
) -> list[tuple[int, int, int]]:
    """Zip two aligned rings by physical arc length with bounded fanout."""

    outer_parameter = _normalized_ring_parameters(outer_points)
    inner_parameter = _normalized_ring_parameters(inner_points)
    n = len(outer_ids)
    m = len(inner_ids)
    i = j = 0
    last_move = -1
    run = 0
    faces: list[tuple[int, int, int]] = []
    while i < n or j < m:
        can_outer = i < n and not (
            last_move == 0 and run >= int(maximum_consecutive_advances)
        )
        can_inner = j < m and not (
            last_move == 1 and run >= int(maximum_consecutive_advances)
        )
        if not can_outer and not can_inner:
            # The consecutive-advance bound is a construction preference, not
            # the final safety decision.  Strongly non-uniform concave rings
            # can reach a state where both locally preferred moves are capped
            # even though a complete strip still exists.  Permit one escape
            # move and let the authoritative topology/bridge audit below
            # reject any genuinely unsafe fanout or long bridge.
            can_outer = i < n
            can_inner = j < m
        if can_outer and not can_inner:
            move = 0
        elif can_inner and not can_outer:
            move = 1
        else:
            # Prefer the advance whose new cross-ring edge is geometrically
            # shorter.  Normalized perimeter alone drifts badly when Clipper
            # trims a concave lobe: the remaining contour has a different arc
            # distribution even though the two rings are locally parallel.
            outer_cross = float(
                np.linalg.norm(outer_points[(i + 1) % n] - inner_points[j % m])
            )
            inner_cross = float(
                np.linalg.norm(outer_points[i % n] - inner_points[(j + 1) % m])
            )
            # A small progress term prevents noisy equal-distance choices from
            # accumulating all count imbalance near the seam.
            outer_progress = abs(float(outer_parameter[i + 1] - inner_parameter[j]))
            inner_progress = abs(float(inner_parameter[j + 1] - outer_parameter[i]))
            move = 0 if (
                outer_cross + 0.05 * outer_progress
                <= inner_cross + 0.05 * inner_progress
            ) else 1
        if move == 0:
            faces.append(
                (
                    int(outer_ids[i % n]),
                    int(outer_ids[(i + 1) % n]),
                    int(inner_ids[j % m]),
                )
            )
            i += 1
        else:
            faces.append(
                (
                    int(outer_ids[i % n]),
                    int(inner_ids[(j + 1) % m]),
                    int(inner_ids[j % m]),
                )
            )
            j += 1
        if move == last_move:
            run += 1
        else:
            last_move = move
            run = 1
    return faces


def _audit_ring_strip(
    faces: list[tuple[int, int, int]],
    point_by_id: dict[int, np.ndarray],
    projected_by_id: dict[int, np.ndarray],
    outer_ids: list[int],
    inner_ids: list[int],
    outer_polygon: np.ndarray,
    inner_polygon: np.ndarray,
    *,
    maximum_fanout: int,
    allow_preconditioned_compact_fan: bool = False,
    allow_projection_bridge_passthrough: bool = False,
    restored_source_ears: tuple[tuple[int, int, int], ...] = (),
    projection_core_outer_ids: list[int] | None = None,
) -> RingStripAudit:
    required_edges = _ring_edges(outer_ids) | _ring_edges(inner_ids)
    outer_set = set(int(value) for value in outer_ids)
    inner_set = set(int(value) for value in inner_ids)
    edge_counts: collections.Counter[tuple[int, int]] = collections.Counter()
    cross_edges: set[tuple[int, int]] = set()
    degenerate = 0
    for face in faces:
        triangle = np.asarray([point_by_id[int(value)] for value in face])
        double_area = float(
            np.linalg.norm(
                np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            )
        )
        if not np.isfinite(double_area) or double_area <= 1e-12:
            degenerate += 1
        for left, right in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(left), int(right))))
            edge_counts[edge] += 1
            if (int(left) in outer_set) != (int(right) in outer_set):
                cross_edges.add(edge)
    open_edges = {edge for edge, count in edge_counts.items() if count == 1}
    over_shared = int(sum(count > 2 for count in edge_counts.values()))
    fanout: collections.Counter[int] = collections.Counter()
    cross_lengths: list[float] = []
    invalid_bridges = 0
    bridge_samples: list[np.ndarray] = []
    bridge_edges: list[tuple[int, int]] = []
    for left, right in cross_edges:
        fanout[int(left)] += 1
        fanout[int(right)] += 1
        first = np.asarray(point_by_id[int(left)], dtype=np.float64)
        second = np.asarray(point_by_id[int(right)], dtype=np.float64)
        cross_lengths.append(float(np.linalg.norm(second - first)))
        projected_first = np.asarray(projected_by_id[int(left)], dtype=np.float64)
        projected_second = np.asarray(projected_by_id[int(right)], dtype=np.float64)
        bridge_edges.append((int(left), int(right)))
        bridge_samples.extend(
            projected_first * (1.0 - ratio) + projected_second * ratio
            for ratio in (0.25, 0.50, 0.75)
        )
    excluded = {tuple(sorted(face)) for face in restored_source_ears}
    projected_faces = [face for face in faces if tuple(sorted(face)) not in excluded]
    projection = audit_projection(
        projected_faces, projected_by_id,
        projection_core_outer_ids if projection_core_outer_ids is not None else outer_ids,
        inner_ids)
    projection_blocks = bool(
        not projection.valid and not allow_projection_bridge_passthrough
    )
    # A strict projection failure already rejects this candidate.  Avoid an
    # O(cross_edges * polygon_edges) containment scan that cannot change that
    # decision; the caller may still repair the small projected ears and audit
    # the restored candidate on its next pass.
    skip_bridge_containment = bool(
        projection_blocks
        and max(len(outer_ids), len(inner_ids))
        >= LARGE_ANNULUS_FAST_PATH_VERTICES
    )
    if bridge_samples and not skip_bridge_containment:
        samples = np.asarray(bridge_samples, dtype=np.float64)

        def points_inside_polygon(query: np.ndarray, polygon: np.ndarray) -> np.ndarray:
            """Use Matplotlib's compiled path scan for dense contour audits.

            Material-expanded boundaries can contain tens of thousands of
            segments and produce three samples for every cross-ring edge.  A
            NumPy query-by-segment matrix is bounded in memory but still does
            O(query * segment) Python-level chunk dispatch and took minutes on
            the reviewed Yoshi interface.  ``Path.contains_points`` performs
            the same even/odd scan in compiled code without building that
            matrix.
            """

            from matplotlib.path import Path

            contour = np.asarray(polygon, dtype=np.float64)
            closed = np.vstack((contour, contour[0]))
            return np.asarray(
                Path(closed, closed=True).contains_points(
                    np.asarray(query, dtype=np.float64)
                ),
                dtype=bool,
            )

        def points_on_polygon_boundary(
            query: np.ndarray,
            polygon: np.ndarray,
            *,
            tolerance: float = 1e-8,
        ) -> np.ndarray:
            """Return boundary ownership without an all-points/all-edges matrix."""

            from matplotlib.path import Path

            contour = np.asarray(polygon, dtype=np.float64)
            closed = Path(np.vstack((contour, contour[0])), closed=True)
            values = np.asarray(query, dtype=np.float64)
            expanded = closed.contains_points(values, radius=2.0 * float(tolerance))
            contracted = closed.contains_points(values, radius=-2.0 * float(tolerance))
            return np.asarray(expanded != contracted, dtype=bool)

        inside_outer = points_inside_polygon(samples, outer_polygon)
        inside_inner = points_inside_polygon(samples, inner_polygon)
        # Boundary ownership can only rescue an otherwise failing containment
        # decision.  Dense valid strips normally have every sample strictly
        # inside the outer loop and strictly outside the inner loop, so running
        # an all-points/all-segments distance matrix here wastes most of the
        # strict audit time without changing one result.
        on_outer = np.zeros(len(samples), dtype=bool)
        outer_suspects = np.flatnonzero(~inside_outer)
        if len(outer_suspects):
            on_outer[outer_suspects] = points_on_polygon_boundary(
                samples[outer_suspects],
                outer_polygon,
            )
        on_inner = np.zeros(len(samples), dtype=bool)
        inner_suspects = np.flatnonzero(inside_inner)
        if len(inner_suspects):
            on_inner[inner_suspects] = points_on_polygon_boundary(
                samples[inner_suspects],
                inner_polygon,
            )
        invalid_bridge_mask = np.any(
            (
                (~inside_outer & ~on_outer)
                | (inside_inner & ~on_inner)
            ).reshape((-1, 3)),
            axis=1,
        )
        invalid_bridges = int(np.count_nonzero(invalid_bridge_mask))
    else:
        invalid_bridge_mask = np.zeros(0, dtype=bool)
    measured_fanout = max(fanout.values(), default=0)
    maximum_cross = max(cross_lengths, default=0.0)
    median_cross = float(np.median(cross_lengths)) if cross_lengths else 0.0
    excessive_cross = bool(
        cross_lengths
        and maximum_cross > max(5.0, 10.0 * max(median_cross, 1e-12))
    )
    reasons: list[str] = []
    if open_edges != required_edges:
        reasons.append("boundary_mismatch")
    if over_shared:
        reasons.append("over_shared_edges")
    if degenerate:
        reasons.append("degenerate_faces")
    # One balanced advance can add a bridge on each side of a vertex, so the
    # ordinary audited ceiling is twice the construction run limit.  Dense
    # vendor contours can also produce a compact corner fan whose every bridge
    # is sub-millimetre, locally uniform, and fully inside the annulus.  That is
    # ordinary tessellation rather than the long visible knife geometry this
    # audit is designed to reject.  Keep the fan only when all three geometric
    # protections hold; otherwise force local refinement/failure.
    audited_fanout_ceiling = 2 * int(maximum_fanout)
    dense_compact_fan = bool(
        max(len(outer_ids), len(inner_ids)) > 512
        and bool(cross_lengths)
        and maximum_cross <= 1.25 + 1e-12
        and maximum_cross <= 3.0 * max(median_cross, 1e-12)
    )
    preconditioned_compact_fan = bool(
        allow_preconditioned_compact_fan
        and measured_fanout <= 4 * int(maximum_fanout)
        and bool(cross_lengths)
        and maximum_cross <= 1.5 + 1e-12
        and maximum_cross <= 3.25 * max(median_cross, 1e-12)
    )
    # Fanout is a sampling/topology statistic, not a defect by itself.  At a
    # real medial-axis event a valid constrained annulus can legitimately fan
    # many dense outer samples into one surviving inner corner.  Block it only
    # when geometry also shows an outlier bridge or an annulus escape.  The
    # caller refines legal broad fans into a smooth continuous height field.
    compact_fan_geometry = bool(
        invalid_bridges == 0
        and (
            not excessive_cross
            or dense_compact_fan
            or preconditioned_compact_fan
        )
    )
    user_reviewed_broad_fanout_accepted = _user_reviewed_broad_fanout_is_bounded(
        enabled=allow_projection_bridge_passthrough,
        maximum_fanout=measured_fanout,
        maximum_cross_edge_mm=maximum_cross,
        invalid_bridge_count=invalid_bridges,
    )
    planar_transition_long_bridge_accepted = False
    if excessive_cross and invalid_bridges == 0 and not skip_bridge_containment:
        long_threshold = max(5.0, 10.0 * max(median_cross, 1e-12))
        long_edges = [
            edge
            for edge, length in zip(bridge_edges, cross_lengths)
            if float(length) > long_threshold
        ]
        dense_bridge_samples: list[np.ndarray] = []
        for left, right in long_edges:
            first = np.asarray(projected_by_id[int(left)], dtype=np.float64)
            second = np.asarray(projected_by_id[int(right)], dtype=np.float64)
            dense_bridge_samples.extend(
                first * (1.0 - ratio) + second * ratio
                for ratio in np.linspace(0.025, 0.975, 39)
            )
        dense_bridges_contained = False
        if dense_bridge_samples:
            dense_samples = np.asarray(dense_bridge_samples, dtype=np.float64)
            dense_inside_outer = points_inside_polygon(dense_samples, outer_polygon)
            dense_inside_inner = points_inside_polygon(dense_samples, inner_polygon)
            dense_on_outer = np.zeros(len(dense_samples), dtype=bool)
            dense_outer_suspects = np.flatnonzero(~dense_inside_outer)
            if len(dense_outer_suspects):
                dense_on_outer[dense_outer_suspects] = points_on_polygon_boundary(
                    dense_samples[dense_outer_suspects],
                    outer_polygon,
                )
            dense_on_inner = np.zeros(len(dense_samples), dtype=bool)
            dense_inner_suspects = np.flatnonzero(dense_inside_inner)
            if len(dense_inner_suspects):
                dense_on_inner[dense_inner_suspects] = points_on_polygon_boundary(
                    dense_samples[dense_inner_suspects],
                    inner_polygon,
                )
            dense_bridges_contained = bool(
                np.all(dense_inside_outer | dense_on_outer)
                and np.all(~dense_inside_inner | dense_on_inner)
            )
        planar_transition_long_bridge_accepted = bool(
            dense_bridges_contained
            and measured_fanout <= audited_fanout_ceiling
            and (
                _parallel_planar_ring_transition(
                    np.asarray(
                        [point_by_id[int(value)] for value in outer_ids],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        [point_by_id[int(value)] for value in inner_ids],
                        dtype=np.float64,
                    ),
                )
                or _long_bridge_neighborhoods_are_coplanar(
                    faces,
                    point_by_id,
                    long_edges,
                )
            )
        )
    if (
        measured_fanout > audited_fanout_ceiling
        and not compact_fan_geometry
        and not user_reviewed_broad_fanout_accepted
    ):
        reasons.append("fanout_exceeded")
    projection_bridge_passthrough_accepted = bool(
        allow_projection_bridge_passthrough and invalid_bridges
    )
    if invalid_bridges and not projection_bridge_passthrough_accepted:
        reasons.append("bridge_leaves_annulus")
    if (
        excessive_cross
        and not planar_transition_long_bridge_accepted
        and not allow_projection_bridge_passthrough
    ):
        reasons.append("cross_edge_outlier")
    worst_bridge_id: tuple[int, int] | None = None
    if cross_edges:
        candidate_edges = [
            edge
            for edge, invalid in zip(bridge_edges, invalid_bridge_mask)
            if bool(invalid)
        ]
        if not candidate_edges and excessive_cross:
            candidate_edges = list(bridge_edges)
        if candidate_edges:
            worst_bridge_id = max(
                candidate_edges,
                key=lambda edge: float(
                    np.linalg.norm(
                        np.asarray(point_by_id[int(edge[1])], dtype=np.float64)
                        - np.asarray(point_by_id[int(edge[0])], dtype=np.float64)
                    )
                ),
            )
    if not projection.valid and not allow_projection_bridge_passthrough:
        reasons.append(projection.reason)
    projection_record = asdict(projection)
    projection_record['restored_source_ears'] = len(excluded)
    projection_record['user_reviewed_projection_passthrough'] = bool(
        allow_projection_bridge_passthrough and not projection.valid)
    return RingStripAudit(
        valid=not reasons,
        face_count=int(len(faces)),
        boundary_edge_count=int(len(required_edges)),
        open_edge_count=int(len(open_edges)),
        over_shared_edge_count=over_shared,
        degenerate_face_count=int(degenerate),
        maximum_fanout=int(measured_fanout),
        maximum_cross_edge_mm=float(maximum_cross),
        median_cross_edge_mm=float(median_cross),
        invalid_bridge_count=int(invalid_bridges),
        worst_bridge_id=worst_bridge_id,
        reason=",".join(reasons),
        projection=projection_record,
        planar_transition_long_bridge_accepted=bool(
            planar_transition_long_bridge_accepted
        ),
        projection_bridge_passthrough_accepted=bool(
            projection_bridge_passthrough_accepted
        ),
        user_reviewed_broad_fanout_accepted=bool(
            user_reviewed_broad_fanout_accepted
        ),
    )


def _parallel_planar_ring_transition(
    outer_points: np.ndarray,
    inner_points: np.ndarray,
    *,
    maximum_planarity_error_mm: float = 0.05,
    maximum_layer_separation_mm: float = 0.50,
    minimum_normal_cosine: float = 0.995,
) -> bool:
    """Prove that a long bridge lies between two nearby parallel planes.

    A concave inset can lose a thin lobe at a medial-axis event.  Its legal
    annular triangulation then needs one long *in-plane* diagonal even though
    the surface is not a 3-D blade.  This proof is intentionally unavailable
    to curved source rims: both complete rings must independently fit nearby,
    parallel planes within the production cap tolerance.
    """

    outer = np.asarray(outer_points, dtype=np.float64)
    inner = np.asarray(inner_points, dtype=np.float64)
    if len(outer) < 3 or len(inner) < 3:
        return False
    outer_normal = fit_plane_normal(outer, np.asarray([0.0, 0.0, 1.0]))
    inner_normal = fit_plane_normal(inner, outer_normal)
    if float(np.dot(outer_normal, inner_normal)) < 0.0:
        inner_normal = -inner_normal
    normal_cosine = abs(float(np.dot(outer_normal, inner_normal)))
    outer_origin = outer.mean(axis=0)
    inner_origin = inner.mean(axis=0)
    outer_error = float(np.abs((outer - outer_origin) @ outer_normal).max())
    inner_error = float(np.abs((inner - inner_origin) @ inner_normal).max())
    layer_separation = abs(float((inner_origin - outer_origin) @ outer_normal))
    return bool(
        outer_error <= float(maximum_planarity_error_mm) + 1e-12
        and inner_error <= float(maximum_planarity_error_mm) + 1e-12
        and layer_separation <= float(maximum_layer_separation_mm) + 1e-12
        and normal_cosine >= float(minimum_normal_cosine) - 1e-12
    )


def _long_bridge_neighborhoods_are_coplanar(
    faces: list[tuple[int, int, int]],
    point_by_id: dict[int, np.ndarray],
    long_edges: list[tuple[int, int]],
    *,
    maximum_local_planarity_error_mm: float = 0.05,
    minimum_normal_cosine: float = 0.995,
) -> bool:
    """Prove that each long cross edge is only a flat internal diagonal.

    Early backing layers may still inherit curvature from the source rim, so
    requiring two *complete* rings to be planar is unnecessarily strong.  A
    long triangulation edge is harmless exactly when its two incident faces
    form one locally flat patch.  Inspect the four local vertices directly;
    a folded ridge, tent, or blade fails either coplanarity or normal agreement.
    """

    if not long_edges:
        return False
    edge_faces: dict[tuple[int, int], list[tuple[int, int, int]]] = {
        tuple(sorted((int(left), int(right)))): []
        for left, right in long_edges
    }
    for face in faces:
        for left, right in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(left), int(right))))
            if edge in edge_faces:
                edge_faces[edge].append(tuple(int(value) for value in face))
    for incident in edge_faces.values():
        if len(incident) != 2:
            return False
        triangle_points = [
            np.asarray(
                [point_by_id[int(value)] for value in face],
                dtype=np.float64,
            )
            for face in incident
        ]
        normals = [
            np.cross(points[1] - points[0], points[2] - points[0])
            for points in triangle_points
        ]
        lengths = [float(np.linalg.norm(normal)) for normal in normals]
        if min(lengths) <= 1e-12:
            return False
        normal_cosine = abs(
            float(np.dot(normals[0], normals[1])) / (lengths[0] * lengths[1])
        )
        if normal_cosine < float(minimum_normal_cosine) - 1e-12:
            return False
        vertex_ids = sorted(
            {int(value) for face in incident for value in face}
        )
        local_points = np.asarray(
            [point_by_id[int(value)] for value in vertex_ids],
            dtype=np.float64,
        )
        local_normal = fit_plane_normal(local_points, normals[0])
        local_origin = local_points.mean(axis=0)
        local_error = float(
            np.abs((local_points - local_origin) @ local_normal).max()
        )
        if local_error > float(maximum_local_planarity_error_mm) + 1e-12:
            return False
    return True


def _quick_ring_strip_maximum_cross_edge(
    faces: list[tuple[int, int, int]],
    point_by_id: dict[int, np.ndarray],
    outer_ids: list[int],
) -> float:
    """Rank seam candidates in O(face-count) before the strict O(n²) audit.

    The strict audit remains authoritative.  Its candidate selection metric is
    the maximum cross-ring edge, so evaluating that same metric cheaply lets
    us audit candidates in best-first order without changing which valid seam
    wins.
    """

    outer_set = set(int(value) for value in outer_ids)
    cross_edges: set[tuple[int, int]] = set()
    for face in faces:
        for left, right in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            if (int(left) in outer_set) != (int(right) in outer_set):
                cross_edges.add(tuple(sorted((int(left), int(right)))))
    return max(
        (
            float(
                np.linalg.norm(
                    np.asarray(point_by_id[right], dtype=np.float64)
                    - np.asarray(point_by_id[left], dtype=np.float64)
                )
            )
            for left, right in cross_edges
        ),
        default=0.0,
    )


def _points_inside_polygon_vectorized(
    query: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """Classify many 2-D points without constructing an unbounded matrix."""

    query = np.asarray(query, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    result = np.zeros(len(query), dtype=bool)
    left = polygon
    right = np.roll(left, -1, axis=0)
    denominator = right[:, 1] - left[:, 1]
    safe = np.where(np.abs(denominator) <= 1e-15, 1.0, denominator)
    for start in range(0, len(query), 512):
        block = query[start : start + 512]
        y = block[:, 1, None]
        x = block[:, 0, None]
        crosses = (left[:, 1][None, :] > y) != (right[:, 1][None, :] > y)
        crossing_x = left[:, 0][None, :] + (
            (y - left[:, 1][None, :])
            * (right[:, 0] - left[:, 0])[None, :]
            / safe[None, :]
        )
        result[start : start + len(block)] = np.logical_xor.reduce(
            crosses & (x < crossing_x), axis=1
        )
    return result


def _points_on_polygon_boundary_vectorized(
    query: np.ndarray,
    polygon: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Classify points on polygon segments with the audit's strict tolerance."""

    query = np.asarray(query, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    result = np.zeros(len(query), dtype=bool)
    left = polygon
    right = np.roll(left, -1, axis=0)
    edge = right - left
    length_squared = np.einsum("ij,ij->i", edge, edge)
    edge_length = np.sqrt(length_squared)
    minimum = np.minimum(left, right) - float(tolerance)
    maximum = np.maximum(left, right) + float(tolerance)
    for start in range(0, len(query), 512):
        block = query[start : start + 512]
        x = block[:, 0, None]
        y = block[:, 1, None]
        within_box = (
            (x >= minimum[:, 0][None, :])
            & (x <= maximum[:, 0][None, :])
            & (y >= minimum[:, 1][None, :])
            & (y <= maximum[:, 1][None, :])
        )
        cross = (
            (x - left[:, 0][None, :]) * edge[:, 1][None, :]
            - (y - left[:, 1][None, :]) * edge[:, 0][None, :]
        )
        on_segment = (
            within_box
            & (length_squared[None, :] > 1e-24)
            & (np.abs(cross) <= float(tolerance) * edge_length[None, :])
        )
        result[start : start + len(block)] = np.any(on_segment, axis=1)
    return result


def _bridge_visibility_matrix(
    outer_projected: np.ndarray,
    inner_projected: np.ndarray,
) -> np.ndarray:
    """Return which outer/inner vertex bridges stay inside the annulus.

    A bounded-fan strip is a monotone path through this matrix.  Computing
    visibility once lets the fallback choose a different correspondence near
    concavities instead of accepting a many-to-one triangle fan.
    """

    outer = np.asarray(outer_projected, dtype=np.float64)
    inner = np.asarray(inner_projected, dtype=np.float64)
    visible = np.zeros((len(outer), len(inner)), dtype=bool)
    ratios = np.asarray((0.25, 0.50, 0.75), dtype=np.float64)
    pair_count = len(outer) * len(inner)
    for start in range(0, pair_count, 4096):
        flat = np.arange(start, min(start + 4096, pair_count), dtype=np.int64)
        outer_index = flat // len(inner)
        inner_index = flat % len(inner)
        first = outer[outer_index]
        second = inner[inner_index]
        samples = (
            first[:, None, :] * (1.0 - ratios[None, :, None])
            + second[:, None, :] * ratios[None, :, None]
        ).reshape((-1, 2))
        inside_outer = _points_inside_polygon_vectorized(samples, outer)
        inside_inner = _points_inside_polygon_vectorized(samples, inner)
        on_outer = np.zeros(len(samples), dtype=bool)
        outer_suspects = np.flatnonzero(~inside_outer)
        if len(outer_suspects):
            on_outer[outer_suspects] = _points_on_polygon_boundary_vectorized(
                samples[outer_suspects],
                outer,
            )
        on_inner = np.zeros(len(samples), dtype=bool)
        inner_suspects = np.flatnonzero(inside_inner)
        if len(inner_suspects):
            on_inner[inner_suspects] = _points_on_polygon_boundary_vectorized(
                samples[inner_suspects],
                inner,
            )
        pair_visible = np.all(
            (
                (inside_outer | on_outer)
                & (~inside_inner | on_inner)
            ).reshape((-1, 3)),
            axis=1,
        )
        visible[outer_index, inner_index] = pair_visible
    return visible


def _bounded_visible_ring_strip_faces(
    outer_ids: list[int],
    inner_ids: list[int],
    visibility: np.ndarray,
    *,
    maximum_consecutive_advances: int,
) -> list[tuple[int, int, int]] | None:
    """Find a visibility-constrained monotone strip with bounded local fans."""

    n = len(outer_ids)
    m = len(inner_ids)
    run_limit = max(1, int(maximum_consecutive_advances))
    # last axis: 0 advances outer, 1 advances inner.  A state stores the
    # previous run length; zero means unreachable.
    reached = np.zeros((n + 1, m + 1, 2, run_limit + 1), dtype=bool)
    parent_last = np.full(reached.shape, -1, dtype=np.int8)
    parent_run = np.full(reached.shape, -1, dtype=np.int8)
    reached[0, 0, 0, 1] = True
    reached[0, 0, 1, 1] = True

    def bridge_is_visible(i: int, j: int) -> bool:
        return bool(visibility[i % n, j % m])

    if not bridge_is_visible(0, 0):
        return None
    for i in range(n + 1):
        for j in range(m + 1):
            if i < n and bridge_is_visible(i + 1, j):
                for last in (0, 1):
                    for run in range(1, run_limit + 1):
                        if not reached[i, j, last, run]:
                            continue
                        next_run = run + 1 if last == 0 else 1
                        if next_run > run_limit or reached[i + 1, j, 0, next_run]:
                            continue
                        reached[i + 1, j, 0, next_run] = True
                        parent_last[i + 1, j, 0, next_run] = last
                        parent_run[i + 1, j, 0, next_run] = run
            if j < m and bridge_is_visible(i, j + 1):
                for last in (0, 1):
                    for run in range(1, run_limit + 1):
                        if not reached[i, j, last, run]:
                            continue
                        next_run = run + 1 if last == 1 else 1
                        if next_run > run_limit or reached[i, j + 1, 1, next_run]:
                            continue
                        reached[i, j + 1, 1, next_run] = True
                        parent_last[i, j + 1, 1, next_run] = last
                        parent_run[i, j + 1, 1, next_run] = run

    target: tuple[int, int] | None = None
    for last in (0, 1):
        for run in range(1, run_limit + 1):
            if reached[n, m, last, run]:
                target = (last, run)
                break
        if target is not None:
            break
    if target is None:
        return None

    moves: list[int] = []
    i, j = n, m
    last, run = target
    while i or j:
        moves.append(int(last))
        previous_last = int(parent_last[i, j, last, run])
        previous_run = int(parent_run[i, j, last, run])
        if last == 0:
            i -= 1
        else:
            j -= 1
        last, run = previous_last, previous_run
    moves.reverse()

    faces: list[tuple[int, int, int]] = []
    i = j = 0
    for move in moves:
        if move == 0:
            faces.append(
                (
                    int(outer_ids[i % n]),
                    int(outer_ids[(i + 1) % n]),
                    int(inner_ids[j % m]),
                )
            )
            i += 1
        else:
            faces.append(
                (
                    int(outer_ids[i % n]),
                    int(inner_ids[(j + 1) % m]),
                    int(inner_ids[j % m]),
                )
            )
            j += 1
    return faces


def triangulate_bounded_ring_strip(
    outer_ids: list[int],
    outer_points: np.ndarray,
    inner_ids: list[int],
    inner_points: np.ndarray,
    outer_projected: np.ndarray,
    inner_projected: np.ndarray,
    *,
    maximum_fanout: int = 4,
    allow_projection_bridge_passthrough: bool = False,
) -> RingStripResult:
    """Triangulate a side strip without allowing many-to-one triangle fans."""

    outer_ids = [int(value) for value in outer_ids]
    inner_ids = [int(value) for value in inner_ids]
    outer_points = np.asarray(outer_points, dtype=np.float64)
    inner_points = np.asarray(inner_points, dtype=np.float64)
    outer_projected = np.asarray(outer_projected, dtype=np.float64)
    inner_projected = np.asarray(inner_projected, dtype=np.float64)
    original_ring_count_ratio = max(len(outer_ids), len(inner_ids)) / max(
        min(len(outer_ids), len(inner_ids)),
        1,
    )
    prefer_constrained_annulus = bool(
        max(len(outer_ids), len(inner_ids)) >= 256
        and original_ring_count_ratio > 8.0
    )
    preconditioned_inner_ring = False
    if min(len(outer_ids), len(inner_ids)) < 3:
        raise ValueError("connector strip rings need at least three vertices")
    if signed_area(outer_projected) * signed_area(inner_projected) < 0.0:
        inner_ids.reverse()
        inner_points = inner_points[::-1].copy()
        inner_projected = inner_projected[::-1].copy()

    # A trimmed inset can lose vertices much faster than the source loop.  Add
    # only the minimum collinear samples required by the fanout construction;
    # never resample the real Clipper contour to the dense source count.
    if len(outer_ids) > 1.5 * len(inner_ids):
        minimum_inner_count = int(
            np.ceil(len(outer_ids) / max(int(maximum_fanout), 1))
        )
        inner_ids, inner_points, inner_projected = (
            _refine_ring_to_minimum_count(
            inner_ids,
            inner_points,
            inner_projected,
            minimum_count=minimum_inner_count,
            next_id=max(outer_ids + inner_ids) + 1,
        )
        )
        preconditioned_inner_ring = True
    if max(len(outer_ids), len(inner_ids)) > int(maximum_fanout) * min(
        len(outer_ids), len(inner_ids)
    ):
        raise ValueError("connector strip ring-count ratio exceeds bounded fanout")

    # Backing-profile rings are generated from the same ordered source loop.
    # When their counts match, corresponding indices are the authoritative
    # seam: searching nearby cyclic offsets only repeats the same O(n^2)
    # bridge audit many times on dense vendor contours.  Keep the strict audit
    # as the gate, but try this deterministic one-to-one strip once before the
    # general ranked-seam fallback.
    if len(outer_ids) == len(inner_ids):
        direct_result = _audit_equal_count_ring_strip(
            outer_ids,
            outer_points,
            outer_projected,
            inner_ids,
            inner_points,
            inner_projected,
            maximum_fanout=int(maximum_fanout),
            allow_preconditioned_compact_fan=preconditioned_inner_ring,
            strategy="aligned_equal_count_strict_audit",
        )
        runtime_log(
            "local-connector",
            "aligned_ring_strip_audited",
            "Aligned equal-count connector strip audit completed",
            ring_vertices=int(len(outer_ids)),
            valid=bool(direct_result.audit.valid),
            reason=str(direct_result.audit.reason),
            invalid_bridge_count=int(direct_result.audit.invalid_bridge_count),
            maximum_cross_edge_mm=float(
                direct_result.audit.maximum_cross_edge_mm
            ),
        )
        if direct_result.audit.valid:
            return direct_result

    nearest_distances, nearest_inner = cKDTree(inner_points).query(
        outer_points,
        k=1,
        workers=1,
    )
    seam_outer = int(np.argmin(nearest_distances))
    seam_inner = int(nearest_inner[seam_outer])
    # The original count ratio can be extreme even after the sparse contour
    # has been safely subdivided along its own straight edges.  At that point
    # the effective ratio is bounded by ``maximum_fanout`` and one linear,
    # fully audited seam is both cheap and useful.  Do not let the stale
    # pre-subdivision ratio force an otherwise well-conditioned strip into the
    # general polygon-with-hole solver.
    large_ring_fast_probe = bool(
        prefer_constrained_annulus
        and (
            preconditioned_inner_ring
            or max(len(outer_ids), len(inner_ids))
            >= LARGE_ANNULUS_FAST_PATH_VERTICES
        )
    )
    candidate_offsets = [0] if (not prefer_constrained_annulus or large_ring_fast_probe) else []
    if not prefer_constrained_annulus:
        for offset in range(1, 9):
            candidate_offsets.extend((offset, -offset))
    elif large_ring_fast_probe:
        runtime_log(
            "local-connector",
            "extreme_ring_ratio_linear_probe",
            "Large ring-count mismatch tries one audited linear seam before constrained annulus",
            outer_vertices=int(len(outer_ids)),
            inner_vertices=int(len(inner_ids)),
            original_ring_count_ratio=float(original_ring_count_ratio),
        )
    else:
        runtime_log(
            "local-connector",
            "extreme_ring_ratio_direct_constrained_annulus",
            "Extreme concave ring-count mismatch skips ranked seam search",
            outer_vertices=int(len(outer_ids)),
            inner_vertices=int(len(inner_ids)),
            original_ring_count_ratio=float(original_ring_count_ratio),
        )
    ranked_candidates: list[
        tuple[
            float,
            int,
            list[tuple[int, int, int]],
            list[int],
            list[int],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            dict[int, np.ndarray],
        ]
    ] = []
    best_failure: RingStripAudit | None = None
    required_consecutive_advances = int(
        np.ceil(max(len(outer_ids), len(inner_ids)) / min(len(outer_ids), len(inner_ids)))
    )
    allowed_consecutive_advances = min(
        int(maximum_fanout),
        max(int(maximum_fanout) - 1, required_consecutive_advances),
    )
    for candidate_order, offset in enumerate(candidate_offsets):
        inner_seam = (int(seam_inner) + int(offset)) % len(inner_ids)
        rotated_outer_ids = outer_ids[seam_outer:] + outer_ids[:seam_outer]
        rotated_inner_ids = inner_ids[inner_seam:] + inner_ids[:inner_seam]
        rotated_outer_points = np.concatenate(
            (outer_points[seam_outer:], outer_points[:seam_outer]), axis=0
        )
        rotated_inner_points = np.concatenate(
            (inner_points[inner_seam:], inner_points[:inner_seam]), axis=0
        )
        rotated_outer_projected = np.concatenate(
            (outer_projected[seam_outer:], outer_projected[:seam_outer]), axis=0
        )
        rotated_inner_projected = np.concatenate(
            (inner_projected[inner_seam:], inner_projected[:inner_seam]), axis=0
        )
        try:
            faces = _balanced_ring_strip_faces(
                rotated_outer_ids,
                rotated_outer_points,
                rotated_inner_ids,
                rotated_inner_points,
                maximum_consecutive_advances=max(1, allowed_consecutive_advances),
            )
        except ValueError:
            continue
        point_by_id = {
            **{int(i): p for i, p in zip(rotated_outer_ids, rotated_outer_points)},
            **{int(i): p for i, p in zip(rotated_inner_ids, rotated_inner_points)},
        }
        quick_maximum_cross = _quick_ring_strip_maximum_cross_edge(
            faces,
            point_by_id,
            rotated_outer_ids,
        )
        ranked_candidates.append(
            (
                float(quick_maximum_cross),
                int(candidate_order),
                faces,
                rotated_outer_ids,
                rotated_inner_ids,
                rotated_outer_points,
                rotated_inner_points,
                rotated_outer_projected,
                rotated_inner_projected,
                point_by_id,
            )
        )

    for (
        _quick_maximum_cross,
        _candidate_order,
        faces,
        rotated_outer_ids,
        rotated_inner_ids,
        rotated_outer_points,
        rotated_inner_points,
        rotated_outer_projected,
        rotated_inner_projected,
        point_by_id,
    ) in sorted(ranked_candidates, key=lambda candidate: (candidate[0], candidate[1])):
        projected_by_id = {
            **{int(i): p for i, p in zip(rotated_outer_ids, rotated_outer_projected)},
            **{int(i): p for i, p in zip(rotated_inner_ids, rotated_inner_projected)},
        }
        audit = _audit_ring_strip(
            faces,
            point_by_id,
            projected_by_id,
            rotated_outer_ids,
            rotated_inner_ids,
            rotated_outer_projected,
            rotated_inner_projected,
            maximum_fanout=int(maximum_fanout),
            allow_preconditioned_compact_fan=preconditioned_inner_ring,
        )
        if best_failure is None or audit.maximum_cross_edge_mm < best_failure.maximum_cross_edge_mm:
            best_failure = audit
        if audit.valid:
            return RingStripResult(
                tuple(faces),
                replace(
                    audit,
                    strategy="balanced_arc_length_ranked_seam_bounded_fanout",
                ),
                tuple(int(value) for value in inner_ids),
                np.asarray(inner_points, dtype=np.float64),
            )

    # A greedy arc-length walk can get trapped at a deep concavity even when a
    # clean bounded-fan correspondence exists.  For moderate rings, solve the
    # strip as a monotone path through the matrix of geometrically visible
    # bridges.  This changes the correspondence, not the acceptance threshold:
    # the same strict audit below remains authoritative.
    if (
        (not prefer_constrained_annulus or preconditioned_inner_ring)
        and max(len(outer_ids), len(inner_ids))
        <= MAXIMUM_VISIBILITY_PATH_RING_VERTICES
    ):
        visibility = _bridge_visibility_matrix(
            outer_projected,
            inner_projected,
        )
        visible_candidates: list[
            tuple[float, int, list[tuple[int, int, int]], list[int], list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for candidate_order, offset in enumerate(candidate_offsets):
            inner_seam = (int(seam_inner) + int(offset)) % len(inner_ids)
            rotated_outer_ids = outer_ids[seam_outer:] + outer_ids[:seam_outer]
            rotated_inner_ids = inner_ids[inner_seam:] + inner_ids[:inner_seam]
            rotated_outer_points = np.concatenate(
                (outer_points[seam_outer:], outer_points[:seam_outer]), axis=0
            )
            rotated_inner_points = np.concatenate(
                (inner_points[inner_seam:], inner_points[:inner_seam]), axis=0
            )
            rotated_outer_projected = np.concatenate(
                (outer_projected[seam_outer:], outer_projected[:seam_outer]), axis=0
            )
            rotated_inner_projected = np.concatenate(
                (inner_projected[inner_seam:], inner_projected[:inner_seam]), axis=0
            )
            rotated_visibility = np.roll(
                np.roll(visibility, -int(seam_outer), axis=0),
                -int(inner_seam),
                axis=1,
            )
            required_run = int(
                np.ceil(
                    max(len(rotated_outer_ids), len(rotated_inner_ids))
                    / min(len(rotated_outer_ids), len(rotated_inner_ids))
                )
            )
            for run_limit in range(
                max(1, required_run),
                max(1, 2 * int(maximum_fanout) - 2) + 1,
            ):
                visible_faces = _bounded_visible_ring_strip_faces(
                    rotated_outer_ids,
                    rotated_inner_ids,
                    rotated_visibility,
                    maximum_consecutive_advances=int(run_limit),
                )
                if not visible_faces:
                    continue
                point_by_id = {
                    **{int(i): p for i, p in zip(rotated_outer_ids, rotated_outer_points)},
                    **{int(i): p for i, p in zip(rotated_inner_ids, rotated_inner_points)},
                }
                visible_candidates.append(
                    (
                        _quick_ring_strip_maximum_cross_edge(
                            visible_faces,
                            point_by_id,
                            rotated_outer_ids,
                        ),
                        int(candidate_order * 16 + run_limit),
                        visible_faces,
                        rotated_outer_ids,
                        rotated_inner_ids,
                        rotated_outer_points,
                        rotated_inner_points,
                        rotated_outer_projected,
                        rotated_inner_projected,
                    )
                )
                # The smallest reachable run limit has the strongest fanout
                # guarantee for this seam; larger limits cannot improve it.
                break
        for (
            _quick_maximum_cross,
            _candidate_order,
            faces,
            rotated_outer_ids,
            rotated_inner_ids,
            rotated_outer_points,
            rotated_inner_points,
            rotated_outer_projected,
            rotated_inner_projected,
        ) in sorted(visible_candidates, key=lambda candidate: (candidate[0], candidate[1])):
            point_by_id = {
                **{int(i): p for i, p in zip(rotated_outer_ids, rotated_outer_points)},
                **{int(i): p for i, p in zip(rotated_inner_ids, rotated_inner_points)},
            }
            projected_by_id = {
                **{int(i): p for i, p in zip(rotated_outer_ids, rotated_outer_projected)},
                **{int(i): p for i, p in zip(rotated_inner_ids, rotated_inner_projected)},
            }
            audit = _audit_ring_strip(
                faces,
                point_by_id,
                projected_by_id,
                rotated_outer_ids,
                rotated_inner_ids,
                rotated_outer_projected,
                rotated_inner_projected,
                maximum_fanout=int(maximum_fanout),
                allow_preconditioned_compact_fan=preconditioned_inner_ring,
            )
            if best_failure is None or audit.maximum_cross_edge_mm < best_failure.maximum_cross_edge_mm:
                best_failure = audit
            if audit.valid:
                return RingStripResult(
                    tuple(faces),
                    replace(
                        audit,
                        strategy="visibility_dynamic_program_bounded_fanout",
                    ),
                    tuple(int(value) for value in inner_ids),
                    np.asarray(inner_points, dtype=np.float64),
                )

    # Deeply concave trimmed contours can change perimeter parameterization
    # discontinuously.  Let the constrained polygon-with-hole solver find the
    # exact non-crossing strip, then densify only the offending boundary edges.
    # This preserves both original boundaries and converts a 35-way tent into
    # short, printable local fans without resampling the complete inset ring.
    constrained_faces, constrained_audit = triangulate_connector_annulus(
        outer_ids,
        outer_projected,
        inner_ids,
        inner_projected,
    )
    from .projected_micro_folds import crossing_pairs
    if not constrained_audit.valid or crossing_pairs(outer_projected):
        # A curved rim may fold in projection while remaining valid in 3-D.
        # Solve its simple core and restore the bounded original ears exactly.
        from .projected_micro_folds import isolate_micro_folds
        repair = isolate_micro_folds(outer_points, outer_projected)
        if repair is not None:
            retained = list(repair.retained_indices)
            core_faces, core_audit = triangulate_connector_annulus(
                [outer_ids[index] for index in retained],
                outer_projected[retained], inner_ids, inner_projected)
            if core_audit.valid:
                restored = core_faces + [tuple(outer_ids[index] for index in ear)
                                         for ear in repair.ear_indices]
                points_by_id = dict(zip(outer_ids + inner_ids,
                                       np.concatenate((outer_points, inner_points))))
                projected_by_id = dict(zip(outer_ids + inner_ids,
                                          np.concatenate((outer_projected, inner_projected))))
                restored_audit = _audit_ring_strip(
                    restored, points_by_id, projected_by_id, outer_ids, inner_ids,
                    outer_projected, inner_projected, maximum_fanout=int(maximum_fanout),
                    allow_preconditioned_compact_fan=preconditioned_inner_ring,
                    restored_source_ears=tuple(tuple(outer_ids[i] for i in ear)
                                               for ear in repair.ear_indices),
                    projection_core_outer_ids=[outer_ids[i] for i in retained])
                if restored_audit.valid:
                    runtime_log('local-connector', 'projected_micro_folds_restored',
                                '已保留原三维边界并修复局部投影折返',
                                **repair.evidence, restored_ears=len(repair.ear_indices))
                    return RingStripResult(
                        tuple(restored), replace(restored_audit,
                                                strategy='constrained_annulus_restored_micro_folds'),
                        tuple(inner_ids), np.asarray(inner_points, dtype=np.float64))
    if constrained_audit.valid and constrained_faces:
        current_inner_ids = list(inner_ids)
        current_inner_points = np.asarray(inner_points, dtype=np.float64)
        current_inner_projected = np.asarray(inner_projected, dtype=np.float64)
        next_id = max(outer_ids + inner_ids) + 1
        previous_badness: tuple[int, float] | None = None
        stagnant_iterations = 0
        subdivision_count = 0
        for _ in range(64):
            point_by_id = {
                **{int(i): p for i, p in zip(outer_ids, outer_points)},
                **{
                    int(i): p
                    for i, p in zip(current_inner_ids, current_inner_points)
                },
            }
            projected_by_id = {
                **{int(i): p for i, p in zip(outer_ids, outer_projected)},
                **{
                    int(i): p
                    for i, p in zip(current_inner_ids, current_inner_projected)
                },
            }
            audit = _audit_ring_strip(
                constrained_faces,
                point_by_id,
                projected_by_id,
                outer_ids,
                current_inner_ids,
                outer_projected,
                current_inner_projected,
                maximum_fanout=int(maximum_fanout),
                allow_preconditioned_compact_fan=preconditioned_inner_ring,
            )
            if best_failure is None or audit.maximum_cross_edge_mm < best_failure.maximum_cross_edge_mm:
                best_failure = audit
            if audit.valid:
                strategy = (
                    "constrained_annulus_local_edge_subdivision"
                    if subdivision_count
                    else "constrained_annulus_exact"
                )
                return RingStripResult(
                    tuple(constrained_faces),
                    RingStripAudit(
                        **{
                            **audit.__dict__,
                            "strategy": strategy,
                        }
                    ),
                    tuple(int(value) for value in current_inner_ids),
                    np.asarray(current_inner_points, dtype=np.float64),
                )
            badness = (
                int(audit.maximum_fanout),
                round(float(audit.maximum_cross_edge_mm), 9),
            )
            if badness == previous_badness:
                stagnant_iterations += 1
            else:
                stagnant_iterations = 0
                previous_badness = badness
            if stagnant_iterations >= 3:
                break
            fanout: collections.Counter[int] = collections.Counter()
            outer_set = set(outer_ids)
            inner_set = set(current_inner_ids)
            for face in constrained_faces:
                for left, right in (
                    (face[0], face[1]),
                    (face[1], face[2]),
                    (face[2], face[0]),
                ):
                    if (int(left) in outer_set) != (int(right) in outer_set):
                        inner_id = int(right) if int(left) in outer_set else int(left)
                        fanout[inner_id] += 1
            fanout_limit = 2 * int(maximum_fanout)
            split_vertices = [
                int(vertex_id)
                for vertex_id, count in sorted(
                    fanout.items(),
                    key=lambda item: (-int(item[1]), int(item[0])),
                )
                if int(count) > fanout_limit
            ]
            if not split_vertices and audit.worst_bridge_id is not None:
                bridge_inner_ids = [
                    int(value)
                    for value in audit.worst_bridge_id
                    if int(value) in inner_set
                ]
                if bridge_inner_ids:
                    split_vertices = [
                        max(
                            bridge_inner_ids,
                            key=lambda vertex_id: int(fanout[int(vertex_id)]),
                        )
                    ]
            if not split_vertices:
                break
            point_map = {
                int(i): np.asarray(p, dtype=np.float64)
                for i, p in zip(current_inner_ids, current_inner_points)
            }
            projected_map = {
                int(i): np.asarray(p, dtype=np.float64)
                for i, p in zip(current_inner_ids, current_inner_projected)
            }
            refined_ids: list[int] = []
            refined_points: list[np.ndarray] = []
            refined_projected: list[np.ndarray] = []
            # Refine every current hotspot in one pass.  Large vendor rings can
            # contain dozens of independent high-fanout corners; fixing only
            # one per retriangulation exhausts the retry budget before the
            # first production strip is printable.
            insert_after_indices: set[int] = set()
            for split_vertex in split_vertices[:64]:
                split_index = current_inner_ids.index(int(split_vertex))
                previous_index = (split_index - 1) % len(current_inner_ids)
                following_index = (split_index + 1) % len(current_inner_ids)
                previous_length = float(
                    np.linalg.norm(
                        current_inner_projected[split_index]
                        - current_inner_projected[previous_index]
                    )
                )
                following_length = float(
                    np.linalg.norm(
                        current_inner_projected[following_index]
                        - current_inner_projected[split_index]
                    )
                )
                insert_after_indices.add(
                    split_index
                    if following_length >= previous_length
                    else previous_index
                )
            for index, vertex_id in enumerate(current_inner_ids):
                following_id = current_inner_ids[(index + 1) % len(current_inner_ids)]
                refined_ids.append(int(vertex_id))
                refined_points.append(point_map[int(vertex_id)])
                refined_projected.append(projected_map[int(vertex_id)])
                if index not in insert_after_indices:
                    continue
                refined_ids.append(int(next_id))
                refined_points.append(
                    0.5 * (point_map[int(vertex_id)] + point_map[int(following_id)])
                )
                refined_projected.append(
                    0.5
                    * (
                        projected_map[int(vertex_id)]
                        + projected_map[int(following_id)]
                    )
                )
                next_id += 1
            current_inner_ids = refined_ids
            current_inner_points = np.asarray(refined_points, dtype=np.float64)
            current_inner_projected = np.asarray(refined_projected, dtype=np.float64)
            subdivision_count += int(len(insert_after_indices))
            constrained_faces, constrained_audit = triangulate_connector_annulus(
                outer_ids,
                outer_projected,
                current_inner_ids,
                current_inner_projected,
            )
            if not constrained_audit.valid:
                break

    # A strongly sculpted source rim can fold over itself only in the chosen
    # two-dimensional assembly projection.  Once the user has visually
    # approved that rim, the projection-containment test is advisory: build a
    # deterministic bounded-run strip and retain the hard three-dimensional
    # checks for boundary incidence, shared edges, fanout, and degeneracy.
    # The same audited path is useful both after sparse-ring subdivision and
    # for moderate count ratios whose selected 2-D projection still folds.
    # Its finite fanout/cross-edge policy and hard topology audit remain the
    # gate, so preconditioning is not itself a safety requirement.
    advisory_failure: RingStripAudit | None = None
    if allow_projection_bridge_passthrough:
        rotated_outer_ids = outer_ids[seam_outer:] + outer_ids[:seam_outer]
        rotated_inner_ids = inner_ids[seam_inner:] + inner_ids[:seam_inner]
        rotated_outer_points = np.concatenate(
            (outer_points[seam_outer:], outer_points[:seam_outer]), axis=0
        )
        rotated_inner_points = np.concatenate(
            (inner_points[seam_inner:], inner_points[:seam_inner]), axis=0
        )
        rotated_outer_projected = np.concatenate(
            (outer_projected[seam_outer:], outer_projected[:seam_outer]), axis=0
        )
        rotated_inner_projected = np.concatenate(
            (inner_projected[seam_inner:], inner_projected[:seam_inner]), axis=0
        )
        advisory_faces = _bounded_visible_ring_strip_faces(
            rotated_outer_ids,
            rotated_inner_ids,
            np.ones(
                (len(rotated_outer_ids), len(rotated_inner_ids)),
                dtype=bool,
            ),
            maximum_consecutive_advances=max(
                1,
                int(
                    np.ceil(
                        max(len(rotated_outer_ids), len(rotated_inner_ids))
                        / min(len(rotated_outer_ids), len(rotated_inner_ids))
                    )
                ),
            ),
        )
        if advisory_faces:
            point_by_id = {
                **{
                    int(i): p
                    for i, p in zip(rotated_outer_ids, rotated_outer_points)
                },
                **{
                    int(i): p
                    for i, p in zip(rotated_inner_ids, rotated_inner_points)
                },
            }
            projected_by_id = {
                **{
                    int(i): p
                    for i, p in zip(
                        rotated_outer_ids,
                        rotated_outer_projected,
                    )
                },
                **{
                    int(i): p
                    for i, p in zip(
                        rotated_inner_ids,
                        rotated_inner_projected,
                    )
                },
            }
            advisory_audit = _audit_ring_strip(
                advisory_faces,
                point_by_id,
                projected_by_id,
                rotated_outer_ids,
                rotated_inner_ids,
                rotated_outer_projected,
                rotated_inner_projected,
                maximum_fanout=int(maximum_fanout),
                allow_preconditioned_compact_fan=True,
                allow_projection_bridge_passthrough=True,
            )
            advisory_failure = advisory_audit
            if advisory_audit.valid:
                return RingStripResult(
                    tuple(advisory_faces),
                    replace(
                        advisory_audit,
                        strategy=(
                            "bounded_parameter_path_"
                            "user_reviewed_projection_advisory"
                        ),
                    ),
                    tuple(int(value) for value in inner_ids),
                    np.asarray(inner_points, dtype=np.float64),
                )
    failure_reason = "no_valid_bounded_fanout_seam"
    if best_failure is not None:
        failure_reason += (
            f":best={best_failure.reason},"
            f"fanout={best_failure.maximum_fanout},"
            f"cross={best_failure.maximum_cross_edge_mm:.6f},"
            f"median={best_failure.median_cross_edge_mm:.6f},"
            f"invalid_bridges={best_failure.invalid_bridge_count}"
        )
    if advisory_failure is not None:
        failure_reason += (
            f";advisory={advisory_failure.reason},"
            f"fanout={advisory_failure.maximum_fanout},"
            f"cross={advisory_failure.maximum_cross_edge_mm:.6f},"
            f"invalid_bridges={advisory_failure.invalid_bridge_count},"
            f"degenerate={advisory_failure.degenerate_face_count},"
            f"open={advisory_failure.open_edge_count},"
            f"over_shared={advisory_failure.over_shared_edge_count}"
        )
    failure_reason += (
        f";advisory_context=enabled={bool(allow_projection_bridge_passthrough)},"
        f"preconditioned={bool(preconditioned_inner_ring)}"
    )
    empty = RingStripAudit(
        False,
        0,
        int(len(outer_ids) + len(inner_ids)),
        0,
        0,
        0,
        0,
        0.0,
        0.0,
        0,
        failure_reason,
        "balanced_arc_length_bounded_fanout",
        None,
        False,
    )
    return RingStripResult(
        tuple(),
        empty,
        tuple(int(value) for value in inner_ids),
        np.asarray(inner_points, dtype=np.float64),
    )


def audit_annulus_faces(
    faces: list[tuple[int, int, int]],
    point_by_id: dict[int, np.ndarray],
    outer_ids: list[int],
    hole_ids: list[int],
    *,
    area_tolerance: float | None = None,
    allow_signed_projection_folds: bool = False,
) -> PatchTopologyAudit:
    """Require exact boundary ownership, finite faces, and exact planar area."""

    required_edges = _ring_edges(outer_ids) | _ring_edges(hole_ids)
    edge_counts: collections.Counter[tuple[int, int]] = collections.Counter()
    degenerate = 0
    covered = 0.0
    signed_covered = 0.0
    negative_face_count = 0
    for face in faces:
        if len(set(int(value) for value in face)) != 3:
            degenerate += 1
            continue
        triangle = np.asarray(
            [point_by_id[int(value)] for value in face],
            dtype=np.float64,
        )
        signed_double_area = float(
            (triangle[1, 0] - triangle[0, 0])
            * (triangle[2, 1] - triangle[0, 1])
            - (triangle[1, 1] - triangle[0, 1])
            * (triangle[2, 0] - triangle[0, 0])
        )
        double_area = abs(signed_double_area)
        scale = max(float(np.linalg.norm(np.ptp(triangle, axis=0))), 1.0)
        if double_area <= scale * scale * 1e-13:
            degenerate += 1
        covered += 0.5 * double_area
        signed_covered += 0.5 * signed_double_area
        negative_face_count += int(signed_double_area < 0.0)
        for left, right in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge_counts[tuple(sorted((int(left), int(right))))] += 1
    open_edges = {edge for edge, count in edge_counts.items() if int(count) == 1}
    over_shared = int(sum(int(count) > 2 for count in edge_counts.values()))
    expected = abs(
        float(
            signed_area(
                np.asarray([point_by_id[int(value)] for value in outer_ids])
            )
        )
    ) - abs(
        float(
            signed_area(
                np.asarray([point_by_id[int(value)] for value in hole_ids])
            )
        )
    )
    tolerance = (
        max(1e-7, expected * 1e-8)
        if area_tolerance is None
        else max(float(area_tolerance), 0.0)
    )
    reasons: list[str] = []
    if not faces:
        reasons.append("no_faces")
    if degenerate:
        reasons.append("degenerate_faces")
    if over_shared:
        reasons.append("over_shared_edges")
    if open_edges != required_edges:
        reasons.append("boundary_mismatch")
    if any(int(count) not in (1, 2) for count in edge_counts.values()):
        reasons.append("edge_incidence")
    projection_fold_budget = max(1e-7, expected * 2e-5)
    signed_projection_fold_is_safe = bool(
        allow_signed_projection_folds
        and negative_face_count <= 8
        and abs(abs(signed_covered) - expected) <= tolerance
        and covered - abs(signed_covered) <= projection_fold_budget
    )
    if abs(covered - expected) > tolerance and not signed_projection_fold_is_safe:
        reasons.append("area_mismatch")
    return PatchTopologyAudit(
        valid=not reasons,
        face_count=int(len(faces)),
        boundary_edge_count=int(len(required_edges)),
        open_edge_count=int(len(open_edges)),
        over_shared_edge_count=over_shared,
        degenerate_face_count=int(degenerate),
        covered_area=float(covered),
        expected_area=float(expected),
        reason=",".join(reasons),
    )


def _cyclic_arc(values: list, start: int, stop: int) -> list:
    result = [values[int(start)]]
    cursor = int(start)
    while cursor != int(stop):
        cursor = (cursor + 1) % len(values)
        result.append(values[cursor])
        if len(result) > len(values) + 1:
            raise RuntimeError("connector ring arc did not close")
    return result


def _polar_ring_order(
    ids: list[int],
    points: np.ndarray,
    center: np.ndarray,
) -> tuple[list[int], np.ndarray, np.ndarray] | None:
    """Return one CCW star-shaped ring with monotonically unwrapped angles."""

    ring_ids = [int(value) for value in ids]
    ring = np.asarray(points, dtype=np.float64)
    if signed_area(ring) < 0.0:
        ring_ids.reverse()
        ring = ring[::-1].copy()
    raw = np.arctan2(ring[:, 1] - center[1], ring[:, 0] - center[0])
    seam = int(np.argmin(raw))
    ring_ids = ring_ids[seam:] + ring_ids[:seam]
    ring = np.concatenate((ring[seam:], ring[:seam]), axis=0)
    angles = np.unwrap(
        np.concatenate((raw[seam:], raw[:seam]), axis=0)
    )
    angles -= float(angles[0])
    scale = max(float(np.ptp(ring[:, 0])), float(np.ptp(ring[:, 1])), 1.0)
    radius = np.linalg.norm(ring - center[None, :], axis=1)
    if np.any(radius <= scale * 1e-10):
        return None
    if np.any(np.diff(angles) < -1e-9):
        return None
    # ``angles`` was translated so the seam is zero.  Comparing it again to
    # the original raw seam angle mixes two coordinate systems and rejected
    # ordinary concentric circles.  In the translated frame the closing ray is
    # exactly one full revolution from the seam.
    if 2.0 * np.pi < float(angles[-1]) - 1e-9:
        return None
    return ring_ids, ring, angles


def triangulate_star_shaped_annulus(
    outer_ids: list[int],
    outer_points: np.ndarray,
    hole_ids: list[int],
    hole_points: np.ndarray,
) -> tuple[list[tuple[int, int, int]], PatchTopologyAudit]:
    """Linear-time monotone zipper for two rings star-shaped about one center."""

    outer = np.asarray(outer_points, dtype=np.float64)
    hole = np.asarray(hole_points, dtype=np.float64)
    center = np.asarray(hole, dtype=np.float64).mean(axis=0)
    if not (
        point_in_poly(center, outer)
        and point_in_poly(center, hole)
    ):
        empty = PatchTopologyAudit(
            False, 0, 0, 0, 0, 0, 0.0, 0.0,
            "rings_do_not_share_star_center", "polar_monotone_zipper"
        )
        return [], empty
    outer_order = _polar_ring_order(outer_ids, outer, center)
    hole_order = _polar_ring_order(hole_ids, hole, center)
    if outer_order is None or hole_order is None:
        empty = PatchTopologyAudit(
            False, 0, 0, 0, 0, 0, 0.0, 0.0,
            "ring_not_star_shaped", "polar_monotone_zipper"
        )
        return [], empty
    outer_ring, outer, outer_angles = outer_order
    hole_ring, hole, hole_angles = hole_order
    outer_angles = outer_angles * (
        2.0 * np.pi / max(float(outer_angles[-1] + 1e-12), 2.0 * np.pi)
    )
    hole_angles = hole_angles * (
        2.0 * np.pi / max(float(hole_angles[-1] + 1e-12), 2.0 * np.pi)
    )
    n = len(outer_ring)
    m = len(hole_ring)
    faces: list[tuple[int, int, int]] = []
    i = j = 0
    while i < n or j < m:
        next_outer = (
            float(outer_angles[i + 1])
            if i + 1 < n
            else 2.0 * np.pi
        )
        next_hole = (
            float(hole_angles[j + 1])
            if j + 1 < m
            else 2.0 * np.pi
        )
        if i < n and (j >= m or next_outer <= next_hole):
            faces.append(
                (
                    int(outer_ring[i % n]),
                    int(outer_ring[(i + 1) % n]),
                    int(hole_ring[j % m]),
                )
            )
            i += 1
        else:
            faces.append(
                (
                    int(outer_ring[i % n]),
                    int(hole_ring[(j + 1) % m]),
                    int(hole_ring[j % m]),
                )
            )
            j += 1
    point_by_id = {
        **{
            int(vertex_id): np.asarray(point, dtype=np.float64)
            for vertex_id, point in zip(outer_ring, outer)
        },
        **{
            int(vertex_id): np.asarray(point, dtype=np.float64)
            for vertex_id, point in zip(hole_ring, hole)
        },
    }
    audit = audit_annulus_faces(
        faces,
        point_by_id,
        outer_ring,
        hole_ring,
    )
    return faces, PatchTopologyAudit(
        **{**audit.__dict__, "strategy": "polar_monotone_zipper"}
    )


def _bridge_is_visible(
    outer_point: np.ndarray,
    hole_point: np.ndarray,
    outer: np.ndarray,
    hole: np.ndarray,
    epsilon: float,
) -> bool:
    if float(np.linalg.norm(outer_point - hole_point)) <= epsilon:
        return False
    # Several interior samples avoid accepting a bridge whose midpoint happens
    # to sit inside while another segment portion leaves a deep concave bay.
    for ratio in (0.20, 0.40, 0.60, 0.80):
        sample = outer_point * (1.0 - ratio) + hole_point * ratio
        if not (
            point_in_poly(sample, outer)
            or point_on_poly_boundary(sample, outer, epsilon)
        ):
            return False
        if point_in_poly(sample, hole) and not point_on_poly_boundary(
            sample, hole, epsilon
        ):
            return False
    for ring in (outer, hole):
        for index in range(len(ring)):
            left = np.asarray(ring[index], dtype=np.float64)
            right = np.asarray(ring[(index + 1) % len(ring)], dtype=np.float64)
            if segments_intersect_2d_strict(
                outer_point,
                hole_point,
                left,
                right,
                epsilon,
            ):
                return False
    return True


def triangulate_annulus_with_visible_bridges(
    outer_ids: list[int],
    outer_points: np.ndarray,
    hole_ids: list[int],
    hole_points: np.ndarray,
) -> tuple[list[tuple[int, int, int]], PatchTopologyAudit]:
    """Triangulate one concave annulus using two explicit visible bridges.

    Two bridges partition the annulus into genuinely simple polygons, avoiding
    the duplicated vertices of a weakly-simple one-bridge representation.  The
    candidate search is geometry based and the result is accepted only after an
    exact edge-incidence and area audit.
    """

    outer = np.asarray(outer_points, dtype=np.float64)
    hole = np.asarray(hole_points, dtype=np.float64)
    outer_ring = [int(value) for value in outer_ids]
    hole_ring = [int(value) for value in hole_ids]
    if len(outer_ring) < 3 or len(hole_ring) < 3:
        empty = PatchTopologyAudit(False, 0, 0, 0, 0, 0, 0.0, 0.0, "short_ring")
        return [], empty
    if signed_area(outer) < 0.0:
        outer = outer[::-1].copy()
        outer_ring.reverse()
    if signed_area(hole) > 0.0:
        hole = hole[::-1].copy()
        hole_ring.reverse()
    scale = max(
        float(np.ptp(outer[:, 0])),
        float(np.ptp(outer[:, 1])),
        1.0,
    )
    epsilon = scale * 1e-10
    point_by_id = {
        **{
            int(vertex_id): np.asarray(point, dtype=np.float64)
            for vertex_id, point in zip(outer_ring, outer)
        },
        **{
            int(vertex_id): np.asarray(point, dtype=np.float64)
            for vertex_id, point in zip(hole_ring, hole)
        },
    }

    visible: list[tuple[float, int, int]] = []
    for hole_index, hole_point in enumerate(hole):
        candidates: list[tuple[float, int, int]] = []
        for outer_index, outer_point in enumerate(outer):
            if not _bridge_is_visible(
                outer_point,
                hole_point,
                outer,
                hole,
                epsilon,
            ):
                continue
            distance = float(np.sum((outer_point - hole_point) ** 2))
            candidates.append((distance, int(outer_index), int(hole_index)))
        candidates.sort(key=lambda item: item[0])
        # Retain alternatives across concave bays; four was too aggressive for
        # dense V-shaped vendor boundaries.
        visible.extend(candidates[:16])

    pairs: list[
        tuple[float, float, tuple[float, int, int], tuple[float, int, int]]
    ] = []
    for left_position, left in enumerate(visible):
        for right in visible[left_position + 1 :]:
            if int(left[1]) == int(right[1]) or int(left[2]) == int(right[2]):
                continue
            if segments_intersect_2d_strict(
                outer[int(left[1])],
                hole[int(left[2])],
                outer[int(right[1])],
                hole[int(right[2])],
                epsilon,
            ):
                continue
            outer_delta = abs(int(left[1]) - int(right[1]))
            hole_delta = abs(int(left[2]) - int(right[2]))
            outer_balance = min(outer_delta, len(outer_ring) - outer_delta) / float(
                len(outer_ring)
            )
            hole_balance = min(hole_delta, len(hole_ring) - hole_delta) / float(
                len(hole_ring)
            )
            balance = min(outer_balance, hole_balance)
            pairs.append((-balance, float(left[0] + right[0]), left, right))
    pairs.sort(key=lambda item: (item[0], item[1]))

    attempted_pairs = 0
    triangulation_failures = 0
    topology_failures = 0
    for _balance, _distance, left, right in pairs[:64]:
        attempted_pairs += 1
        outer_left, hole_left = int(left[1]), int(left[2])
        outer_right, hole_right = int(right[1]), int(right[2])
        polygons = (
            (
                _cyclic_arc(outer_ring, outer_left, outer_right)
                + _cyclic_arc(hole_ring, hole_right, hole_left),
                _cyclic_arc(list(outer), outer_left, outer_right)
                + _cyclic_arc(list(hole), hole_right, hole_left),
            ),
            (
                _cyclic_arc(outer_ring, outer_right, outer_left)
                + _cyclic_arc(hole_ring, hole_left, hole_right),
                _cyclic_arc(list(outer), outer_right, outer_left)
                + _cyclic_arc(list(hole), hole_left, hole_right),
            ),
        )
        faces: list[tuple[int, int, int]] = []
        failed = False
        for polygon_ids, polygon_points in polygons:
            local_faces = triangulate_polygon_ear_clip(
                np.asarray(polygon_points, dtype=np.float64)
            )
            if len(local_faces) != len(polygon_ids) - 2:
                failed = True
                break
            faces.extend(
                tuple(int(polygon_ids[int(index)]) for index in face)
                for face in local_faces
            )
        if failed:
            triangulation_failures += 1
            continue
        audit = audit_annulus_faces(
            faces,
            point_by_id,
            outer_ring,
            hole_ring,
        )
        if audit.valid:
            return faces, audit
        topology_failures += 1

    empty = audit_annulus_faces([], point_by_id, outer_ring, hole_ring)
    return [], PatchTopologyAudit(
        **{
            **empty.__dict__,
            "reason": (
                "no_valid_visible_bridge_pair:"
                f"visible={len(visible)},pairs={len(pairs)},"
                f"attempted={attempted_pairs},"
                f"triangulation_failures={triangulation_failures},"
                f"topology_failures={topology_failures}"
            ),
        }
    )


def triangulate_connector_annulus(
    outer_ids: list[int],
    outer_points: np.ndarray,
    hole_ids: list[int],
    hole_points: np.ndarray,
) -> tuple[list[tuple[int, int, int]], PatchTopologyAudit]:
    """Strategy dispatcher for the common star-shaped and general concave cases."""

    outer = np.asarray(outer_points, dtype=np.float64)
    hole = np.asarray(hole_points, dtype=np.float64)
    outer_ring = [int(value) for value in outer_ids]
    hole_ring = [int(value) for value in hole_ids]
    if signed_area(outer) < 0.0:
        outer = outer[::-1].copy()
        outer_ring.reverse()
    if signed_area(hole) > 0.0:
        hole = hole[::-1].copy()
        hole_ring.reverse()
    total_ring_vertices = int(len(outer_ring) + len(hole_ring))
    star_audit: PatchTopologyAudit | None = None
    if total_ring_vertices >= LARGE_ANNULUS_FAST_PATH_VERTICES:
        started_at = time.perf_counter()
        faces, star_audit = triangulate_star_shaped_annulus(
            outer_ring,
            outer,
            hole_ring,
            hole,
        )
        runtime_log(
            "local-connector",
            "dense_annulus_star_probe_done",
            "Audited dense-annulus linear star probe completed",
            ring_vertices=total_ring_vertices,
            valid=bool(star_audit.valid),
            reason=str(star_audit.reason),
            elapsed_seconds=float(time.perf_counter() - started_at),
        )
        if star_audit.valid:
            return faces, star_audit

    manifold_available = True
    manifold_audit: PatchTopologyAudit | None = None
    manifold_started_at = time.perf_counter()
    try:
        import manifold3d

        flattened_ids = outer_ring + hole_ring
        triangles = np.asarray(
            manifold3d.triangulate([outer, hole], allow_convex=False),
            dtype=np.int64,
        ).reshape((-1, 3))
        faces = [
            tuple(int(flattened_ids[int(index)]) for index in triangle)
            for triangle in triangles
        ]
        point_by_id = {
            **{
                int(vertex_id): np.asarray(point, dtype=np.float64)
                for vertex_id, point in zip(outer_ring, outer)
            },
            **{
                int(vertex_id): np.asarray(point, dtype=np.float64)
                for vertex_id, point in zip(hole_ring, hole)
            },
        }
        manifold_audit = audit_annulus_faces(
            faces,
            point_by_id,
            outer_ring,
            hole_ring,
            allow_signed_projection_folds=True,
        )
        if manifold_audit.valid:
            runtime_log(
                "local-connector",
                "constrained_annulus_done",
                "Constrained annulus triangulation completed",
                ring_vertices=total_ring_vertices,
                valid=True,
                elapsed_seconds=float(time.perf_counter() - manifold_started_at),
            )
            return faces, PatchTopologyAudit(
                **{
                    **manifold_audit.__dict__,
                    "strategy": "manifold3d_constrained_polygon_with_hole",
                }
            )
    except ImportError:
        manifold_available = False
    except (RuntimeError, ValueError):
        pass

    runtime_log(
        "local-connector",
        "constrained_annulus_done",
        "Constrained annulus triangulation did not yield an accepted patch",
        ring_vertices=total_ring_vertices,
        available=bool(manifold_available),
        reason=(
            str(manifold_audit.reason)
            if manifold_audit is not None
            else "solver_exception_or_unavailable"
        ),
        elapsed_seconds=float(time.perf_counter() - manifold_started_at),
    )

    if star_audit is None:
        faces, star_audit = triangulate_star_shaped_annulus(
            outer_ring,
            outer,
            hole_ring,
            hole,
        )
        if star_audit.valid:
            return faces, star_audit
    if total_ring_vertices > DENSE_VISIBLE_BRIDGE_FALLBACK_VERTICES:
        reason = (
            "manifold3d_required_for_dense_concave_annulus"
            if not manifold_available
            else "constrained_solver_failed_for_dense_concave_annulus"
        )
        return [], PatchTopologyAudit(
            **{
                **star_audit.__dict__,
                "reason": reason,
                "strategy": "dependency_guard",
            }
        )
    faces, bridge_audit = triangulate_annulus_with_visible_bridges(
        outer_ring,
        outer,
        hole_ring,
        hole,
    )
    return faces, PatchTopologyAudit(
        **{
            **bridge_audit.__dict__,
            "strategy": "two_visible_bridge_partition",
        }
    )


def orient_face_patch_consistently(
    faces: list[tuple[int, int, int]],
    point_by_id: dict[int, np.ndarray],
    outward_normal: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Orient one manifold triangle patch through shared-edge propagation."""

    oriented = [list(int(value) for value in face) for face in faces]
    edge_faces: dict[
        tuple[int, int], list[tuple[int, tuple[int, int]]]
    ] = collections.defaultdict(list)
    for face_index, face in enumerate(oriented):
        for edge in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge_faces[tuple(sorted(edge))].append((int(face_index), edge))
    if any(len(records) not in (1, 2) for records in edge_faces.values()):
        raise ValueError("connector patch is not edge-manifold")
    normal = np.asarray(outward_normal, dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    visited: set[int] = set()
    for start in range(len(oriented)):
        if start in visited:
            continue
        component: list[int] = []
        queue = collections.deque([int(start)])
        visited.add(int(start))
        while queue:
            current = int(queue.popleft())
            component.append(current)
            face = oriented[current]
            for edge in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            ):
                records = edge_faces[tuple(sorted(edge))]
                if len(records) != 2:
                    continue
                neighbor = records[0][0] if records[1][0] == current else records[1][0]
                if int(neighbor) in visited:
                    continue
                neighbor_face = oriented[int(neighbor)]
                neighbor_edges = (
                    (neighbor_face[0], neighbor_face[1]),
                    (neighbor_face[1], neighbor_face[2]),
                    (neighbor_face[2], neighbor_face[0]),
                )
                if edge in neighbor_edges:
                    neighbor_face[1], neighbor_face[2] = (
                        neighbor_face[2], neighbor_face[1]
                    )
                visited.add(int(neighbor))
                queue.append(int(neighbor))
        component_dot = 0.0
        for index in component:
            triangle = np.asarray(
                [point_by_id[int(value)] for value in oriented[index]],
                dtype=np.float64,
            )
            component_dot += float(
                np.dot(
                    np.cross(
                        triangle[1] - triangle[0],
                        triangle[2] - triangle[0],
                    ),
                    normal,
                )
            )
        if component_dot < 0.0:
            for index in component:
                oriented[index][1], oriented[index][2] = (
                    oriented[index][2], oriented[index][1]
                )
    return [tuple(face) for face in oriented]


class ConnectorTopologyService:
    """Stateless façade used by builders and harnesses."""

    triangulate_annulus = staticmethod(triangulate_connector_annulus)
    triangulate_ring_strip = staticmethod(triangulate_bounded_ring_strip)
    audit_annulus = staticmethod(audit_annulus_faces)
    orient_patch = staticmethod(orient_face_patch_consistently)
