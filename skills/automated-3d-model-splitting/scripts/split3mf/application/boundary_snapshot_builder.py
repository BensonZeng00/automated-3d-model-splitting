from __future__ import annotations

import hashlib

import numpy as np

from ..mesh import boundary_loops, build_local_mesh
from ..common import Component
from .recognized_boundaries import RecognizedBoundaries

MINIMUM_BOUNDARY_LOOP_SPAN_MM = 1.0


class BoundarySnapshotBuilder:
    """Freeze component boundary loops from the final recognized ownership."""

    def build_with_component_filter(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        components: list[Component],
        retained_fraction: float = 0.05,
    ) -> tuple[list[Component], RecognizedBoundaries, list[dict]]:
        """Exclude components with no accepted ring during recognition itself."""
        candidate_components = list(components)
        boundaries = self.build(vertices, faces, candidate_components, retained_fraction)
        excluded = [dict(record) for record in boundaries.excluded_components]
        excluded_indices = {
            int(record["candidate_component_index"])
            for record in boundaries.excluded_components
        }
        for record in excluded:
            record["candidate_part"] = f"P{int(record['candidate_component_index']):02d}"
        retained_components = [
            component
            for index, component in enumerate(candidate_components, start=1)
            if index not in excluded_indices
        ]
        if not retained_components:
            raise ValueError("recognition boundary filtering removed every component")
        return retained_components, boundaries, excluded

    def build(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        components: list[Component],
        retained_fraction: float = 0.05,
    ) -> RecognizedBoundaries:
        component_loops: list[tuple[tuple[int, ...], ...]] = []
        component_loop_points: list[tuple[tuple[tuple[float, float, float], ...], ...]] = []
        simplification_records: list[dict] = []
        excluded_components: list[dict] = []
        digest = hashlib.sha256()
        for component_index, component in enumerate(components, start=1):
            _local_vertices, local_faces, _global_to_local, global_vertex_ids = build_local_mesh(
                vertices, faces, component
            )
            raw_loops = tuple(
                tuple(int(global_vertex_ids[int(local_id)]) for local_id in loop)
                for loop in boundary_loops(local_faces, _local_vertices)
            )
            loops_for_component = []
            points_for_component = []
            loop_perimeters = [
                float(np.linalg.norm(
                    np.roll(np.asarray(vertices[np.asarray(loop, dtype=np.int64)]), -1, axis=0)
                    - np.asarray(vertices[np.asarray(loop, dtype=np.int64)]), axis=1
                ).sum())
                for loop in raw_loops
            ]
            for loop_index, source_loop in enumerate(raw_loops, start=1):
                source_ids = np.asarray(source_loop, dtype=np.int64)
                source_points = np.asarray(vertices[source_ids], dtype=np.float64)
                perimeter = loop_perimeters[loop_index - 1]
                span = float(np.max(np.ptp(source_points, axis=0))) if len(source_points) else 0.0
                if (
                    span >= MINIMUM_BOUNDARY_LOOP_SPAN_MM
                    and _is_geometrically_degenerate_loop(source_points)
                ):
                    simplification_records.append({
                        "component_index": component_index,
                        "loop_index": loop_index,
                        "status": "filtered",
                        "reason": "geometrically_degenerate_boundary_loop",
                        "source_vertex_count": int(len(source_ids)),
                        "simplified_vertex_count": 0,
                        "retained_fraction": float(retained_fraction),
                        "perimeter_mm": perimeter,
                        "span_mm": span,
                    })
                    continue
                if span < MINIMUM_BOUNDARY_LOOP_SPAN_MM:
                    simplification_records.append({
                        "component_index": component_index,
                        "loop_index": loop_index,
                        "status": "filtered",
                        "reason": "below_minimum_boundary_span",
                        "source_vertex_count": int(len(source_ids)),
                        "simplified_vertex_count": 0,
                        "retained_fraction": float(retained_fraction),
                        "filtered_spike_vertex_count": 0,
                        "perimeter_mm": perimeter,
                        "span_mm": span,
                        "minimum_span_mm": MINIMUM_BOUNDARY_LOOP_SPAN_MM,
                    })
                    continue
                retained = np.arange(len(source_ids), dtype=np.int64)
                filtered_source_indices = _remove_isolated_spikes(source_points)
                status = "unchanged"
                reason = None
                if len(source_ids) < 3:
                    status, reason = "filtered", "fewer_than_three_vertices"
                else:
                    filtered_points = source_points[filtered_source_indices]
                    retained_count = max(
                        3, int(round(len(filtered_source_indices) * float(retained_fraction)))
                    )
                    retained_count = min(len(filtered_source_indices), retained_count)
                    retained_in_filtered = _equal_arc_sample_indices(
                        filtered_points, retained_count
                    )
                    retained = filtered_source_indices[retained_in_filtered]
                    status = (
                        "simplified"
                        if len(retained) < len(source_ids) or len(filtered_source_indices) < len(source_ids)
                        else "unchanged"
                    )
                    if len(filtered_source_indices) < len(source_ids):
                        reason = "isolated_spike_vertices_filtered"
                if status == "filtered":
                    simplification_records.append({
                        "component_index": component_index,
                        "loop_index": loop_index,
                        "status": status,
                        "reason": reason,
                        "source_vertex_count": int(len(source_ids)),
                        "simplified_vertex_count": 0,
                        "retained_fraction": float(retained_fraction),
                    })
                    continue
                simplified_ids = source_ids[retained]
                simplified_points = _taubin_smooth_closed_loop(
                    np.asarray(vertices[simplified_ids], dtype=np.float64)
                )
                record = {
                    "component_index": component_index,
                    "loop_index": loop_index,
                    "status": status,
                    "reason": reason,
                    "source_vertex_count": int(len(source_ids)),
                    "simplified_vertex_count": int(len(simplified_ids)),
                    "retained_fraction": float(retained_fraction),
                    "filtered_spike_vertex_count": int(
                        len(source_ids) - len(filtered_source_indices)
                    ),
                    "perimeter_mm": perimeter,
                    "span_mm": span,
                    "smoothing_algorithm": "taubin_closed_loop",
                    "smoothing_iterations": 6,
                }
                simplification_records.append(record)
                loops_for_component.append(tuple(int(value) for value in simplified_ids))
                points_for_component.append(tuple(
                    tuple(float(value) for value in point)
                    for point in simplified_points
                ))
            loops = tuple(loops_for_component)
            point_loops = tuple(points_for_component)
            digest.update(np.asarray([component_index], dtype=np.int64).tobytes())
            for loop in loops:
                digest.update(np.asarray(loop, dtype=np.int64).tobytes())
                digest.update(b"\0")
            if not loops:
                filtered_reasons = sorted({
                    str(record.get("reason"))
                    for record in simplification_records
                    if int(record.get("component_index", -1)) == component_index
                    and record.get("status") == "filtered"
                })
                excluded_components.append({
                    "candidate_component_index": component_index,
                    "reason": (
                        "all_boundary_loops_filtered"
                        if raw_loops
                        else "no_closed_boundary"
                    ),
                    "filter_reasons": filtered_reasons,
                    "face_count": int(component.face_count),
                    "color_code": str(component.color_code),
                })
                continue
            component_loops.append(loops)
            component_loop_points.append(point_loops)
        for point_loops in component_loop_points:
            for loop_points in point_loops:
                digest.update(np.asarray(loop_points, dtype=np.float64).tobytes())
                digest.update(b"\0")
        return RecognizedBoundaries(
            component_loops=tuple(component_loops),
            component_loop_points=tuple(component_loop_points),
            source_vertex_count=int(len(vertices)),
            source_face_count=int(len(faces)),
            fingerprint=digest.hexdigest(),
            simplification_records=tuple(simplification_records),
            excluded_components=tuple(excluded_components),
        )


def _remove_isolated_spikes(points: np.ndarray) -> np.ndarray:
    """Remove sparse, sharp excursions while retaining cyclic source order."""
    values = np.asarray(points, dtype=np.float64)
    retained = list(range(len(values)))
    maximum_removals = max(1, int(len(values) * 0.02))
    removed = 0
    for _pass in range(maximum_removals):
        if len(retained) <= 3 or removed >= maximum_removals:
            break
        current = values[np.asarray(retained, dtype=np.int64)]
        edges = np.linalg.norm(np.roll(current, -1, axis=0) - current, axis=1)
        if len(current) <= 7:
            break
        previous = np.roll(current, 1, axis=0)
        following = np.roll(current, -1, axis=0)
        chord = following - previous
        chord_squared = np.einsum("ij,ij->i", chord, chord)
        projection = np.einsum("ij,ij->i", current - previous, chord)
        fraction = np.zeros(len(current), dtype=np.float64)
        valid_chords = chord_squared > 1e-18
        fraction[valid_chords] = np.clip(
            projection[valid_chords] / chord_squared[valid_chords], 0.0, 1.0
        )
        nearest = previous + fraction[:, None] * chord
        deviation = np.linalg.norm(current - nearest, axis=1)
        deviation[~valid_chords] = np.linalg.norm(
            current[~valid_chords] - previous[~valid_chords], axis=1
        )
        chord_length = np.sqrt(chord_squared)
        incoming = np.roll(edges, 1)
        outgoing = edges
        local_reference = np.stack((
            np.roll(edges, 4), np.roll(edges, 3),
            np.roll(edges, -1), np.roll(edges, -2),
        ), axis=1)
        local_reference[local_reference <= 1e-12] = np.nan
        local_scale = np.nanmedian(local_reference, axis=1)
        local_scale = np.nan_to_num(local_scale, nan=0.0)
        path_length = incoming + outgoing
        candidate_mask = (
            (local_scale > 1e-9)
            & (deviation > np.maximum(0.02, 4.0 * local_scale))
            & (path_length > 4.0 * local_scale)
            & (path_length > 2.5 * np.maximum(chord_length, 1e-9))
        )
        candidate_indices = np.flatnonzero(candidate_mask)
        if not len(candidate_indices):
            break
        scores = deviation[candidate_indices] / local_scale[candidate_indices]
        index_to_remove = int(candidate_indices[int(np.argmax(scores))])
        retained.pop(index_to_remove)
        removed += 1
    return np.asarray(retained, dtype=np.int64)


def _is_geometrically_degenerate_loop(points: np.ndarray) -> bool:
    """Reject graph cycles that collapse to a line in source geometry.

    Vertex-only boundary graphs can create tiny closed walks along collinear
    T-junction subdivisions. They are topological cycles but have no enclosed
    interface area, and rendering them produces the short dangling strokes
    visible in recognition previews.
    """
    values = np.asarray(points, dtype=np.float64)
    if len(values) < 3:
        return True
    centered = values - values.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered)
    largest = float(eigenvalues[-1])
    second = float(eigenvalues[-2])
    return largest <= 1e-18 or second <= largest * 1e-8


def _taubin_smooth_closed_loop(
    points: np.ndarray,
    *,
    iterations: int = 6,
    relaxation: float = 0.35,
    inflation: float = -0.36,
) -> np.ndarray:
    """Smooth a cyclic polyline in place conceptually, preserving sample count.

    Taubin's positive/negative Laplacian passes reduce high-frequency contour
    noise while counteracting the shrinkage of ordinary Laplacian smoothing.
    Every input sample remains represented by one output point.
    """
    smoothed = np.asarray(points, dtype=np.float64).copy()
    if len(smoothed) < 3:
        return smoothed
    for _ in range(max(0, int(iterations))):
        neighbors = (np.roll(smoothed, 1, axis=0) + np.roll(smoothed, -1, axis=0)) * 0.5
        smoothed += float(relaxation) * (neighbors - smoothed)
        neighbors = (np.roll(smoothed, 1, axis=0) + np.roll(smoothed, -1, axis=0)) * 0.5
        smoothed += float(inflation) * (neighbors - smoothed)
    return smoothed


def _equal_arc_sample_indices(points: np.ndarray, sample_count: int) -> np.ndarray:
    """Choose unique source vertices nearest equal physical arc intervals."""
    values = np.asarray(points, dtype=np.float64)
    count = min(max(3, int(sample_count)), len(values))
    edge_lengths = np.linalg.norm(np.roll(values, -1, axis=0) - values, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
    perimeter = float(cumulative[-1])
    if perimeter <= 1e-12:
        return np.floor(np.arange(count) * len(values) / count).astype(np.int64)
    targets = np.arange(count, dtype=np.float64) * perimeter / count
    selected: list[int] = []
    for target in targets:
        right = min(int(np.searchsorted(cumulative, target, side="left")), len(values))
        left = (right - 1) % len(values)
        right_index = right % len(values)
        left_distance = abs(float(target) - float(cumulative[right - 1])) if right else perimeter
        right_distance = abs(float(cumulative[right]) - float(target)) if right < len(values) else perimeter - float(target)
        candidate = left if left_distance <= right_distance else right_index
        if candidate not in selected:
            selected.append(candidate)
    if len(selected) < count:
        for index in np.argsort(np.min(
            np.abs(cumulative[:-1, None] - targets[None, :]), axis=1
        )):
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == count:
                break
    return np.asarray(sorted(selected), dtype=np.int64)
