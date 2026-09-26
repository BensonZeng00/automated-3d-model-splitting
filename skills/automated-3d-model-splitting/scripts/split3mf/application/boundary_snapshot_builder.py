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
        source_component_loops: list[tuple[tuple[int, ...], ...]] = []
        source_component_loop_points: list[tuple[tuple[tuple[float, float, float], ...], ...]] = []
        simplification_records: list[dict] = []
        excluded_components: list[dict] = []
        digest = hashlib.sha256()
        for component_index, component in enumerate(components, start=1):
            _local_vertices, local_faces, _global_to_local, global_vertex_ids = build_local_mesh(
                vertices, faces, component
            )
            raw_loops = tuple(
                tuple(int(global_vertex_ids[int(local_id)]) for local_id in loop)
                for loop in boundary_loops(local_faces)
            )
            raw_point_loops = tuple(
                tuple(tuple(float(value) for value in vertices[vertex_id]) for vertex_id in loop)
                for loop in raw_loops
            )
            loops_for_component = []
            points_for_component = []
            retained_source_loops = []
            retained_source_point_loops = []
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
                simplified_points = np.asarray(vertices[simplified_ids], dtype=np.float64)
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
                }
                simplification_records.append(record)
                retained_source_loops.append(source_loop)
                retained_source_point_loops.append(raw_point_loops[loop_index - 1])
                loops_for_component.append(tuple(int(value) for value in simplified_ids))
                points_for_component.append(tuple(
                    tuple(float(value) for value in point)
                    for point in simplified_points
                ))
            loops = tuple(loops_for_component)
            point_loops = tuple(points_for_component)
            digest.update(np.asarray([component_index], dtype=np.int64).tobytes())
            for loop in raw_loops:
                digest.update(np.asarray(loop, dtype=np.int64).tobytes())
                digest.update(b"\1")
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
            # Keep the unsimplified topology only for accepted rings. This keeps
            # downstream seam planning from resurrecting discarded micro-loops.
            source_component_loops.append(tuple(retained_source_loops))
            source_component_loop_points.append(tuple(retained_source_point_loops))
        for point_loops in component_loop_points:
            for loop_points in point_loops:
                digest.update(np.asarray(loop_points, dtype=np.float64).tobytes())
                digest.update(b"\0")
        for point_loops in source_component_loop_points:
            for loop_points in point_loops:
                digest.update(np.asarray(loop_points, dtype=np.float64).tobytes())
                digest.update(b"\1")
        return RecognizedBoundaries(
            component_loops=tuple(component_loops),
            component_loop_points=tuple(component_loop_points),
            source_component_loops=tuple(source_component_loops),
            source_component_loop_points=tuple(source_component_loop_points),
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
    for _pass in range(4):
        if len(retained) <= 3 or removed >= maximum_removals:
            break
        current = values[np.asarray(retained, dtype=np.int64)]
        edges = np.linalg.norm(np.roll(current, -1, axis=0) - current, axis=1)
        candidates = []
        for index, point in enumerate(current):
            if len(current) <= 7:
                break
            previous = current[(index - 1) % len(current)]
            following = current[(index + 1) % len(current)]
            chord = following - previous
            chord_squared = float(np.dot(chord, chord))
            if chord_squared <= 1e-18:
                deviation = float(np.linalg.norm(point - previous))
                chord_length = 0.0
            else:
                fraction = float(np.clip(np.dot(point - previous, chord) / chord_squared, 0.0, 1.0))
                deviation = float(np.linalg.norm(point - (previous + fraction * chord)))
                chord_length = float(np.sqrt(chord_squared))
            incoming = float(edges[(index - 1) % len(edges)])
            outgoing = float(edges[index])
            local_reference = [
                float(edges[(index - 4) % len(edges)]),
                float(edges[(index - 3) % len(edges)]),
                float(edges[(index + 1) % len(edges)]),
                float(edges[(index + 2) % len(edges)]),
            ]
            local_scale = float(np.median([length for length in local_reference if length > 1e-12]) or 0.0)
            if local_scale <= 1e-9:
                continue
            path_length = incoming + outgoing
            if (
                deviation > max(0.02, 4.0 * local_scale)
                and path_length > 4.0 * local_scale
                and path_length > 2.5 * max(chord_length, 1e-9)
            ):
                candidates.append((deviation / local_scale, index))
        if not candidates:
            break
        _, index_to_remove = max(candidates)
        retained.pop(index_to_remove)
        removed += 1
    return np.asarray(retained, dtype=np.int64)


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
