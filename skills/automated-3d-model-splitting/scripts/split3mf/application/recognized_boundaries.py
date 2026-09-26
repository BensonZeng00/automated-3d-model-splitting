from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecognizedBoundaries:
    """Immutable source-mesh boundary loops frozen after recognition.

    ``component_loops`` is indexed by one-based component identity minus one.
    Each loop stores source vertex IDs in traversal order.
    """

    component_loops: tuple[tuple[tuple[int, ...], ...], ...]
    component_loop_points: tuple[tuple[tuple[tuple[float, float, float], ...], ...], ...]
    source_component_loops: tuple[tuple[tuple[int, ...], ...], ...]
    source_component_loop_points: tuple[tuple[tuple[tuple[float, float, float], ...], ...], ...]
    source_vertex_count: int
    source_face_count: int
    fingerprint: str
    simplification_records: tuple[dict, ...] = ()
    excluded_components: tuple[dict, ...] = ()

    def loops_for_component(self, component_index: int) -> tuple[tuple[int, ...], ...]:
        index = int(component_index) - 1
        if index < 0 or index >= len(self.component_loops):
            raise IndexError(f"unknown recognized component P{component_index:02d}")
        return self.component_loops[index]

    def source_loops_for_component(self, component_index: int) -> tuple[tuple[int, ...], ...]:
        """Return unsimplified loops for exact source-edge topology operations."""
        index = int(component_index) - 1
        if index < 0 or index >= len(self.source_component_loops):
            raise IndexError(f"unknown recognized component P{component_index:02d}")
        return self.source_component_loops[index]

    def flattened_arrays(self) -> dict[str, object]:
        import numpy as np

        component_offsets = [0]
        loop_offsets = [0]
        vertices: list[int] = []
        points: list[tuple[float, float, float]] = []
        source_component_offsets = [0]
        source_loop_offsets = [0]
        source_vertices: list[int] = []
        source_points: list[tuple[float, float, float]] = []
        for loops, point_loops in zip(
            self.source_component_loops, self.source_component_loop_points
        ):
            for loop, loop_points in zip(loops, point_loops):
                source_vertices.extend(loop)
                source_points.extend(loop_points)
                source_loop_offsets.append(len(source_vertices))
            source_component_offsets.append(len(source_loop_offsets) - 1)
        for loops, point_loops in zip(self.component_loops, self.component_loop_points):
            for loop, loop_points in zip(loops, point_loops):
                vertices.extend(loop)
                points.extend(loop_points)
                loop_offsets.append(len(vertices))
            component_offsets.append(len(loop_offsets) - 1)
        return {
            "boundary_vertex_ids": np.asarray(vertices, dtype=np.int64),
            "boundary_points": np.asarray(points, dtype=np.float64).reshape((-1, 3)),
            "boundary_loop_offsets": np.asarray(loop_offsets, dtype=np.int64),
            "component_loop_offsets": np.asarray(component_offsets, dtype=np.int64),
            "source_boundary_vertex_ids": np.asarray(source_vertices, dtype=np.int64),
            "source_boundary_points": np.asarray(source_points, dtype=np.float64).reshape((-1, 3)),
            "source_boundary_loop_offsets": np.asarray(source_loop_offsets, dtype=np.int64),
            "source_component_loop_offsets": np.asarray(source_component_offsets, dtype=np.int64),
        }
