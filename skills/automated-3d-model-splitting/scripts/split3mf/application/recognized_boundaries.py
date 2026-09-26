from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecognizedBoundaries:
    """Immutable simplified, smoothed boundaries frozen after recognition.

    ``component_loops`` is indexed by one-based component identity minus one.
    Each loop stores IDs of source vertices sampled before smoothing. The IDs
    identify sample correspondence; ``component_loop_points`` contains the
    final coordinates passed to later stages.
    """

    component_loops: tuple[tuple[tuple[int, ...], ...], ...]
    component_loop_points: tuple[tuple[tuple[tuple[float, float, float], ...], ...], ...]
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
        """Compatibility alias; raw boundary loops are no longer retained."""
        return self.loops_for_component(component_index)

    def flattened_arrays(self) -> dict[str, object]:
        import numpy as np

        component_offsets = [0]
        loop_offsets = [0]
        vertices: list[int] = []
        points: list[tuple[float, float, float]] = []
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
        }
