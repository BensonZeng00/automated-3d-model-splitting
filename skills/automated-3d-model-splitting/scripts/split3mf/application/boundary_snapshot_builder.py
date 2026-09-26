from __future__ import annotations

import hashlib

import numpy as np

from ..mesh import boundary_loops, build_local_mesh
from ..common import Component
from .recognized_boundaries import RecognizedBoundaries


class BoundarySnapshotBuilder:
    """Freeze component boundary loops from the final recognized ownership."""

    def build(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        components: list[Component],
    ) -> RecognizedBoundaries:
        component_loops = []
        component_loop_points = []
        digest = hashlib.sha256()
        for component_index, component in enumerate(components, start=1):
            _local_vertices, local_faces, _global_to_local, global_vertex_ids = build_local_mesh(
                vertices, faces, component
            )
            loops = tuple(
                tuple(int(global_vertex_ids[int(local_id)]) for local_id in loop)
                for loop in boundary_loops(local_faces)
            )
            point_loops = tuple(
                tuple(
                    tuple(float(value) for value in vertices[vertex_id])
                    for vertex_id in loop
                )
                for loop in loops
            )
            component_loops.append(loops)
            component_loop_points.append(point_loops)
            digest.update(np.asarray([component_index], dtype=np.int64).tobytes())
            for loop in loops:
                digest.update(np.asarray(loop, dtype=np.int64).tobytes())
                digest.update(b"\0")
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
        )
