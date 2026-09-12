from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from split3mf.common import Component
from split3mf.interface_retreat import retreat_interface_ownership


def component(vertices, faces, face_ids, color_code) -> Component:
    face_ids = np.asarray(face_ids, dtype=np.int64)
    triangles = vertices[faces[face_ids]]
    points = triangles.reshape(-1, 3)
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    return Component(
        color_code=color_code,
        global_faces=face_ids,
        face_count=len(face_ids),
        area=float(0.5 * np.linalg.norm(cross, axis=1).sum()),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        center=points.mean(axis=0),
    )


def grid_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [(float(x), float(y), 0.0) for y in range(5) for x in range(5)],
        dtype=np.float64,
    )
    faces = []
    for y in range(4):
        for x in range(4):
            lower_left = y * 5 + x
            lower_right = lower_left + 1
            upper_left = lower_left + 5
            upper_right = upper_left + 1
            faces.extend(
                [
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                ]
            )
    return vertices, np.asarray(faces, dtype=np.int64)


class InterfaceRetreatTests(unittest.TestCase):
    def test_geodesic_retreat_transfers_local_parent_patch(self) -> None:
        vertices, faces = grid_mesh()
        centroids = vertices[faces].mean(axis=1)
        child_ids = np.flatnonzero(centroids[:, 0] < 2.0)
        parent_ids = np.flatnonzero(centroids[:, 0] >= 2.0)
        components = [
            component(vertices, faces, child_ids, "BEIGE"),
            component(vertices, faces, parent_ids, "YELLOW"),
        ]

        result = retreat_interface_ownership(
            vertices,
            faces,
            components,
            child_index=1,
            parent_index=2,
            seed_point_mm=np.asarray([2.0, 1.5, 0.0]),
            seed_radius_mm=0.8,
            retreat_distance_mm=0.85,
            maximum_parent_face_fraction=0.75,
        )

        self.assertGreater(result.record["transferred_face_count"], 0)
        self.assertLess(
            result.record["transferred_face_count"],
            len(parent_ids),
        )
        self.assertEqual(
            result.components[0].face_count + result.components[1].face_count,
            len(faces),
        )
        self.assertTrue(result.record["source_materials_preserved"])
        self.assertEqual(result.record["new_child_connected_region_count"], 1)
        self.assertEqual(result.record["new_parent_connected_region_count"], 1)

    def test_retreat_rejects_seed_away_from_shared_boundary(self) -> None:
        vertices, faces = grid_mesh()
        centroids = vertices[faces].mean(axis=1)
        child_ids = np.flatnonzero(centroids[:, 0] < 2.0)
        parent_ids = np.flatnonzero(centroids[:, 0] >= 2.0)
        components = [
            component(vertices, faces, child_ids, "BEIGE"),
            component(vertices, faces, parent_ids, "YELLOW"),
        ]
        with self.assertRaisesRegex(ValueError, "does not reach"):
            retreat_interface_ownership(
                vertices,
                faces,
                components,
                child_index=1,
                parent_index=2,
                seed_point_mm=np.asarray([100.0, 100.0, 0.0]),
                seed_radius_mm=0.5,
                retreat_distance_mm=1.0,
            )


if __name__ == "__main__":
    unittest.main()
