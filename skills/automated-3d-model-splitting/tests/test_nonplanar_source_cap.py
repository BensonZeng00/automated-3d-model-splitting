from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.part_interface_composition import _ordered_bridge, compose_part_with_interfaces


class NonplanarSourceCapTests(unittest.TestCase):
    def test_boundary_bridge_tracks_arclength_across_unequal_reversed_rings(self):
        source_count, target_count = 101, 98
        source_angles = np.linspace(0, 2 * np.pi, source_count, endpoint=False)
        target_angles = np.linspace(0, 2 * np.pi, target_count, endpoint=False)
        source = np.column_stack((np.cos(source_angles), np.sin(source_angles), np.zeros(source_count)))
        target = np.column_stack((
            np.cos(target_angles + 0.17), np.sin(target_angles + 0.17),
            np.full(target_count, 0.1),
        ))[::-1]
        source_ids = np.arange(source_count)
        target_ids = np.arange(target_count) + source_count
        bridge = _ordered_bridge(source, source_ids, target, target_ids)
        vertices = np.vstack((source, target))
        edge_lengths = [
            np.linalg.norm(vertices[first] - vertices[second])
            for face in bridge
            for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
        ]
        self.assertLess(max(edge_lengths), 0.26)

    def test_large_nearly_convex_spatial_rim_gets_planar_cap(self):
        count = 64
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        top = np.column_stack((
            10 * np.cos(angles), 8 * np.sin(angles), 0.6 * np.sin(3 * angles)
        ))
        bottom = np.column_stack((
            10 * np.cos(angles), 8 * np.sin(angles), np.full(count, -3.0)
        ))
        vertices = np.vstack((top, bottom))
        faces = []
        for position in range(count):
            following = (position + 1) % count
            faces.extend((
                [position, following, count + position],
                [following, count + following, count + position],
            ))
        mesh, record = compose_part_with_interfaces(
            vertices, np.asarray(faces, dtype=np.int64), []
        )
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertEqual(int(np.count_nonzero(mesh.area_faces <= 1e-10)), 0)
        self.assertEqual(len(record["reconstructed_nonplanar_caps"]), 1)
        self.assertTrue(record["reconstructed_nonplanar_caps"][0]["cap_quality"]["valid"])


if __name__ == "__main__":
    unittest.main()
