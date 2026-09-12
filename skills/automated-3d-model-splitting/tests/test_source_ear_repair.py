from pathlib import Path
import sys
import unittest

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.source_ear_repair import restore_source_ears
from split3mf.print_tolerance import PrintTolerance, tolerance_scope


class SourceEarRepairTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.Trimesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [.5, -.01, .01],
                               [.5, 1, 0], [.5, .5, 1]]) * .1,
            faces=[[1, 0, 3], [0, 1, 2], [0, 1, 4], [1, 3, 4], [3, 0, 4]],
            process=False)

    def test_restore_real_boundary_without_changing_source_or_vertices(self):
        result, record = restore_source_ears(self.mesh, 2)
        self.assertTrue(record['applied'])
        self.assertTrue(result.is_watertight)
        np.testing.assert_array_equal(result.vertices[:len(self.mesh.vertices)], self.mesh.vertices)
        np.testing.assert_array_equal(result.faces[:2], self.mesh.faces[:2])
        self.assertEqual(len(result.faces), len(self.mesh.faces) + 3)
        self.assertTrue(np.all(result.area_faces > 0))

    def test_strict_and_large_patches_do_not_change(self):
        with tolerance_scope(PrintTolerance(micro_area_mm2=0)):
            result, record = restore_source_ears(self.mesh, 2)
            self.assertFalse(record['applied'])
        enlarged = self.mesh.copy()
        enlarged.apply_scale(100)
        _, record = restore_source_ears(enlarged, 2)
        self.assertFalse(record['applied'])

    def test_no_generated_chord_is_not_a_source_deletion_license(self):
        _, record = restore_source_ears(self.mesh, len(self.mesh.faces))
        self.assertFalse(record['applied'])

    def test_two_triangle_source_ear_restores_the_complete_open_path(self):
        vertices = np.vstack((self.mesh.vertices, [.025, -.001, .0005]))
        faces = [[1, 0, 3], [0, 1, 2], [0, 2, 5]] + self.mesh.faces[2:].tolist()
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        result, record = restore_source_ears(mesh, 3)
        self.assertTrue(record['applied'])
        self.assertTrue(result.is_watertight)
        np.testing.assert_array_equal(result.faces[:3], mesh.faces[:3])


if __name__ == '__main__':
    unittest.main()
