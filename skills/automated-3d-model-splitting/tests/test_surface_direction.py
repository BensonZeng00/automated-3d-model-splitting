from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.mesh import average_outward_normal
from split3mf.surface_direction import audit_inward_axis


class SurfaceDirectionTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[-1, -1, -2], [1, -1, -2], [1, 1, -2], [-1, 1, -2.]])
        self.faces = [[0, 1, 2], [0, 2, 3]]

    def test_recess_normal_must_not_be_flipped_toward_radial_direction(self):
        for translation in (np.zeros(3), np.array([200., 150., 60.])):
            normal = average_outward_normal(self.vertices+translation, np.asarray(self.faces),
                                            self.vertices.mean(axis=0)+translation, translation)
            np.testing.assert_allclose(normal, [0, 0, 1])

    def test_outward_extrusion_is_blocked_before_geometry_generation(self):
        with self.assertRaisesRegex(ValueError, 'points outside'):
            audit_inward_axis(self.vertices, self.faces, [0, 1, 2, 3], [0, 0, 1])
        record = audit_inward_axis(self.vertices, self.faces, [0, 1, 2, 3], [0, 0, -1])
        self.assertEqual(record['outward_dot_inward'], -1)


if __name__ == '__main__':
    unittest.main()
