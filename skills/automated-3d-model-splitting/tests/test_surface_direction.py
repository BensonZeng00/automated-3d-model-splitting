from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.mesh import average_outward_normal
from split3mf.surface_direction import audit_inward_axis, resolve_inward_axis


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

    def test_coherent_source_rim_flips_component_axis_before_planning(self):
        resolved, record = resolve_inward_axis(
            self.vertices, self.faces, [0, 1, 2, 3], [0, 0, 1]
        )

        np.testing.assert_allclose(resolved, [0, 0, -1])
        self.assertTrue(record['axis_flipped'])
        self.assertEqual(record['resolution'], 'flipped_by_coherent_source_rim')
        self.assertEqual(record['proposed_outward_dot_inward'], 1)
        self.assertEqual(record['resolved_outward_dot_inward'], -1)
        self.assertEqual(
            audit_inward_axis(self.vertices, self.faces, [0, 1, 2, 3], resolved)[
                'outward_dot_inward'
            ],
            -1,
        )

    def test_each_oppositely_oriented_interface_resolves_independently(self):
        opposite_faces = [[0, 2, 1], [0, 3, 2]]
        first, first_record = resolve_inward_axis(
            self.vertices, self.faces, [0, 1, 2, 3], [0, 0, 1]
        )
        second, second_record = resolve_inward_axis(
            self.vertices, opposite_faces, [0, 1, 2, 3], [0, 0, 1]
        )

        np.testing.assert_allclose(first, [0, 0, -1])
        np.testing.assert_allclose(second, [0, 0, 1])
        self.assertTrue(first_record['axis_flipped'])
        self.assertFalse(second_record['axis_flipped'])

    def test_missing_source_rim_keeps_component_fallback(self):
        resolved, record = resolve_inward_axis(
            self.vertices, self.faces, [10, 11, 12], [0, 1, 0]
        )

        np.testing.assert_allclose(resolved, [0, 1, 0])
        self.assertFalse(record['axis_flipped'])
        self.assertEqual(record['resolution'], 'kept_component_fallback')


if __name__ == '__main__':
    unittest.main()
