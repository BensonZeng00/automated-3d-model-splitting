import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.curved_backing import first_exit
from split3mf.local_ray_probe import LocalRayProbe
from split3mf.parent_ray_probe import parent_exit_distances, ParentSafetyError


class RecoveryRayTests(unittest.TestCase):
    def test_segmented_and_native_exits_match_exhaustive_triangles(self):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=3.)
        rng = np.random.default_rng(812)
        points = rng.uniform(-4, 4, (45, 3))
        directions = rng.normal(size=points.shape)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        reference = np.array([first_exit(mesh.triangles, [p], axis)[0]
                              for p, axis in zip(points, directions)])
        probe = LocalRayProbe(mesh)
        for bound in (.2, 3., 130.):
            expected = reference.copy()
            expected[expected > bound] = np.nan
            np.testing.assert_allclose(probe.exits(points, directions, bound)[0],
                                       expected, atol=1e-9, rtol=0, equal_nan=True)
            np.testing.assert_allclose(parent_exit_distances(mesh, points, directions, bound),
                                       expected, atol=1e-9, rtol=0, equal_nan=True)

    def test_failure_records_first_sample_and_partial_coverage(self):
        with self.assertRaises(ParentSafetyError) as caught:
            parent_exit_distances(trimesh.creation.box(), [[2,0,0], [0,0,0]],
                                  [1,0,0], 10., minimum_reserve_mm=.05)
        record = caught.exception.record
        self.assertIsNone(record['exit_mm'])
        self.assertEqual(record['tested_samples'], 1)
        self.assertEqual(record['total_samples'], 2)
        self.assertFalse(record['complete'])
        self.assertEqual(record['point_mm'], [2.,0.,0.])

    def test_reserve_is_not_relaxed(self):
        with self.assertRaises(ParentSafetyError) as caught:
            parent_exit_distances(trimesh.creation.box(), [[.49,0,0]], [1,0,0],
                                  1., minimum_reserve_mm=.05)
        self.assertAlmostEqual(caught.exception.record['exit_mm'], .01)

    def test_older_kernel_uses_exact_fallback(self):
        kernel = SimpleNamespace(status=lambda:'Error.NoError', is_empty=lambda:False)
        with patch('split3mf.parent_ray_probe._manifold64', return_value=kernel):
            np.testing.assert_allclose(parent_exit_distances(
                trimesh.creation.box(), [[0,0,0]], [1,0,0], 2.), [.5])

    def test_invalid_parameters_and_open_parent_are_rejected(self):
        mesh = trimesh.creation.box()
        for axis in ([0,0,0], [float('nan'),0,1]):
            with self.assertRaises(ValueError):
                parent_exit_distances(mesh, [[0,0,0]], axis, 2.)
        mesh.update_faces(np.arange(11))
        with self.assertRaises(ValueError):
            parent_exit_distances(mesh, [[0,0,0]], [0,0,1], 2.)


if __name__ == '__main__':
    unittest.main()
