from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common

common.load_core_dependencies()

from split3mf.mesh import fit_orientation_preserving_affine


class OrientationPreservingAffineTests(unittest.TestCase):
    def test_planar_boundary_fit_keeps_unobserved_normal_orientation(self) -> None:
        source = np.asarray(
            [
                [-2.0, -1.0, 0.0],
                [2.0, -1.0, 0.0],
                [2.0, 1.0, 0.0],
                [-2.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        target = source.copy()
        target[:, :2] *= 0.8
        target[:, 2] = -3.0

        matrix, record = fit_orientation_preserving_affine(source, target)
        transformed = np.column_stack((source, np.ones(len(source)))) @ matrix

        self.assertGreater(float(np.linalg.det(matrix[:3, :])), 0.0)
        self.assertTrue(np.allclose(transformed, target, atol=1e-8))
        self.assertGreater(record["relative_regularization"], 0.0)

    def test_nonplanar_orientation_preserving_fit_remains_exact(self) -> None:
        source = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        linear = np.asarray(
            [[0.9, 0.1, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 0.7]],
            dtype=np.float64,
        )
        target = source @ linear + np.asarray([2.0, -1.0, 0.5])

        matrix, record = fit_orientation_preserving_affine(source, target)
        transformed = np.column_stack((source, np.ones(len(source)))) @ matrix

        self.assertTrue(np.allclose(transformed, target, atol=1e-8))
        self.assertGreater(float(np.linalg.det(matrix[:3, :])), 0.0)
        self.assertEqual(record["relative_regularization"], 0.0)


if __name__ == "__main__":
    unittest.main()
