import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split3mf.boolean_cutters import (
    _area_weighted_source_vertex_normals,
)


class BooleanCutterSourceNormalTests(unittest.TestCase):
    def test_recovers_one_isolated_area_weighted_cancellation(self):
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 3, 4]], dtype=np.int64)

        vertex_ids, directions, record = _area_weighted_source_vertex_normals(
            vertices,
            faces,
        )

        center = int(np.flatnonzero(vertex_ids == 0)[0])
        self.assertTrue(np.all(np.isfinite(directions)))
        self.assertAlmostEqual(float(np.linalg.norm(directions[center])), 1.0)
        self.assertEqual(record["recovered_undefined_normal_count"], 1)
        self.assertEqual(record["recovered_undefined_normal_vertex_samples"], [0])

    def test_keeps_clustered_undefined_normals_blocking(self):
        vertices = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "reason=clustered"):
            _area_weighted_source_vertex_normals(vertices, faces)


if __name__ == "__main__":
    unittest.main()


