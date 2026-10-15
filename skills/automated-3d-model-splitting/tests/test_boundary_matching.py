from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split3mf.boundary_matching import distances_to_loop


def test_sparse_source_contour_matches_points_between_its_vertices():
    source_loop = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    dense_points = np.array([
        [0.5, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 2.0, 0.05],
    ])

    np.testing.assert_allclose(distances_to_loop(dense_points, source_loop), [0, 0, 0, 0.05])
