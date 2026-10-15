"""Regression for a T-junction walk that doubles back along a seam."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split3mf.application.boundary_snapshot_builder import _remove_short_backtracks
from split3mf.spatial_intersections import count_polyline_self_intersections_3d


def test_short_collinear_return_is_removed_without_moving_source_vertices():
    source = np.array([
        [0., 0., 0.], [1., 0., 0.], [.5, 0., 0.],
        [.5, 1., 0.], [0., 1., 0.],
    ])
    unchanged = source.copy()
    retained = _remove_short_backtracks(source, np.arange(len(source)))
    assert retained.tolist() == [0, 2, 3, 4]
    assert count_polyline_self_intersections_3d(source[retained])[
        "segment_intersection_count"
    ] == 0
    np.testing.assert_array_equal(source, unchanged)


def test_long_return_keeps_contour_for_review():
    source = np.array([
        [0., 0., 0.], [4., 0., 0.], [1., 0., 0.],
        [1., 2., 0.], [0., 2., 0.],
    ])
    retained = _remove_short_backtracks(source, np.arange(len(source)))
    assert retained.tolist() == list(range(len(source)))
