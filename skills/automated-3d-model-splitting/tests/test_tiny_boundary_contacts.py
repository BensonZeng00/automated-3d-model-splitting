"""Tiny point contacts are cleaned before a shared seam is frozen."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split3mf.spatial_intersections import (
    clean_tiny_boundary_contacts_3d,
    count_polyline_self_intersections_3d,
)


def _point_contact(height: float = 0.5) -> np.ndarray:
    return np.array([
        [0., 0., 0.], [4., 0., 0.], [4., 1., 0.], [0., 1., 0.],
        [.2, height, 0.], [0., height, 0.], [.4, height, 0.],
    ])


def test_tiny_contact_preserves_source_point_correspondence():
    points = _point_contact()
    source_ids = np.arange(len(points)) + 100
    cleaned, cleaned_ids, audit = clean_tiny_boundary_contacts_3d(points, source_ids)
    assert audit["removed_crossing_loop_count"] == 1
    assert audit["remaining_intersection_count"] == 0
    assert audit["discarded_perimeter_mm"] < 1.0
    assert audit["maximum_contour_deviation_mm"] < 1.0
    assert len(cleaned) == len(points) - 1
    np.testing.assert_array_equal(cleaned, points[cleaned_ids - 100])
    assert count_polyline_self_intersections_3d(cleaned)["segment_intersection_count"] == 0


def test_multi_vertex_contact_respects_budget_and_source_ids():
    points = np.array([
        [0., 0., 0.], [4., 0., 0.], [4., 1., 0.], [0., 1., 0.],
        [.2, .5, 0.], [0., .5, 0.], [.1, .5, 0.], [.4, .5, 0.],
    ])
    source_ids = np.arange(len(points)) + 100
    cleaned, cleaned_ids, audit = clean_tiny_boundary_contacts_3d(
        points, source_ids, maximum_change_mm=2.0
    )
    assert len(cleaned) == len(points) - 2
    assert audit["remaining_intersection_count"] == 0
    assert audit["discarded_perimeter_mm"] <= 2.0
    assert audit["maximum_contour_deviation_mm"] <= 2.0
    np.testing.assert_array_equal(cleaned, points[cleaned_ids - 100])


def test_contact_outside_one_millimetre_budget_remains_for_diagnosis():
    points = _point_contact()
    cleaned, ids, audit = clean_tiny_boundary_contacts_3d(
        points, np.arange(len(points)), maximum_change_mm=0.1
    )
    np.testing.assert_array_equal(cleaned, points)
    np.testing.assert_array_equal(ids, np.arange(len(points)))
    assert audit["remaining_intersection_count"] == 1


def test_interior_crossing_is_not_silently_cleaned():
    points = np.array([
        [0., 0., 0.], [2., 2., 0.], [0., 2., 0.], [2., 0., 0.],
    ])
    cleaned, _, audit = clean_tiny_boundary_contacts_3d(points, np.arange(len(points)))
    np.testing.assert_array_equal(cleaned, points)
    assert audit["removed_crossing_loop_count"] == 0
    assert audit["remaining_intersection_count"] == 1


def test_frozen_contact_uses_bounded_policy_without_mutating_input():
    from split3mf.admission import audit_frozen_boundary
    points = _point_contact()
    original = points.copy()
    audit = audit_frozen_boundary(points)
    assert audit["tolerated_point_contacts"]["remaining_intersection_count"] == 0
    np.testing.assert_array_equal(points, original)
