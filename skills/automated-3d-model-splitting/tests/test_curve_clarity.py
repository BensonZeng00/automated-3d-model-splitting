from __future__ import annotations
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.curve_clarity import propose_clear_curve, crossings
from split3mf.planar_arc import CurveClarityRequired, fit_stable_plane
from split3mf.boundary_review import BoundaryReviewService, BoundaryReviewRequired
from split3mf.domain import PlanarArcRetopologyConfig, PlanarArcRetopologyContext
from split3mf.interface_retopology import InterfaceRetopologyService

BASIS = (np.zeros(3), np.array([1.,0,0]), np.array([0.,1,0]), np.array([0.,0,1]))


def lobe():
    xy = np.array([[0,0],[10,0],[10,5],[2,5],[3,4.9],[4,5.1],[1,5],[0,5]], dtype=float)
    return np.column_stack((xy,np.zeros(len(xy))))


class CurveClarityTests(unittest.TestCase):
    def test_large_clear_curve_is_not_rejected_by_candidate_work_budget(self):
        angle = np.arange(4100)*2*np.pi/4100
        points = np.column_stack((5*np.cos(angle),4*np.sin(angle),angle*0))
        result = propose_clear_curve(points, *BASIS)
        self.assertEqual(result.record['status'], 'clear_direct')
        np.testing.assert_array_equal(result.points, points)

    def test_clear_concave_contour_is_exact_and_allocates_no_mapping(self):
        points = np.array([[0,0,0],[4,0,0],[4,4,0],[2,2,0],[0,4,0]], float)
        result = propose_clear_curve(points, *BASIS)
        np.testing.assert_array_equal(result.points, points)
        self.assertEqual(result.record['status'], 'clear_direct')
        self.assertEqual(result.record['attempts'], [])
        self.assertIsNone(result.source_weights)

    def test_crossing_lobe_removed_and_mapping_reconstructs_actual_path(self):
        points = lobe()
        result = propose_clear_curve(points, *BASIS)
        self.assertGreater(len(crossings(points[:,:2])), 0)
        self.assertEqual(result.record['status'], 'proposed')
        self.assertEqual(crossings(result.points[:,:2]), [])
        np.testing.assert_allclose(result.source_weights @ points, result.points)
        np.testing.assert_allclose(np.asarray(result.source_weights.sum(axis=1)), 1)
        self.assertTrue(result.record['requires_surface_remesh'])
        self.assertFalse(result.record['source_id_assignment_valid'])
        self.assertTrue(result.record['requires_user_confirmation'])

    def test_source_and_far_anchors_are_unchanged(self):
        points = lobe(); original = points.copy()
        result = propose_clear_curve(points, *BASIS)
        np.testing.assert_array_equal(points, original)
        for anchor in points[:3]:
            self.assertTrue(np.any(np.all(result.points == anchor, axis=1)))

    def test_cyclic_and_reverse_input_produces_same_curve(self):
        points = lobe()
        expected = propose_clear_curve(points, *BASIS)
        for other in (np.roll(points,3,axis=0), points[::-1]):
            actual = propose_clear_curve(other, *BASIS)
            np.testing.assert_allclose(actual.points, expected.points, atol=1e-12)

    def test_depth_disagreement_is_not_silently_welded_into_mesh(self):
        points = lobe(); points[3:6,2] = [2,3,4]
        result = propose_clear_curve(points, *BASIS)
        self.assertGreater(result.record['depth_ambiguous_junctions'], 0)
        self.assertTrue(result.record['requires_user_confirmation'])
        self.assertTrue(result.record['requires_surface_remesh'])

    def test_no_candidate_when_area_budget_is_zero(self):
        result = propose_clear_curve(lobe(), *BASIS, max_removed_area_fraction=0)
        self.assertEqual(result.record['status'], 'unresolved_review')
        self.assertEqual(len(result.record['attempts']), 3)
        np.testing.assert_array_equal(result.points, lobe())

    def test_requested_attempt_count_cannot_exceed_three(self):
        result = propose_clear_curve(lobe(), *BASIS, max_attempts=99, max_removed_area_fraction=0)
        self.assertEqual(len(result.record['attempts']), 3)

    def test_clear_interface_never_invokes_review_callback(self):
        angle = np.arange(96)*2*np.pi/96
        points = np.column_stack((5*np.cos(angle),4*np.sin(angle),angle*0))
        with patch('split3mf.boundary_review.export_curve_review', create=True) as unused:
            context = PlanarArcRetopologyContext(PlanarArcRetopologyConfig(),
                curve_review_sink=lambda _: self.fail('clear curve invoked review'))
            InterfaceRetopologyService.retopologize_loop(points, np.arange(len(points)), context)
            unused.assert_not_called()

    def test_review_callback_stops_before_surface_work_and_old_approval_is_not_reused(self):
        points = lobe(); proposal = propose_clear_curve(points, *BASIS)
        failure = CurveClarityRequired(points, points, proposal, BASIS)
        with tempfile.TemporaryDirectory() as directory:
            service = BoundaryReviewService(Path(directory))
            context = PlanarArcRetopologyContext(PlanarArcRetopologyConfig(), curve_review_sink=service.review_curve)
            with patch('split3mf.interface_retopology.build_planar_arc_boundary', side_effect=failure):
                with self.assertRaises(BoundaryReviewRequired) as raised:
                    InterfaceRetopologyService.retopologize_loop(points, np.arange(len(points)), context)
            self.assertFalse(raised.exception.report['previous_ownership_approval_reused'])
            self.assertTrue((raised.exception.directory/'clear_curve.npz').exists())
            self.assertTrue((raised.exception.directory/'clear_curve_comparison.png').exists())


if __name__ == '__main__':
    unittest.main()
