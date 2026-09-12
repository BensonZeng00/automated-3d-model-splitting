from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.annulus_projection import audit_projection
from split3mf.backing_shape import internal_dihedral_report
from split3mf.connector_topology import _audit_ring_strip, triangulate_bounded_ring_strip


class AnnulusProjectionTests(unittest.TestCase):
    def setUp(self):
        with np.load(Path(__file__).parent/'fixtures/concave_strip_fold.npz') as data:
            self.xyz = data['points']
            self.xy = data['projected']
            self.faces = data['faces'].tolist()
            self.outer = data['outer_ids'].tolist()
            self.inner = data['inner_ids'].tolist()
        self.points = dict(enumerate(self.xy))

    def test_folded_strip_fails_despite_correct_boundary_and_bridge_samples(self):
        audit = _audit_ring_strip(
            self.faces, dict(enumerate(self.xyz)), self.points,
            self.outer, self.inner, self.xy[self.outer], self.xy[self.inner],
            maximum_fanout=4)
        self.assertEqual(audit.invalid_bridge_count, 0)
        self.assertEqual(audit.open_edge_count, audit.boundary_edge_count)
        self.assertFalse(audit.valid)
        self.assertIn('projected_fold', audit.reason)
        self.assertEqual(audit.projection['reversed_faces'], 66)

    def test_production_solver_recovers_nonoverlapping_strip_with_same_boundaries(self):
        result = triangulate_bounded_ring_strip(
            self.outer, self.xyz[self.outer], self.inner, self.xyz[self.inner],
            self.xy[self.outer], self.xy[self.inner], maximum_fanout=4)
        self.assertTrue(result.audit.valid)
        self.assertEqual(result.audit.projection['reversed_faces'], 0)
        self.assertEqual(result.audit.projection['crossing_edges'], 0)
        for faces in (result.faces, [tuple(reversed(f)) for f in result.faces]):
            audit = audit_projection(faces, self.points, self.outer, self.inner)
            self.assertTrue(audit.valid)
            self.assertAlmostEqual(audit.absolute_area_mm2, audit.expected_area_mm2)
        np.testing.assert_array_equal(result.inner_points, self.xyz[list(result.inner_ids)])

    def test_within_layer_measurement_catches_old_fold_and_is_winding_independent(self):
        first = internal_dihedral_report(self.xyz, self.faces)
        flipped = np.asarray(self.faces).copy()
        flipped[::2] = flipped[::2, ::-1]
        second = internal_dihedral_report(self.xyz, flipped)
        self.assertEqual(first['checked_edges'], 438)
        self.assertGreater(first['maximum_degrees'], 175)
        self.assertEqual(first['checked_edges'], second['checked_edges'])
        self.assertEqual(first['over_60_degrees'], second['over_60_degrees'])
        self.assertAlmostEqual(first['p95_degrees'], second['p95_degrees'])
        self.assertAlmostEqual(first['maximum_degrees'], second['maximum_degrees'])

    def test_no_samples_is_not_a_zero_degree_pass(self):
        result = internal_dihedral_report(self.xyz, self.faces[:1])
        self.assertEqual(result['status'], 'not_evaluated')
        self.assertIsNone(result['p95_degrees'])
        self.assertIsNone(result['maximum_degrees'])

    def test_source_projection_fold_restores_original_3d_boundary(self):
        with np.load(Path(__file__).parent/'fixtures/source_projection_fold.npz') as data:
            xyz, xy = data['points'], data['projected']
            outer, inner = data['outer_ids'].tolist(), data['inner_ids'].tolist()
        result = triangulate_bounded_ring_strip(
            outer, xyz[outer], inner, xyz[inner], xy[outer], xy[inner], maximum_fanout=4)
        self.assertTrue(result.audit.valid)
        self.assertEqual(result.audit.projection['restored_source_ears'], 1)
        self.assertEqual(result.audit.projection['crossing_edges'], 0)
        np.testing.assert_array_equal(result.inner_points, xyz[list(result.inner_ids)])


if __name__ == '__main__':
    unittest.main()
