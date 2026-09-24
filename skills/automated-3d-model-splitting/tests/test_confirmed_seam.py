from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from split3mf import common
common.load_core_dependencies()
from split3mf.confirmed_seam import preserve_confirmed_loop, audit_unchanged_surface
from split3mf.domain import PlanarArcRetopologyConfig, PlanarArcRetopologyContext
from split3mf.interface_retopology import InterfaceRetopologyService


class ConfirmedSeamTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0., 0, 0], [2., 0, .1], [2., 2, 0], [0., 2, -.1]])
        self.faces = np.array([[0, 1, 2], [0, 2, 3]])
        self.context = PlanarArcRetopologyContext(PlanarArcRetopologyConfig(preserve_confirmed_seam=True))

    def test_default_is_unchanged(self):
        self.assertFalse(PlanarArcRetopologyConfig().preserve_confirmed_seam)

    def test_exact_points_and_parent_reversal(self):
        first, record = preserve_confirmed_loop(self.points, np.arange(4))
        second, _ = preserve_confirmed_loop(self.points[::-1], np.arange(4)[::-1])
        np.testing.assert_array_equal(first, self.points)
        np.testing.assert_array_equal(first, second[::-1])
        self.assertEqual(record['maximum_target_offset_mm'], 0)
        self.assertFalse(np.shares_memory(first, self.points))

    def test_no_spline_or_surface_repair_on_confirmed_source(self):
        original_faces = self.faces.copy()
        with patch('split3mf.interface_retopology.build_planar_arc_boundary', side_effect=AssertionError('spline')):
            with patch('split3mf.interface_retopology._surface_band_deformation', side_effect=AssertionError('deformation')):
                result, records = InterfaceRetopologyService.retopologize_local_loops(
                    self.points, [[0, 1, 2, 3]], np.arange(4), self.context, self.faces)
        np.testing.assert_array_equal(result, self.points)
        np.testing.assert_array_equal(self.faces, original_faces)
        self.assertTrue(records[0]['visible_top_source_preserved'])

    def test_invalid_source_winding_still_blocks(self):
        with self.assertRaisesRegex(ValueError, 'topology'):
            audit_unchanged_surface(self.points, [[0, 1, 2], [0, 3, 2]], [[0, 1, 2, 3]])

    def test_repeated_ids_and_nonfinite_values_still_block(self):
        with self.assertRaises(ValueError):
            preserve_confirmed_loop(self.points, [0, 1, 2, 2])
        invalid = self.points.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            preserve_confirmed_loop(invalid, np.arange(4))


if __name__ == '__main__':
    unittest.main()
