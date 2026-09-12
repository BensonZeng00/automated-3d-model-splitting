"""Regression tests for subtract-first, whole-insert uniform scaling."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.cli import build_parser
from split3mf.uniform_fit import configure_uniform_fit, exact_unscaled_cutter, scale_finished_insert


class UniformFitTests(unittest.TestCase):
    def test_only_fit_policy_disables_all_old_clearances(self):
        args = build_parser().parse_args(['--input', 'example.3mf'])
        configure_uniform_fit(args)
        self.assertEqual(args.post_split_uniform_scale, .99)
        self.assertEqual(args.clearance_profile, 'fixed')
        for key in ['fit_clearance_mm', 'bottom_clearance_mm', 'flat_clearance_mm',
                    'clearance_min_mm', 'sibling_clearance_mm']:
            self.assertEqual(getattr(args, key), 0)

    def test_old_fit_switches_removed(self):
        parser = build_parser()
        options = parser._option_string_actions
        for key in ['--fit-clearance-mm', '--clearance-mode', '--bottom-clearance-mm']:
            self.assertNotIn(key, options)
        self.assertIn('--post-split-uniform-scale', parser.format_help())

    def test_scale_preserves_own_center_and_triangles(self):
        mesh = trimesh.creation.box(extents=[10, 12, 8])
        mesh.apply_translation([140, 135, 18])
        before = mesh.vertices.copy()
        scaled, record = scale_finished_insert(mesh, .99)
        np.testing.assert_allclose(scaled.bounds.mean(axis=0), mesh.bounds.mean(axis=0))
        np.testing.assert_allclose(scaled.extents, mesh.extents * .99)
        np.testing.assert_array_equal(scaled.faces, mesh.faces)
        np.testing.assert_array_equal(mesh.vertices, before)
        self.assertTrue(record['visible_surface_scaled'])
        self.assertTrue(scaled.is_watertight)
        self.assertAlmostEqual(scaled.volume / mesh.volume, .99**3)

    def test_cutter_is_unscaled_independent_copy(self):
        mesh = trimesh.creation.box()
        cutter, record = exact_unscaled_cutter(mesh)
        np.testing.assert_array_equal(cutter.vertices, mesh.vertices)
        np.testing.assert_array_equal(cutter.faces, mesh.faces)
        self.assertEqual(record['exterior_overshoot_mm'], 0)
        cutter.vertices += 4
        self.assertFalse(np.array_equal(cutter.vertices, mesh.vertices))

    def test_subtraction_precedes_scaling(self):
        stock = trimesh.creation.box(extents=[12, 12, 6])
        child = trimesh.creation.box(extents=[4, 4, 4])
        child.apply_translation([0, 0, 3])
        cutter, _ = exact_unscaled_cutter(child)
        parent = trimesh.boolean.difference([stock, cutter], engine='manifold')
        parent_before = parent.vertices.copy()
        scaled, _ = scale_finished_insert(child, .99)
        np.testing.assert_array_equal(parent.vertices, parent_before)
        self.assertAlmostEqual(parent.volume, stock.volume - 32, places=5)
        self.assertTrue(parent.is_watertight)
        overlap = trimesh.boolean.intersection([parent, scaled], engine='manifold')
        self.assertLess(abs(overlap.volume), 1e-8)

    def test_bad_factors_rejected(self):
        for factor in [0, -1, 1, 1.1, float('nan'), float('inf')]:
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                scale_finished_insert(trimesh.creation.box(), factor)

    def test_optional_cleanup_failure_preserves_validated_difference(self):
        from split3mf import common
        common.load_core_dependencies()
        from split3mf.local_connectors import subtract_socket_cutters
        stock = trimesh.creation.box(extents=[12, 12, 6])
        cutter = trimesh.creation.box(extents=[4, 4, 4])
        cutter.apply_translation([0, 0, 3])

        def invalid_optional_cleanup(candidate, previous):
            invalid = candidate.copy()
            invalid.apply_scale(1.1)
            return invalid, dict(applied=True, removed_component_count=1,
                                 removed_face_count=1, removed_volume_mm3=1e-8)

        with patch('split3mf.local_connectors.remove_new_redundant_boolean_micro_shells',
                   side_effect=invalid_optional_cleanup):
            result, record = subtract_socket_cutters(stock, [cutter])
        self.assertTrue(result.is_watertight)
        self.assertAlmostEqual(result.volume, stock.volume - 32, places=5)


if __name__ == '__main__':
    unittest.main()
