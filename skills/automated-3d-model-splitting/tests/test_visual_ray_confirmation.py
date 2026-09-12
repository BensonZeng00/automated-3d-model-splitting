"""Exact visibility distinguishes hidden backing from a real exposed cap."""
import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.surface_rays import first_surface_hit
from split3mf.validation import validate_multiview_visual_consistency
from split3mf.visual_ray_confirmation import confirm_front_intrusions


class VisualRayConfirmationTests(unittest.TestCase):
    def test_hit_identifies_the_front_triangle(self):
        mesh = trimesh.creation.box(extents=[4, 4, 4])
        depth, ids = first_surface_hit(mesh, [[.2, .1, 3], [9, 9, 3]], [0, 0, -1])
        self.assertAlmostEqual(depth[0], 1)
        self.assertGreater(mesh.face_normals[ids[0], 2], .9)
        self.assertTrue(np.isinf(depth[1]))
        self.assertEqual(ids[1], -1)

    def audit(self, elevation):
        source = trimesh.creation.icosphere(subdivisions=2, radius=3)
        insert = trimesh.creation.icosphere(subdivisions=2, radius=1)
        insert.apply_translation([0, 0, elevation])
        return validate_multiview_visual_consistency(
            source.vertices, source.faces, np.full(len(source.faces), 2),
            [{'mesh': insert}, {'mesh': source, 'source_surface_face_count': len(source.faces)}],
            resolution=48, view_count=12, min_coverage_ratio=0,
            max_intrusion_ratio=0, max_material_mismatch_ratio=0,
            max_local_material_mismatch_pixels=0)

    def test_enclosed_backing_is_not_visible(self):
        report = self.audit(1.8)
        self.assertTrue(report['valid'], report['errors'])

    def test_sparse_screen_candidate_hidden_by_triangle_is_dismissed(self):
        source = trimesh.creation.box(extents=[4, 4, 4])
        confirmed, labels, evidence = confirm_front_intrusions(
            source, source, np.array([[.2, .1, 1.8]]), np.full(len(source.faces), 2),
            np.array([0, 0, 1]), .12)
        self.assertFalse(confirmed[0])
        self.assertEqual(labels[0], 2)
        self.assertEqual(evidence['occluded_by_assembly'], 1)

    def test_missing_source_hit_is_not_silently_accepted(self):
        source = trimesh.creation.box()
        exposed = trimesh.creation.box(); exposed.apply_translation([5, 0, 0])
        confirmed, _, evidence = confirm_front_intrusions(
            source, exposed, np.array([[5., 0., .5]]), np.ones(len(source.faces)),
            np.array([0, 0, 1]), .12)
        self.assertTrue(confirmed[0])
        self.assertEqual(evidence['missing_source_hits'], 1)

    def test_real_exposed_cap_remains_blocking(self):
        report = self.audit(2.8)
        self.assertFalse(report['valid'])
        self.assertGreater(report['front_material_mismatch_ratio'], 0)


if __name__ == '__main__':
    unittest.main()
