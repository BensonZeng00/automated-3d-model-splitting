from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.region_review import strip_evidence, select_region_review_groups
from split3mf.print_tolerance import small_patch_report, tolerance_scope, PrintTolerance
from split3mf.connector_surface import refine_connector_annulus_heightfield
from split3mf.recognition import classify_review_groups_semantically
from split3mf.cli import build_parser
from split3mf.stage_cache import normalized_run_arguments


def strip_mesh(segments):
    vertices = np.array([[x, y, 0] for x in np.linspace(0, 40, segments+1)
                         for y in (0, .5)])
    faces = np.array([[2*i, 2*i+2, 2*i+1] for i in range(segments)]
                     + [[2*i+1, 2*i+2, 2*i+3] for i in range(segments)])
    return vertices, faces


class PrintPriorityTests(unittest.TestCase):
    def test_refinement_mode_changes_cache_identity(self):
        parser = build_parser()
        production = parser.parse_args(['--input', 'example.3mf'])
        diagnostic = parser.parse_args(['--input', 'example.3mf',
                                        '--hidden-surface-refinement', 'refine'])
        self.assertEqual(production.hidden_surface_refinement, 'preserve')
        self.assertNotEqual(normalized_run_arguments(production),
                            normalized_run_arguments(diagnostic))

    def test_strip_review_independent_of_tessellation(self):
        for count in (1, 600):
            v, f = strip_mesh(count)
            evidence = strip_evidence(v, f, np.arange(len(f)))
            self.assertTrue(evidence['long_thin_candidate'])
            self.assertAlmostEqual(evidence['visible_area_mm2'], 20)
            ordinary, review = select_region_review_groups(v, f, [np.arange(len(f))], 100, 999)
            self.assertEqual((len(ordinary), len(review)), (0, 1))

    def test_hidden_faces_do_not_inflate_strip_area(self):
        v, f = strip_mesh(4)
        visible = np.arange(len(f)) % 2 == 0
        evidence = strip_evidence(v, f, np.arange(len(f)), visible)
        self.assertAlmostEqual(evidence['visible_area_mm2'], 10)
        self.assertFalse(strip_evidence(v, f, np.arange(len(f)), np.zeros(len(f), bool))['long_thin_candidate'])
        v[:, 1] *= 30
        self.assertFalse(strip_evidence(v, f, np.arange(len(f)))['long_thin_candidate'])

    def test_disconnected_patches_keep_single_area_budget(self):
        v = np.array([[0,0,0],[1,0,0],[0,.02,0],[70,0,0],[71,0,0],[70,.02,0]])
        f = [[0,1,2],[3,4,5]]
        result = small_patch_report(v, f, maximum_span_mm=5, separate_components=True)
        self.assertTrue(result['accepted'])
        self.assertGreater(result['affected_span_mm'], 70)
        self.assertLess(result['checked_span_mm'], 2)
        with tolerance_scope(PrintTolerance(.015)):
            self.assertFalse(small_patch_report(v, f, maximum_span_mm=5, separate_components=True)['accepted'])
        self.assertFalse(small_patch_report(v, [[0,4,2]], maximum_span_mm=5, separate_components=True)['accepted'])

    def test_no_candidates_skip_semantic_geometry(self):
        with patch('split3mf.recognition.triangle_areas', side_effect=AssertionError('unnecessary scan')):
            self.assertEqual(classify_review_groups_semantically(
                np.empty((0,3)), np.empty((0,3), int), [], [], [], [], 1000), [])

    def test_print_preservation_keeps_audited_geometry_exactly(self):
        v = [[-3,-3,0],[3,-3,0],[3,3,0],[-3,3,0],
             [-1,-1,-2],[1,-1,-2],[1,1,-2],[-1,1,-2]]
        f = []
        for i in range(4):
            j=(i+1)%4
            f.extend([[i,j,4+i],[j,4+j,4+i]])
        before_v=np.array(v); before_f=np.array(f)
        plan=dict(center=np.zeros(3), inward=np.array([0,0,-1]),
                  u=np.array([1,0,0]), v=np.array([0,1,0]))
        with patch('split3mf.connector_surface._regularize_unequal_ring_topology',
                   side_effect=AssertionError('hidden refinement must not run')):
            result=refine_connector_annulus_heightfield(output_vertices=v, output_faces=f,
                face_start=0, outer_ids=list(range(4)), inner_ids=list(range(4,8)),
                boundary_points=before_v[:4], plan=plan, taper_depth_mm=2,
                preserve_audited_surface=True)
        self.assertEqual(result.passes, 0)
        self.assertEqual(result.inserted_vertices, 0)
        self.assertEqual(result.skipped_reason, 'print_preserved_audited_hidden_surface')
        np.testing.assert_array_equal(v, before_v)
        np.testing.assert_array_equal(f, before_f)


if __name__ == '__main__':
    unittest.main()
