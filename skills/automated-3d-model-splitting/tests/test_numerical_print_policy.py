from pathlib import Path
from collections import Counter
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
import trimesh
from split3mf.domain import CapDecision
from split3mf.part_geometry import remap_cap_decision
from split3mf.boundary_correspondence import source_loop_print_match
from split3mf.surface_preservation import audit_replaced_surface, face_key
from split3mf.print_tolerance import PrintTolerance, tolerance_scope
from split3mf.cap_template import fit_affine_cap_inside_parent


class NumericalPrintPolicyTests(unittest.TestCase):
    def test_ordered_source_detour_uses_print_bound_without_moving_rims(self):
        source=np.array([[0,0,0],[2,0,0],[2,2,0],[0,2,0]], float)
        requested=np.insert(source, 1, [1,.02,0], axis=0)
        original=requested.copy()
        decision=CapDecision('flat', (10,11,12,13), source.copy(),
            np.tile([0.,0.,1.], (4,1)), np.ones(4), {'cap_mode':'flat'}, source.copy())
        fit, depth, direction, record=remap_cap_decision(decision, (10,99,11,12,13),
            requested_fit_points=requested, requested_source_points=requested)
        self.assertAlmostEqual(record['cap_boundary_reconciliation']['tolerance_mm'], .05)
        np.testing.assert_allclose(fit[1], [1,0,0])
        np.testing.assert_allclose((fit+depth[:,None]*direction)[:,2], 1)
        np.testing.assert_array_equal(requested, original)
        np.testing.assert_array_equal(decision.source_points, source)
        with tolerance_scope(PrintTolerance(0,0)), self.assertRaises(ValueError):
            remap_cap_decision(decision, (10,99,11,12,13),
                requested_fit_points=requested, requested_source_points=requested)

    def test_correspondence_rejects_reordered_and_distant_loops(self):
        source=np.array([[0,0,0],[2,0,0],[2,2,0],[0,2,0]], float)
        self.assertIsNone(source_loop_print_match(source, source[[0,2,1,3]], .001))
        self.assertIsNone(source_loop_print_match(source, source+[0,0,.051], .001))
        self.assertIsNotNone(source_loop_print_match(source, source[::-1]+[0,0,.02], .001))

    @staticmethod
    def surface_case(z=.0005, duplicate=False, missing=False):
        old=np.array([[0,0,0],[4,0,0],[0,4,0]], float)
        vertices=np.array([[0,0,0],[4,0,0],[0,4,0],[2,0,z]], float)
        faces=np.array([[0,3,2],[3,1,2]], int)
        if duplicate:
            faces[1]=faces[0]
        replacements=vertices[faces]
        mesh=trimesh.Trimesh(vertices=vertices, faces=faces[:1] if missing else faces, process=False)
        mesh.metadata['boundary_closure_subdivisions']=[dict(source_triangle=old.tolist(),
            replacement_triangles=replacements.tolist())]
        return old, mesh

    def test_large_traced_fan_accepts_rounding_without_surface_search(self):
        old, mesh=self.surface_case()
        from unittest.mock import patch
        with patch('split3mf.surface_preservation.cKDTree', side_effect=AssertionError('expensive fallback')):
            report=audit_replaced_surface(Counter({face_key(old):1}),
                Counter(face_key(t) for t in mesh.triangles), mesh)
        self.assertTrue(report['accepted'])
        self.assertEqual(report['equivalent_subdivided_source_faces'],1)
        self.assertLessEqual(report['subdivision_proofs'][0]['maximum_source_error_mm'], .001)

    def test_subdivision_rejects_overlap_missing_faces_and_excess_rounding(self):
        for options in (dict(z=0,duplicate=True), dict(missing=True), dict(z=.002)):
            old, mesh=self.surface_case(**options)
            result=audit_replaced_surface(Counter({face_key(old):1}),
                Counter(face_key(t) for t in mesh.triangles), mesh)
            self.assertFalse(result['accepted'])
        old, mesh=self.surface_case()
        with tolerance_scope(PrintTolerance(0,0)):
            result=audit_replaced_surface(Counter({face_key(old):1}),
                Counter(face_key(t) for t in mesh.triangles), mesh)
        self.assertFalse(result['accepted'])

    def test_cap_solves_oblique_translation_then_remeasures(self):
        fit=np.array([[0,0,0],[1,0,0],[0,1,0]], float)
        template=fit+[.8,0,10]
        original=template.copy()
        calls=[]
        def safety(points, directions, maximum):
            calls.append(directions.copy())
            return 1., {'measured_limit_mm':1.}
        result=fit_affine_cap_inside_parent(template_points=template,
            boundary_indices=np.arange(3), fit_points=fit, inward_axis=np.array([0,0,1.]),
            safety_limit=safety, maximum_depth_mm=10.)
        self.assertEqual(result.attempts,2)
        self.assertEqual(len(calls),2)
        self.assertFalse(np.allclose(calls[0],calls[1]))
        self.assertLessEqual(result.distances.max(),1.)
        self.assertGreaterEqual(result.distances.min(),.08)
        self.assertEqual(result.trace[0]['backoff_method'],'measured_quadratic_interval')
        np.testing.assert_array_equal(template, original)

    def test_cap_cannot_publish_when_fresh_measurement_is_unsafe(self):
        fit=np.array([[0,0,0],[1,0,0],[0,1,0]], float)
        calls=[]
        def safety(*args):
            calls.append(1)
            return (1. if len(calls)==1 else .01), {}
        with self.assertRaises(ValueError):
            fit_affine_cap_inside_parent(template_points=fit+[.8,0,10],
                boundary_indices=np.arange(3), fit_points=fit, inward_axis=np.array([0,0,1.]),
                safety_limit=safety, maximum_depth_mm=10.)
        self.assertGreaterEqual(len(calls),2)


if __name__=='__main__':
    unittest.main()
