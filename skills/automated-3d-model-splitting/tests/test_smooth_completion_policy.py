import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.boundary_budget import BoundaryAreaBudget
from split3mf.boundary_clarity import BoundaryGraph
from split3mf.boundary_review import BoundaryReviewService
from split3mf.cli import build_parser
from split3mf.domain import PlanarArcRetopologyConfig


class SmoothCompletionPolicyTests(unittest.TestCase):
    def budget(self, count=200):
        vertices=np.array([[0.,0,0],[1,0,0],[0,2,0]])
        faces=np.tile([0,1,2], (2*count,1))
        owners=np.array(['a']*count+['b']*count)
        return BoundaryAreaBudget(vertices,faces,owners), owners

    def test_default_and_explicit_smooth_reach_spline_config(self):
        for extra in ([], ['--boundary-shape','Smooth']):
            args=build_parser().parse_args(['--input','model.3mf']+extra)
            self.assertEqual(args.boundary_shape,'smooth')
            config=PlanarArcRetopologyConfig.from_namespace(args)
            self.assertGreater(config.smooth_passes,0)
            self.assertFalse(hasattr(config,'preserve_confirmed_seam'))

    def test_removed_source_cannot_be_selected_or_injected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(['--input','model.3mf','--boundary-shape','source'])
        args=build_parser().parse_args(['--input','model.3mf'])
        args.boundary_shape='source'
        with self.assertRaises(ValueError):
            PlanarArcRetopologyConfig.from_namespace(args)

    def test_strict_threshold_and_cumulative_budget(self):
        budget, owners=self.budget()
        one=owners.copy(); one[0]='b'
        self.assertTrue(budget.evaluate(owners,one,commit=True)['automatic'])
        two=one.copy(); two[1]='b'
        report=budget.evaluate(one,two)
        self.assertFalse(report['automatic'])
        self.assertEqual(report['regions'][0]['affected_area_ratio'],.01)
        self.assertEqual(report['above_threshold_action'],
                         'completion_principle_assessment_not_automatic_rejection')
        self.assertTrue(budget.evaluate(one,one)['automatic'])

    def test_big_body_cannot_hide_loss_of_small_part(self):
        budget, owners=self.budget()
        owners[:199]='b'
        budget=BoundaryAreaBudget(np.array([[0.,0,0],[1,0,0],[0,2,0]]),
                                 np.tile([0,1,2],(400,1)),owners)
        result=owners.copy(); result[199]='b'
        report=budget.evaluate(owners,result)
        self.assertFalse(report['automatic'])
        self.assertFalse(report['identities_preserved'])

    def test_small_branched_region_is_repaired_without_prompt_or_repaint(self):
        mesh=trimesh.creation.icosphere(subdivisions=3)
        graph=BoundaryGraph(mesh.vertices,mesh.faces)
        labels=np.where(mesh.triangles_center[:,0]>0,'a','b')
        for face in range(len(labels)):
            trial=labels.copy(); trial[face]='b' if labels[face]=='a' else 'a'
            assessment=graph.assess(trial)
            if not assessment.clear:
                candidates=graph.propose(trial,assessment)
                if any(c['admissible'] for c in candidates):
                    break
        else:
            self.fail('No branched fixture')
        vertices,faces=mesh.vertices.copy(),mesh.faces.copy()
        with tempfile.TemporaryDirectory() as folder:
            with patch('split3mf.boundary_preview.render_boundary_review',
                       side_effect=AssertionError('Small anomaly must not prompt')):
                result,record=BoundaryReviewService(Path(folder)).prepare(vertices,faces,trial)
        self.assertTrue(graph.assess(result).clear)
        self.assertEqual(record['status'],'automatic_local_merge')
        self.assertTrue(record['area_policy']['automatic'])
        np.testing.assert_array_equal(vertices,mesh.vertices)
        np.testing.assert_array_equal(faces,mesh.faces)
        self.assertEqual(set(result),set(trial))


if __name__=='__main__':
    unittest.main()
