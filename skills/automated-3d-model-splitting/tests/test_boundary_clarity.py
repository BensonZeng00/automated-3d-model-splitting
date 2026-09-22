from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf import common
common.load_core_dependencies()
from split3mf.boundary_clarity import BoundaryGraph
from split3mf.boundary_review import (
    BoundaryReviewService, BoundaryReviewRequired, BoundaryDecisionError,
    owners_from_components, apply_component_ownership,
)
from split3mf.recognition import make_component_from_global_faces, triangle_areas
from split3mf.cli import build_parser


class BoundaryClarityTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.icosphere(subdivisions=1)
        self.graph = BoundaryGraph(self.mesh.vertices, self.mesh.faces)
        self.labels = np.where(self.mesh.triangles_center[:, 0] > 0, 'A', 'B')

    def test_clear_fast_path_does_not_search_or_render(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'absent'
            with patch.object(BoundaryGraph, 'propose', side_effect=AssertionError('slow path')):
                owners, report = BoundaryReviewService(destination).prepare(
                    self.mesh.vertices, self.mesh.faces, self.labels)
            self.assertEqual(report['status'], 'clear')
            np.testing.assert_array_equal(owners, self.labels)
            self.assertFalse(destination.exists())

    def test_three_way_junction_is_valid(self):
        centers = self.mesh.triangles_center
        labels = np.where(centers[:, 0] > 0, 'A', np.where(centers[:, 1] > 0, 'B', 'C'))
        self.assertTrue(self.graph.assess(labels).clear)

    def ambiguous_labels(self):
        labels = self.labels.copy()
        # Deterministically find a single-face misassignment producing a seam branch.
        for index in range(len(labels)):
            trial = labels.copy()
            trial[index] = 'B' if trial[index] == 'A' else 'A'
            if not self.graph.assess(trial).clear:
                return trial
        self.fail('Fixture did not produce a branched seam')

    def test_removed_source_branch_option_is_rejected(self):
        with self.assertRaises(TypeError):
            BoundaryReviewService(Path('unused'), preserve_source_branches=True)

    def test_local_proposals_keep_anchors_and_geometry(self):
        labels = self.ambiguous_labels()
        vertices, faces = self.mesh.vertices.copy(), self.mesh.faces.copy()
        assessment = self.graph.assess(labels)
        candidates = self.graph.propose(labels, assessment)
        self.assertLessEqual(len(candidates), 3)
        self.assertGreater(len(candidates), 0)
        for candidate in candidates:
            self.assertTrue(set(candidate['changed_faces']).issubset(assessment.uncertain_faces))
            self.assertEqual(set(candidate['owners']), set(labels))
        np.testing.assert_array_equal(self.mesh.vertices, vertices)
        np.testing.assert_array_equal(self.mesh.faces, faces)
        np.testing.assert_array_equal(labels, self.ambiguous_labels())

    def test_explicit_review_decision_requires_matching_approval(self):
        labels = self.ambiguous_labels()
        with tempfile.TemporaryDirectory() as directory:
            candidate = next(c for c in self.graph.propose(labels,self.graph.assess(labels)) if c['admissible'])
            path = Path(directory) / 'approval.json'
            decision = dict(fingerprint=self.graph.fingerprint(labels,'input'), user_confirmed=True,
                            selected_candidate=candidate['id'],
                            candidate_fingerprint=self.graph.fingerprint(candidate['owners'],'input'))
            path.write_text(json.dumps({'decisions': [decision]}), encoding='utf-8')
            approved, result = BoundaryReviewService(Path(directory), path).prepare(
                self.mesh.vertices, self.mesh.faces, labels)
            self.assertEqual(result['status'], 'user_confirmed')
            self.assertTrue(self.graph.assess(approved).clear)
            decision['candidate_fingerprint'] = 'stale'
            path.write_text(json.dumps(decision), encoding='utf-8')
            with self.assertRaises(BoundaryDecisionError):
                BoundaryReviewService(Path(directory), path).prepare(self.mesh.vertices, self.mesh.faces, labels)
            decision['user_confirmed'] = False
            path.write_text(json.dumps(decision), encoding='utf-8')
            with self.assertRaises(BoundaryDecisionError):
                BoundaryReviewService(Path(directory), path).prepare(self.mesh.vertices, self.mesh.faces, labels)

    def test_above_one_percent_uses_impact_assessment_and_continues(self):
        labels = self.ambiguous_labels()
        with tempfile.TemporaryDirectory() as directory:
            actual, report = BoundaryReviewService(Path(directory)).prepare(
                self.mesh.vertices, self.mesh.faces, labels)
        self.assertEqual(report['status'],'completion_priority_local_merge')
        self.assertFalse(report['area_policy']['automatic'])
        self.assertTrue(report['completion_assessment']['exterior_unchanged'])
        self.assertTrue(self.graph.assess(actual).clear)

    def test_no_candidate_still_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(BoundaryGraph, 'propose', return_value=[]) as search:
                with patch('split3mf.boundary_preview.render_boundary_review', return_value=[]):
                    with self.assertRaises(BoundaryReviewRequired):
                        BoundaryReviewService(Path(directory)).prepare(
                            self.mesh.vertices, self.mesh.faces, self.ambiguous_labels())
            self.assertEqual(search.call_args.kwargs['max_candidates'], 3)

    def test_component_reassignment_does_not_repaint_or_mutate_components(self):
        areas = triangle_areas(self.mesh.vertices, self.mesh.faces)
        components = [make_component_from_global_faces(self.mesh.vertices, self.mesh.faces,
                       areas, np.flatnonzero(self.labels == label), label) for label in ('A', 'B')]
        owners = owners_from_components(len(self.mesh.faces), components)
        changed = owners.copy()
        changed[0] = '2' if owners[0] == '1' else '1'
        original_faces = [c.global_faces.copy() for c in components]
        paint = self.labels.copy()
        result = apply_component_ownership(self.mesh.vertices, self.mesh.faces, components, changed)
        np.testing.assert_array_equal(owners_from_components(len(self.mesh.faces), result), changed)
        for component, ids in zip(components, original_faces):
            np.testing.assert_array_equal(component.global_faces, ids)
        np.testing.assert_array_equal(paint, self.labels)
        changed[0] = 'unknown'
        with self.assertRaises(ValueError):
            apply_component_ownership(self.mesh.vertices, self.mesh.faces, components, changed)

    def test_cli_defaults_to_conditional_checks(self):
        args = build_parser().parse_args(['--input', 'pending.3mf'])
        self.assertIsNone(args.boundary_review_json)
        self.assertFalse(args.boundary_check_only)

    def test_clear_input_pipeline_continues_to_original_recognition(self):
        from split3mf.domain import SplitConfig
        from split3mf.pipeline import SplitPipeline
        class RecognitionReached(Exception):
            pass
        saved_info, saved_order = dict(common.COLOR_INFO), dict(common.COLOR_ORDER)
        self.addCleanup(common.COLOR_CATALOG.replace, saved_info, saved_order)
        with tempfile.TemporaryDirectory() as directory:
            parser = build_parser()
            args = parser.parse_args(['--input', str(Path(directory)/'synthetic.3mf'),
                                      '--output', str(Path(directory)/'output.3mf')])
            pipeline = SplitPipeline(SplitConfig(args, Path(args.input), {}), parser)
            paint = np.where(self.labels == 'A', '0C', '8').tolist()
            with patch.object(pipeline.reader, 'read', return_value=(
                    self.mesh.vertices, self.mesh.faces, paint, {})):
                with patch.object(BoundaryGraph, 'propose', side_effect=AssertionError('slow path')):
                    with patch('split3mf.pipeline.exterior_visible_face_mask', side_effect=RecognitionReached):
                        with self.assertRaises(RecognitionReached):
                            pipeline.run()
            self.assertFalse((Path(directory)/'output_boundary_review').exists())


if __name__ == '__main__':
    unittest.main()
