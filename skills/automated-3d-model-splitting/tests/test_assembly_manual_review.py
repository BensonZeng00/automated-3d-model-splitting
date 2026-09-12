"""Manual assembly handoff retains valid geometry and complete fit evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.assembly_case import save_assembly_case, load_assembly_case
from split3mf.assembly_visibility import validate_and_seat_assembly
from split3mf.assembly_seating import SeatingError
from split3mf.assembly_review import annotate_manual_adjustment


class AssemblyManualReviewTests(unittest.TestCase):
    def setUp(self):
        self.parts = []
        for key, parent, x in [('P01', None, 0), ('P02', 'P01', 1.9), ('P03', 'P02', 3.8)]:
            mesh = trimesh.creation.box(extents=[2, 2, 2])
            mesh.apply_translation([x, 0, 0])
            self.parts.append(dict(part_id=key, mesh=mesh, color_hex='#FF0000',
                face_filament_slot_indices=[1] * len(mesh.faces),
                annotation=dict(parent_part=parent,
                                selected_processing_mode='body' if parent is None else 'inward')))
        source = trimesh.util.concatenate([p['mesh'] for p in self.parts])
        self.source = (source.vertices, source.faces, np.repeat([1, 2, 3], 12))

    def test_manual_review_measures_all_pairs_without_committing_failed_motion(self):
        before = [p['mesh'].vertices.copy() for p in self.parts]
        result = validate_and_seat_assembly(self.parts, *self.source, allow_manual_adjustment=True,
                                             ignore_overlap_below_mm3=0)
        self.assertFalse(result['valid'])
        self.assertTrue(result['manual_adjustment_required'])
        self.assertEqual(len(result['final_pairs']), 3)
        self.assertEqual(sum(not p['accepted'] for p in result['final_pairs']), 2)
        self.assertIn('collision', result['failure']['detail'])
        for part, vertices in zip(self.parts, before):
            np.testing.assert_array_equal(part['mesh'].vertices, vertices)
            self.assertEqual(part['annotation']['assembly_fit']['status'], 'manual_adjustment_required')

    def test_strict_mode_still_raises_measured_seating_error(self):
        with self.assertRaises(SeatingError) as caught:
            validate_and_seat_assembly(self.parts, *self.source, ignore_overlap_below_mm3=0)
        self.assertGreater(caught.exception.record['intersection_before_mm3'], 0)

    def test_manual_policy_does_not_hide_non_fit_runtime_errors(self):
        with patch('split3mf.assembly_visibility._validate_and_seat', side_effect=ValueError('bad source')):
            with self.assertRaisesRegex(ValueError, 'bad source'):
                validate_and_seat_assembly(self.parts, *self.source, allow_manual_adjustment=True)

    def test_visual_failure_remains_false_and_is_recorded_in_part_metadata(self):
        visual = dict(valid=False, errors=['front-material mismatch'], coverage_ratio=.8)
        self.assertTrue(annotate_manual_adjustment(self.parts, dict(valid=True), visual))
        self.assertFalse(visual['valid'])
        self.assertEqual(self.parts[0]['annotation']['assembly_fit']['visual_errors'], visual['errors'])

    def test_scaled_case_round_trip_preserves_source_colors_and_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            save_assembly_case(directory, self.parts, *self.source, dict(post_fit_difference=True))
            parts, source, options = load_assembly_case(directory)
            self.assertTrue(options['post_fit_difference'])
            for actual, expected in zip(source, self.source):
                np.testing.assert_array_equal(actual, expected)
            for actual, expected in zip(parts, self.parts):
                np.testing.assert_array_equal(actual['mesh'].vertices, expected['mesh'].vertices)
                self.assertEqual(actual['annotation'], expected['annotation'])
                self.assertEqual(actual['face_filament_slot_indices'], expected['face_filament_slot_indices'])
            path = Path(directory) / 'scaled_inputs.npz'
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_assembly_case(directory)


if __name__ == '__main__':
    unittest.main()
