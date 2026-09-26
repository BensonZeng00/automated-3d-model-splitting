"""Per-pair physical tolerance must not weaken geometric Boolean audits."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.assembly_overlap import measure_overlap, union_mesh
from split3mf.assembly_review import measure_manual_adjustment, annotate_manual_adjustment
from split3mf.assembly_seating import seat_insert_outward
from split3mf.assembly_visibility import validate_and_seat_assembly
from split3mf.overlap_policy import overlap_is_ignored
from split3mf.post_fit_difference import subtract_parent


def box_pair(volume):
    first = trimesh.creation.box(extents=[2, 2, 2])
    second = first.copy()
    second.apply_translation([2 - volume / 4, 0, 0])
    return first, second


def assembly(volume):
    parts = [dict(part_id=key, mesh=mesh, annotation=dict(parent_part=parent,
                  selected_processing_mode='inward' if parent else 'body'))
             for key, mesh, parent in zip(['P01', 'P02'], box_pair(volume), [None, 'P01'])]
    source = trimesh.util.concatenate([p['mesh'] for p in parts])
    return parts, (source.vertices, source.faces, np.repeat([1, 2], 12))


class OverlapVolumePolicyTests(unittest.TestCase):
    def test_strict_boundary_and_invalid_thresholds(self):
        self.assertTrue(overlap_is_ignored(.999999, 1))
        self.assertFalse(overlap_is_ignored(np.nextafter(1., 0.), 1))
        self.assertFalse(overlap_is_ignored(1., 1))
        for value in [-1, float('nan'), float('inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                overlap_is_ignored(.5, value)

    def test_real_intersections_below_equal_and_above_limit(self):
        for volume, expected in [(.999, True), (1., False), (1.001, False)]:
            with self.subTest(volume=volume):
                result = measure_overlap(*box_pair(volume), ignore_overlap_below_mm3=1)
                self.assertAlmostEqual(result['intersection_mm3'], volume)
                self.assertEqual(result['accepted'], expected)
        self.assertFalse(measure_overlap(*box_pair(.5), ignore_overlap_below_mm3=0)['accepted'])

    def test_disconnected_intersections_are_summed_for_one_pair(self):
        left, right = box_pair(.6)
        left2, right2 = left.copy(), right.copy()
        left2.apply_translation([0, 5, 0])
        right2.apply_translation([0, 5, 0])
        result = measure_overlap(trimesh.util.concatenate([left, left2]),
                                 trimesh.util.concatenate([right, right2]),
                                 ignore_overlap_below_mm3=1)
        self.assertAlmostEqual(result['intersection_mm3'], 1.2)
        self.assertFalse(result['accepted'])

    def test_difference_keeps_ignored_geometry_and_cuts_at_threshold(self):
        for volume, changed in [(.5, False), (1., True), (1.5, True)]:
            child, parent = box_pair(volume)
            original = dict(part_id='P02', mesh=child, annotation={})
            result, record = subtract_parent(original, parent, ignore_overlap_below_mm3=1)
            self.assertEqual(record['changed'], changed)
            if changed:
                self.assertLessEqual(record['residual_intersection_mm3'], 1e-8)
            else:
                self.assertIs(result, original)
                self.assertEqual(record['removed_volume_mm3'], 0)
                self.assertAlmostEqual(record['retained_intersection_mm3'], volume)

    def test_default_assembly_keeps_small_overlap_without_seating_or_warning(self):
        parts, source = assembly(.5)
        parts[1]['annotation']['assembly_cutting_reference'] = dict(
            parent_part='P01', cutting_volume_mm3=100.)
        originals = [p['mesh'].vertices.copy() for p in parts]
        with patch('split3mf.assembly_visibility.assess_insert_visibility', return_value={}), \
             patch('split3mf.assembly_visibility.seat_insert_outward') as seating:
            report = validate_and_seat_assembly(parts, *source, post_fit_difference=True,
                                                allow_manual_adjustment=True)
        self.assertTrue(report['valid'])
        seating.assert_not_called()
        self.assertFalse(annotate_manual_adjustment(parts, report, dict(valid=True)))
        self.assertTrue(report['final_pairs'][0]['ignored_by_cut_volume_ratio'])
        for part, original in zip(parts, originals):
            np.testing.assert_array_equal(part['mesh'].vertices, original)
            self.assertNotIn('assembly_fit', part['annotation'])

    def test_post_difference_residual_uses_same_physical_limit(self):
        child, parent = box_pair(1.5)
        operand = MagicMock()
        residual = operand.__xor__.return_value
        residual.status.return_value = 'Error.NoError'
        for volume in [.5, 1.]:
            residual.volume.return_value = volume
            with patch('split3mf.post_fit_difference._manifold64', return_value=operand):
                if volume < 1:
                    _, report = subtract_parent(dict(mesh=child, annotation={}), parent,
                                                 ignore_overlap_below_mm3=1)
                    self.assertEqual(report['residual_intersection_mm3'], volume)
                    self.assertTrue(report['volume_balance']['valid'])
                else:
                    with self.assertRaisesRegex(ValueError, 'residual intersection'):
                        subtract_parent(dict(mesh=child, annotation={}), parent,
                                        ignore_overlap_below_mm3=1)

    def test_manual_review_lists_only_pairs_outside_tolerance(self):
        parts, source = assembly(1.5)
        third = parts[1]['mesh'].copy()
        third.apply_translation([1.875, 0, 0])
        parts.append(dict(part_id='P03', mesh=third, annotation={}))
        with patch('split3mf.assembly_review.assess_insert_visibility', return_value={}):
            report = measure_manual_adjustment(parts, *source, failure={},
                volume_tolerance_mm3=1e-8, depth_tolerance_mm=0, area_budget_mm2=1,
                ignore_overlap_below_mm3=1)
        annotate_manual_adjustment(parts, report)
        review = parts[0]['annotation']['assembly_fit']
        self.assertEqual(len(review['unresolved_pairs']), 1)
        self.assertEqual(report['affected_parts'], ['P01', 'P02'])
        self.assertIn('判断', review['instruction'])

    def test_read_only_reassessment_clears_obsolete_warning(self):
        parts, source = assembly(.5)
        parts[0]['annotation']['assembly_fit'] = dict(status='manual_adjustment_required')
        with patch('split3mf.assembly_review.assess_insert_visibility', return_value={}):
            report = measure_manual_adjustment(parts, *source,
                volume_tolerance_mm3=1e-8, depth_tolerance_mm=0, area_budget_mm2=1,
                ignore_overlap_below_mm3=1)
        self.assertTrue(report['valid'])
        self.assertFalse(annotate_manual_adjustment(parts, report, dict(valid=True)))
        self.assertNotIn('assembly_fit', parts[0]['annotation'])

    def test_coupled_seating_accepts_each_pair_instead_of_group_sum(self):
        left, parent = box_pair(.6)
        other = left.copy()
        other.apply_translation([0, 5, 0])
        other_parent = parent.copy()
        other_parent.apply_translation([0, 5, 0])
        obstacle = union_mesh([parent, other_parent])
        _, report = seat_insert_outward(union_mesh([left, other]), obstacle, [1, 0, 0],
            moving_parts=[left, other], obstacles=[obstacle], ignore_overlap_below_mm3=1)
        self.assertGreater(report['intersection_before_mm3'], 1)
        self.assertEqual(report['total_travel_mm'], 0)
        self.assertTrue(all(p['accepted'] for p in report['final_contacts']))


if __name__ == '__main__':
    unittest.main()
