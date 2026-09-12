"""Actual cutting-volume ratios, including scale invariance and missing data."""
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
from split3mf.assembly_overlap import measure_overlap, union_mesh
from split3mf.assembly_visibility import validate_and_seat_assembly
from split3mf.assembly_seating import seat_insert_outward
from split3mf.cutting_reference import attach_cutting_references, pair_cutting_volume
from split3mf.local_connectors import subtract_socket_cutters
from split3mf.overlap_policy import ratio_volume_limit
from split3mf.post_fit_difference import subtract_parent


def part(key, parent=None):
    return dict(part_id=key, mesh=trimesh.creation.box(extents=[10]*3),
                annotation=dict(parent_part=parent, selected_processing_mode='inward' if parent else 'body'))


class CuttingVolumeRatioTests(unittest.TestCase):
    def test_reference_uses_actual_material_removed_not_whole_cutter_volume(self):
        parent, child = part('P01'), part('P02', 'P01')
        child['mesh'].apply_translation([8, 0, 0])
        _, record = subtract_socket_cutters(parent['mesh'], [child['mesh']])
        record['child_indices'] = [2]
        attach_cutting_references([parent, child], {'P01': record})
        self.assertAlmostEqual(child['mesh'].volume, 1000)
        self.assertAlmostEqual(pair_cutting_volume(parent, child), 200)
        child['mesh'].apply_scale(.5)
        self.assertAlmostEqual(pair_cutting_volume(parent, child), 200)
        self.assertEqual(pair_cutting_volume(parent, child), pair_cutting_volume(child, parent))

    def test_ratio_boundary_and_scale_invariance(self):
        for scale in [.1, 1, 10]:
            for ratio in [.009, .01, .011]:
                first = trimesh.creation.box(extents=[10]*3)
                second = first.copy()
                second.apply_translation([10-ratio*10, 0, 0])
                first.apply_scale(scale)
                second.apply_scale(scale)
                report = measure_overlap(first, second, cutting_volume_mm3=1000*scale**3,
                                         ignore_overlap_ratio=.01)
                self.assertAlmostEqual(report['overlap_cutting_volume_ratio'], ratio)
                self.assertEqual(report['accepted'], ratio < .01)

    def test_fixed_one_cubic_mm_is_no_longer_the_decision(self):
        for overlap, cutting, accepted in [(.5, 10, False), (2, 1000, True)]:
            first = trimesh.creation.box(extents=[10]*3)
            second = first.copy()
            second.apply_translation([10-overlap/100, 0, 0])
            report = measure_overlap(first, second, cutting_volume_mm3=cutting, ignore_overlap_ratio=.01)
            self.assertEqual(report['accepted'], accepted)

    def test_disconnected_overlap_fragments_are_summed_before_ratio(self):
        left = trimesh.creation.box(extents=[10]*3)
        right = left.copy()
        right.apply_translation([9.94, 0, 0])
        left2, right2 = left.copy(), right.copy()
        left2.apply_translation([0, 20, 0])
        right2.apply_translation([0, 20, 0])
        report = measure_overlap(trimesh.util.concatenate([left, left2]),
            trimesh.util.concatenate([right, right2]), cutting_volume_mm3=1000,
            ignore_overlap_ratio=.01)
        self.assertAlmostEqual(report['intersection_mm3'], 12)
        self.assertFalse(report['accepted'])

    def test_missing_zero_and_invalid_references_do_not_ignore_overlap(self):
        mesh = trimesh.creation.box()
        for volume in [None, 0]:
            self.assertFalse(measure_overlap(mesh, mesh, cutting_volume_mm3=volume,
                                            ignore_overlap_ratio=.01)['accepted'])
        for ratio in [-.01, 1.1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                ratio_volume_limit(10, ratio)
        for volume in [-1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                ratio_volume_limit(volume, .01)
        self.assertFalse(measure_overlap(mesh, mesh, cutting_volume_mm3=1000,
                                        ignore_overlap_ratio=0)['accepted'])

    def test_siblings_use_smaller_cut_and_missing_nonroot_reference_is_not_invented(self):
        first, second = part('P02', 'P01'), part('P03', 'P01')
        attach_cutting_references([first, second], {'P01': dict(child_indices=[2, 3],
            cutter_steps=[dict(cutter_index=0, intersection_volume_mm3=10),
                          dict(cutter_index=1, intersection_volume_mm3=100)])})
        self.assertEqual(pair_cutting_volume(first, second), 10)
        second['annotation'].pop('assembly_cutting_reference')
        self.assertIsNone(pair_cutting_volume(first, second))

    def test_real_split_assembly_ratio_survives_case_save_and_no_warning(self):
        first, second = part('P01'), part('P02', 'P01')
        second['mesh'].apply_translation([9.98, 0, 0])  # 2 mm³ residual / 1000 = .2%
        attach_cutting_references([first, second], {'P01': dict(child_indices=[2],
            cutter_steps=[dict(cutter_index=0, intersection_volume_mm3=1000)])})
        source = trimesh.util.concatenate([p['mesh'] for p in [first, second]])
        with tempfile.TemporaryDirectory() as directory, \
             patch('split3mf.assembly_visibility.assess_insert_visibility', return_value={}):
            report = validate_and_seat_assembly([first, second], source.vertices, source.faces,
                np.repeat([1, 2], 12), post_fit_difference=True, recovery_dir=directory)
            self.assertTrue(report['valid'])
            self.assertEqual(report['corrected_parts'], 0)
            self.assertEqual(report['post_fit_difference'], [])
            saved = json.loads((Path(directory) / 'assembly_case/case.json').read_text())
            self.assertEqual(saved['options']['ignore_overlap_ratio'], .01)
            self.assertEqual(saved['parts'][1]['annotation']['assembly_cutting_reference']
                             ['cutting_volume_mm3'], 1000)
            self.assertNotIn('assembly_fit', second['annotation'])

    def test_difference_uses_frozen_reference_and_keeps_conservation_gate(self):
        first, second = part('P01'), part('P02', 'P01')
        second['mesh'].apply_translation([9.98, 0, 0])
        kept, record = subtract_parent(second, first['mesh'], cutting_volume_mm3=1000,
                                       ignore_overlap_ratio=.01)
        self.assertIs(kept, second)
        self.assertTrue(record['ignored_by_cut_volume_ratio'])
        _, cut = subtract_parent(second, first['mesh'], cutting_volume_mm3=10,
                                  ignore_overlap_ratio=.01)
        self.assertTrue(cut['changed'])
        self.assertTrue(cut['volume_balance']['valid'])

    def test_coupled_search_uses_each_pairs_frozen_denominator(self):
        mesh = trimesh.creation.box(extents=[10]*3)
        other = mesh.copy()
        other.apply_translation([0, 20, 0])
        obstacles = [mesh.copy(), other.copy()]
        for obstacle in obstacles:
            obstacle.apply_translation([9.94, 0, 0])  # 6 mm³ per pair
        _, report = seat_insert_outward(union_mesh([mesh, other]), union_mesh(obstacles), [1, 0, 0],
            moving_parts=[mesh, other], obstacles=obstacles, ignore_overlap_ratio=.01,
            cutting_volumes_mm3=[[1000, 1000], [1000, 1000]])
        self.assertAlmostEqual(report['intersection_before_mm3'], 12)
        self.assertEqual(report['total_travel_mm'], 0)


if __name__ == '__main__':
    unittest.main()
