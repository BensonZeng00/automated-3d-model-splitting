import sys
import unittest
from pathlib import Path
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.assembly_overlap import measure_overlap, union_mesh, measure_patch_bounds
from split3mf.assembly_visibility import validate_and_seat_assembly
from split3mf.assembly_seating import SeatingError
import numpy as np
from unittest.mock import patch


class PhysicalOverlapTests(unittest.TestCase):
    def test_face_planes_tighten_a_containing_slab_without_relaxing_limit(self):
        points=np.array([[0,0,0],[1,0,0],[0,1,0],
                         [0,0,.0099],[1,0,.002],[0,1,.002]])*.5
        mesh=trimesh.convex.convex_hull(points)
        limit=.0098996*.5
        report=measure_patch_bounds(mesh,depth_tolerance_mm=limit,area_budget_mm2=1)
        self.assertGreater(report['components'][0]['pca_slab_thickness_bound_mm'],limit)
        self.assertLessEqual(report['maximum_slab_thickness_bound_mm'],limit)
        self.assertTrue(report['accepted'])

    def measurement(self, depth, side=.4):
        first = trimesh.creation.box(extents=[side, side, 1])
        second = first.copy()
        second.apply_translation([0, 0, 1-depth])
        return measure_overlap(first, second, depth_tolerance_mm=.01)

    def test_shallow_local_overlap_accepted(self):
        self.assertTrue(self.measurement(.009)['accepted'])

    def test_deep_overlap_rejected(self):
        self.assertFalse(self.measurement(.02)['accepted'])

    def test_wide_shallow_overlap_rejected_by_area(self):
        self.assertFalse(self.measurement(.009, side=3)['accepted'])

    def test_group_union_preserves_volume(self):
        first = trimesh.creation.box()
        second = first.copy()
        second.apply_translation([.5, 0, 0])
        result = union_mesh([first, second])
        self.assertTrue(result.is_watertight)
        self.assertAlmostEqual(result.volume, 1.5)

    def test_sibling_collision_is_not_missed(self):
        root = trimesh.creation.box()
        root.apply_translation([10, 0, 0])
        child = trimesh.creation.box()
        parts = [dict(part_id=key, mesh=mesh, annotation=dict(
            parent_part=parent, selected_processing_mode=mode))
            for key, mesh, parent, mode in [('P01', root, None, 'body'),
                                           ('P02', child, 'P01', 'inward'),
                                           ('P03', child.copy(), 'P01', 'inward')]]
        source = trimesh.util.concatenate([p['mesh'] for p in parts])
        labels = np.repeat([1, 2, 3], 12)
        original = [p['mesh'].vertices.copy() for p in parts]
        with self.assertRaises(SeatingError):
            validate_and_seat_assembly(parts, source.vertices, source.faces, labels)
        for part, vertices in zip(parts, original):
            np.testing.assert_array_equal(part['mesh'].vertices, vertices)

    def test_seating_visits_parent_before_child_despite_input_order(self):
        parts = []
        for number, parent in [(3, 'P02'), (1, None), (2, 'P01')]:
            mesh = trimesh.creation.box()
            mesh.apply_translation([number * 3, 0, 0])
            parts.append(dict(part_id=f'P{number:02}', mesh=mesh, annotation=dict(
                parent_part=parent, selected_processing_mode='inward' if parent else 'body')))
        source = trimesh.util.concatenate([p['mesh'] for p in parts])
        labels = np.repeat([3, 1, 2], 12)
        visited = []
        def assess(source_patch, insert, parent, **kwargs):
            visited.append(round(insert.centroid[0] / 3))
            return {'covered_fraction': 0}
        with patch('split3mf.assembly_visibility.assess_insert_visibility', side_effect=assess):
            validate_and_seat_assembly(parts, source.vertices, source.faces, labels)
        self.assertEqual(visited[:2], [2, 3])
