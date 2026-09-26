import sys
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.post_fit_difference import subtract_parent, trim_assembly
from split3mf.post_fit_audit import audit_cut_surface, nearest_faces, audit_volume_partition
from split3mf.assembly_visibility import validate_and_seat_assembly


def part(key, mesh, parent=None):
    return dict(part_id=key, mesh=mesh, annotation=dict(parent_part=parent,
                selected_processing_mode='inward' if parent else 'body'),
                face_filament_slot_indices=list(range(len(mesh.faces))))


class PostFitDifferenceTests(unittest.TestCase):
    def test_deferred_failure_keeps_valid_input_and_does_not_commit_rejected_shape(self):
        root=part('P01',trimesh.creation.box(extents=[4]*3))
        child=part('P02',trimesh.creation.box(),'P01')
        source=trimesh.util.concatenate([root['mesh'],child['mesh']])
        with patch('split3mf.post_fit_difference.subtract_parent',side_effect=ValueError('rejected')):
            result,_=trim_assembly([root,child],source.vertices,source.faces,
                np.repeat([1,2],12),defer_failed_difference=True)
        np.testing.assert_array_equal(result[1]['mesh'].vertices,child['mesh'].vertices)
        self.assertFalse(result[1]['annotation']['post_fit_difference_deferred_to_seating']
                         ['rejected_candidate_committed'])
        self.assertNotIn('post_fit_difference_deferred_to_seating',child['annotation'])

    def test_explicit_physical_tolerance_preserves_already_accepted_contact(self):
        mesh=trimesh.creation.box(extents=[.4,.4,1])
        parent=mesh.copy()
        parent.apply_translation([0,0,.991])
        original=part('P02',mesh)
        result,record=subtract_parent(original,parent,depth_tolerance_mm=.01)
        self.assertIs(result,original)
        self.assertFalse(record['changed'])
        self.assertTrue(record['accepted_physical_overlap']['accepted'])
        self.assertEqual(record['removed_volume_mm3'],0)

    def test_volume_roundoff_cannot_exceed_fixed_micro_envelope(self):
        child, result, overlap = Mock(), Mock(), Mock()
        child.volume.return_value = 1.0
        result.volume.return_value = .9
        overlap.volume.return_value = .1001
        self.assertFalse(audit_volume_partition(child, result, overlap,
                                               depth_tolerance_mm=.01)['valid'])

    def test_volume_roundoff_requires_explicit_physical_tolerance(self):
        child, result, overlap = Mock(), Mock(), Mock()
        child.volume.return_value = 1.0
        result.volume.return_value = .9
        overlap.volume.return_value = .100004
        self.assertFalse(audit_volume_partition(child, result, overlap)['valid'])

    def test_cli_removes_post_fit_and_seating_switches(self):
        from split3mf.cli import build_parser
        options = build_parser()._option_string_actions
        for option in ('--post-fit-parent-difference', '--no-post-fit-parent-difference',
                       '--allow-coupled-seating', '--assembly-fit-validation',
                       '--visual-validation-profile'):
            self.assertNotIn(option, options)

    def test_difference_removes_overlap_without_moving_parent(self):
        insert = trimesh.creation.box(extents=[3, 3, 3])
        parent = insert.copy()
        parent.apply_translation([2, 0, 0])
        original = parent.vertices.copy()
        child = part('P02', insert, 'P01')
        result, audit = subtract_parent(child, parent)
        self.assertAlmostEqual(result['mesh'].volume, 18)
        self.assertLess(audit['residual_intersection_mm3'], 1e-8)
        self.assertEqual(len(result['face_filament_slot_indices']), len(result['mesh'].faces))
        self.assertTrue(set(result['face_filament_slot_indices']) <= set(range(12)))
        np.testing.assert_array_equal(parent.vertices, original)
        self.assertAlmostEqual(child['mesh'].volume, 27)

    def test_empty_insert_is_rejected(self):
        box = trimesh.creation.box()
        with self.assertRaisesRegex(ValueError, 'erased'):
            subtract_parent(part('P02', box), box)

    def test_closed_cavity_is_not_mistaken_for_detached_material(self):
        child = trimesh.creation.box(extents=[3, 3, 3])
        cavity = trimesh.creation.box()
        candidate, record = subtract_parent(part('P02', child), cavity)
        self.assertAlmostEqual(candidate['mesh'].volume, 26)
        self.assertEqual(record['shells_after'], 2)
        self.assertEqual(record['material_components_after'], 1)

    def test_disconnected_result_is_rejected(self):
        child = trimesh.creation.box(extents=[3, 1, 1])
        cutter = trimesh.creation.box(extents=[1, 2, 2])
        with self.assertRaisesRegex(ValueError, 'connectivity'):
            subtract_parent(part('P02', child), cutter)

    def test_nonintersecting_part_identity_and_colors_unchanged(self):
        child = part('P02', trimesh.creation.box())
        parent = trimesh.creation.box()
        parent.apply_translation([5, 0, 0])
        result, record = subtract_parent(child, parent)
        self.assertIs(result, child)
        self.assertFalse(record['changed'])

    def test_retained_faces_keep_material_and_source_prefix(self):
        mesh = trimesh.creation.box(extents=[3, 3, 3])
        cutter = mesh.copy()
        cutter.apply_translation([2, 0, 0])
        child = part('P02', mesh)
        child['source_surface_face_count'] = 4
        result, _ = subtract_parent(child, cutter)
        count = result['source_surface_face_count']
        self.assertGreater(count, 0)
        owners = result['face_filament_slot_indices']
        self.assertTrue(all(owner < 4 for owner in owners[:count]))
        for center, normal, owner in zip(result['mesh'].triangles_center,
                                          result['mesh'].face_normals, owners):
            # Only new x-facing cutter faces may be assigned by nearest surface.
            if abs(center[0] - .5) > 1e-9:
                original = mesh.triangles[owner]
                closest = trimesh.triangles.closest_point(original[None], center[None])[0]
                np.testing.assert_allclose(closest, center, atol=1e-9)
                np.testing.assert_allclose(mesh.face_normals[owner], normal, atol=1e-9)

    def test_nested_child_is_checked_against_root_and_corrected_parent(self):
        root = trimesh.creation.box(extents=[2, 2, 2])
        middle = trimesh.creation.box(extents=[2, 2, 2])
        middle.apply_translation([1, 0, 0])
        leaf = trimesh.creation.box(extents=[1, 1, 1])
        leaf.apply_translation([.75, 1.25, 0])
        parts = [part('P03', leaf, 'P02'), part('P01', root), part('P02', middle, 'P01')]
        source = trimesh.util.concatenate([p['mesh'] for p in parts])
        labels = np.repeat([3, 1, 2], 12)
        with patch('split3mf.post_fit_difference.audit_cut_surface', return_value={'valid': True}):
            candidates, records = trim_assembly(parts, source.vertices, source.faces, labels)
        self.assertEqual([r['part_id'] for r in records], ['P02', 'P03'])
        self.assertEqual([s['parent_part'] for s in records[1]['steps']], ['P01', 'P02'])
        self.assertAlmostEqual(parts[2]['mesh'].volume, 8)
        self.assertAlmostEqual(candidates[2]['mesh'].volume, 4)

    def test_failed_surface_gate_does_not_mutate_callers(self):
        root = trimesh.creation.box(extents=[2, 2, 2])
        child = root.copy()
        child.apply_translation([1, 0, 0])
        parts = [part('P01', root), part('P02', child, 'P01')]
        source = trimesh.util.concatenate([root, child])
        with patch('split3mf.post_fit_difference.audit_cut_surface', return_value={'valid': False}):
            with self.assertRaisesRegex(ValueError, 'surface/thickness'):
                validate_and_seat_assembly(parts, source.vertices, source.faces,
                                          np.repeat([1, 2], 12), post_fit_difference=True)
        self.assertAlmostEqual(parts[1]['mesh'].volume, 8)

    def test_thin_backing_rejected_even_if_watertight(self):
        before = trimesh.creation.box(extents=[4, 4, 2])
        after = trimesh.creation.box(extents=[4, 4, .2])
        after.apply_translation([0, 0, .9])
        front = before.submesh([np.flatnonzero(before.face_normals[:, 2] > .9)], append=True)
        record = audit_cut_surface(front, before, after, sample_count=32)
        self.assertFalse(record['valid'])
        self.assertAlmostEqual(record['minimum_sampled_interior_chord_mm'], .2)

    def test_thick_backing_passes(self):
        box = trimesh.creation.box(extents=[4, 4, 2])
        front = box.submesh([np.flatnonzero(box.face_normals[:, 2] > .9)], append=True)
        self.assertTrue(audit_cut_surface(front, box, box, sample_count=32)['valid'])

    def test_nearest_triangle_matches_exhaustive_distance(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        points = np.random.default_rng(42).normal(size=(20, 3))
        _, distances = nearest_faces(mesh, points)
        _, expected, _ = trimesh.proximity.closest_point_naive(mesh, points)
        np.testing.assert_allclose(distances, expected, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
