"""Geometry-based regressions: no vendor filename, part ID or color special cases."""
import sys
import unittest
from unittest.mock import patch as mock_patch
from pathlib import Path
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from split3mf.curved_backing import build, first_exit
from split3mf.backing_thickness import audit_local_backing
from split3mf.backing_repair import ensure_backing, BackingRepairError
from split3mf.local_ray_probe import LocalRayProbe
from split3mf.print_tolerance import PrintTolerance, tolerance_scope


def surface():
    x,y=np.meshgrid(np.linspace(-2,2,9),np.linspace(-2,2,9))
    vertices=np.column_stack((x.ravel(), y.ravel(), (.08*x*x).ravel()))
    faces=[]
    for row in range(8):
        for col in range(8):
            a=row*9+col
            faces.extend([[a,a+1,a+10],[a,a+10,a+9]])
    return trimesh.Trimesh(vertices,faces,process=False)


class BackingRepairTests(unittest.TestCase):
    def setUp(self):
        self.patch=surface()
        self.parent=trimesh.creation.box(extents=[10]*3)
        self.thin,_=build(self.patch,self.parent,[0,0,-1],preferred_depth=.1)

    def test_local_normal_screen_finds_thin_closed_backing(self):
        record=audit_local_backing(self.patch,self.thin)
        self.assertFalse(record['valid'])
        self.assertGreater(record['thin_interior_area_mm2'],1)
        self.assertGreater(record['thin_exits_on_backing'],0)
        self.assertFalse(record['global_minimum_wall_thickness_measured'])

    def test_repair_preserves_front_and_per_face_material_provenance(self):
        colors=['A' if i%2 else 'B' for i in range(len(self.patch.faces))]
        with tolerance_scope(PrintTolerance(repair_thin_backing=True)):
            mesh, record, result_colors=ensure_backing(self.patch,self.thin,self.parent,
                colors,[0,0,-1],part_id='arbitrary')
        self.assertTrue(record['after']['valid'])
        self.assertTrue(record['repaired'])
        np.testing.assert_array_equal(mesh.triangles[:len(colors)],self.patch.triangles)
        self.assertEqual(result_colors[:len(colors)],colors)
        self.assertEqual(len(result_colors),len(mesh.faces))
        self.assertEqual(set(result_colors[len(colors):]),{'A','B'})

    def test_repair_requires_explicit_enable_and_does_not_mutate(self):
        before=self.thin.vertices.copy()
        with self.assertRaises(BackingRepairError):
            ensure_backing(self.patch,self.thin,self.parent,['A']*len(self.patch.faces),
                           [0,0,-1],part_id='arbitrary')
        np.testing.assert_array_equal(before,self.thin.vertices)

    def test_print_scale_hidden_thin_patch_is_advisory(self):
        audit=dict(valid=False,thin_interior_area_mm2=1.95,
                   thin_exits_on_backing=0,thin_exits_on_source=0)
        with mock_patch('split3mf.backing_repair.audit_local_backing',return_value=audit), \
             mock_patch('split3mf.backing_repair.build') as rebuild:
            result,record,colors=ensure_backing(
                self.patch,self.thin,self.parent,['A']*len(self.patch.faces),
                [0,0,-1],part_id='print-scale-advisory')
        self.assertIs(result,self.thin)
        self.assertTrue(record['accepted_without_repair'])
        self.assertEqual(record['acceptance'],'bounded_hidden_thin_backing_patch')
        self.assertEqual(record['advisory_area_limit_mm2'],2.0)
        self.assertIsNone(colors)
        rebuild.assert_not_called()

    def test_valid_backing_is_returned_unchanged(self):
        mesh,_=build(self.patch,self.parent,[0,0,-1],direction_mode='local-normal')
        result,record,colors=ensure_backing(self.patch,mesh,self.parent,
            ['A']*len(self.patch.faces),[0,0,-1],part_id='arbitrary')
        self.assertIs(result,mesh)
        self.assertFalse(record['repaired'])
        self.assertIsNone(colors)

    def test_non_axis_aligned_ray_matches_brute_force(self):
        points=self.patch.triangles_center[::9]
        axes=-self.patch.face_normals[::9]
        actual,_=LocalRayProbe(self.parent).exits(points,axes,20)
        expected=[first_exit(self.parent.triangles,p[None],a)[0] for p,a in zip(points,axes)]
        np.testing.assert_allclose(actual,expected,atol=1e-10)

    def test_scale_is_accounted_before_parent_subtraction(self):
        mesh,_=build(self.patch,self.parent,[0,0,-1],preferred_depth=.5)
        self.assertTrue(audit_local_backing(self.patch,mesh,scale_factor=1)['valid'])
        self.assertFalse(audit_local_backing(self.patch,mesh,scale_factor=.5)['valid'])

    def test_invalid_direction_mode_rejected(self):
        with self.assertRaises(ValueError):
            build(self.patch,self.parent,[0,0,-1],direction_mode='unknown')

    def test_repaired_cap_cannot_claim_old_plan_or_skip_final_audit(self):
        from split3mf.debug_export import validate_shared_child_cap_decisions
        record=dict(repaired=True,source_front_triangles_preserved=True,
                    outside_volume_mm3=0.,construction=dict(
                        minimum_parent_reserve_mm=.05,maximum_depth_mm=3.))
        with tolerance_scope(PrintTolerance(repair_thin_backing=True)):
            with self.assertRaises(ValueError):
                validate_shared_child_cap_decisions(7,{},
                    {'actual_backing_validation':record},interface_geometry='local-connector')
            record['after_finalization']={'valid':True}
            record['matching_socket_fit']={'valid':True}
            report=validate_shared_child_cap_decisions(7,{0:object()},
                {'actual_backing_validation':record},interface_geometry='local-connector')
        self.assertTrue(report[0]['original_plan_superseded'])

    def test_repair_cannot_disable_visual_validation(self):
        from split3mf.cli import build_parser
        from split3mf.uniform_fit import configure_uniform_fit
        args=build_parser().parse_args(['--input','example.3mf',
            '--repair-thin-backing','--visual-validation-profile','off'])
        with self.assertRaises(ValueError):
            configure_uniform_fit(args)


if __name__=='__main__':
    unittest.main()
