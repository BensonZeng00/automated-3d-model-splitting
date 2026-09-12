import sys
import unittest
from unittest.mock import patch as mock_patch
from pathlib import Path
import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from split3mf.curved_backing import build
from split3mf.backing_thickness import audit_backing
from split3mf.assembly_seating import seat_insert_outward, SeatingError
from split3mf.local_connectors import _manifold64


class CurvedBackingTests(unittest.TestCase):
    def patch(self):
        x,y=np.meshgrid(np.linspace(-2,2,9),np.linspace(-2,2,9))
        v=np.column_stack((x.ravel(),y.ravel(),(.08*x*x).ravel()))
        f=[]
        for row in range(8):
            for col in range(8):
                a=row*9+col
                f.extend([[a,a+1,a+10],[a,a+10,a+9]])
        return trimesh.Trimesh(v,f,process=False)

    def test_curved_front_stays_exact_and_backing_is_solid(self):
        patch=self.patch()
        parent=trimesh.creation.box(extents=[10,10,10])
        solid,report=build(patch,parent,[0,0,-1],preferred_depth=1)
        np.testing.assert_array_equal(solid.triangles[:len(patch.faces)],patch.triangles)
        self.assertTrue(solid.is_watertight)
        self.assertTrue(solid.is_winding_consistent)
        self.assertLessEqual(report['maximum_depth_mm'],1)
        audit=audit_backing(patch,solid,[0,0,-1])
        self.assertEqual(audit['missing_eligible_faces'],0)
        self.assertEqual(audit['thin_interior_area_mm2'],0)

    def test_invalid_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            build(self.patch(),trimesh.creation.box(),[0,0,0])

    def test_diagonal_seating_clears_side_contact_without_shape_change(self):
        insert=trimesh.creation.box(extents=[1,1,1])
        wall=trimesh.creation.box(extents=[1,2,2]);wall.apply_translation([.98,0,0])
        old=insert.vertices.copy()
        seated,report=seat_insert_outward(insert,wall,[0,0,-1])
        self.assertEqual(report['search_strategy'],'bounded_contact_pose')
        self.assertLessEqual(report['total_travel_mm'],.1)
        self.assertLessEqual(abs((_manifold64(seated)^_manifold64(wall)).volume()),1e-8)
        np.testing.assert_array_equal(insert.vertices,old)
        np.testing.assert_allclose(seated.extents,insert.extents)

    def test_failure_keeps_measurements(self):
        with self.assertRaises(SeatingError) as failure:
            seat_insert_outward(trimesh.creation.box(),trimesh.creation.box(extents=[4]*3),[0,0,-1])
        self.assertEqual(len(failure.exception.record['axial_trials']),10)

    def test_visible_insert_still_requires_collision_check(self):
        from split3mf.assembly_visibility import validate_and_seat_assembly
        insert=trimesh.creation.box()
        wall=trimesh.creation.box(extents=[1,2,2]);wall.apply_translation([.98,0,0])
        parts=[{'part_id':'P01_insert','mesh':insert,'annotation':{
            'selected_processing_mode':'inward','parent_part':'P02'}},
            {'part_id':'P02_body','mesh':wall,'annotation':{}}]
        visible={'covered_fraction':0.,'covered_area_estimate_mm2':0.,'inward_axis':[0,0,-1]}
        with mock_patch('split3mf.assembly_visibility.assess_insert_visibility',return_value=visible):
            result=validate_and_seat_assembly(parts,insert.vertices,insert.faces,
                                            np.ones(len(insert.faces),dtype=int),
                                            ignore_overlap_below_mm3=0)
        self.assertEqual(result['corrected_parts'],1)
        self.assertLessEqual(abs((_manifold64(parts[0]['mesh'])^_manifold64(wall)).volume()),1e-8)


if __name__=='__main__':unittest.main()
