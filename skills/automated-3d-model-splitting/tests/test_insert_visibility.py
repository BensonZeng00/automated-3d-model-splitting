"""Thin covers need ownership checks even when depth differences are tiny."""
import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.insert_visibility import assess_insert_visibility
from split3mf.assembly_seating import seat_insert_outward
from split3mf.surface_rays import first_surface_depth


class InsertVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.insert=trimesh.creation.box(extents=[2,2,1])
        self.patch=self.insert.submesh([np.flatnonzero(self.insert.face_normals[:,2]>.9)],append=True)

    def test_near_coplanar_membrane_is_detected(self):
        cover=trimesh.creation.box(extents=[2,2,.01])
        cover.apply_translation([0,0,.51])
        result=assess_insert_visibility(self.patch,self.insert,cover)
        self.assertEqual(result['covered_fraction'],1)

    def test_socket_floor_does_not_occlude(self):
        floor=trimesh.creation.box(extents=[4,4,1])
        floor.apply_translation([0,0,-1.1])
        result=assess_insert_visibility(self.patch,self.insert,floor)
        self.assertEqual(result['covered_samples'],0)

    def test_ray_uses_nearest_triangle_including_backfaces(self):
        points=np.array([[0,0,0],[4,4,0]])
        depth=first_surface_depth(self.insert,points,[0,0,-1])
        self.assertAlmostEqual(depth[0],-.5)
        self.assertTrue(np.isinf(depth[1]))

    def test_rigid_seating_keeps_shape_and_parent(self):
        parent=trimesh.creation.box(extents=[4,4,1])
        parent.apply_translation([0,0,-.96])
        before=parent.vertices.copy()
        seated,record=seat_insert_outward(self.insert,parent,[0,0,-1])
        np.testing.assert_array_equal(parent.vertices,before)
        np.testing.assert_allclose(seated.extents,self.insert.extents)
        np.testing.assert_array_equal(seated.faces,self.insert.faces)
        self.assertAlmostEqual(record['outward_travel_mm'],.04,places=4)
        self.assertLessEqual(record['intersection_after_mm3'],1e-8)

    def test_seating_rejects_unbounded_or_unresolved_motion(self):
        with self.assertRaises(ValueError):
            seat_insert_outward(self.insert,trimesh.creation.box(extents=[4,4,4]),[0,0,-1])


if __name__=='__main__':
    unittest.main()
