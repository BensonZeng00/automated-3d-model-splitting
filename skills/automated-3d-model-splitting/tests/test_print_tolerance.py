from pathlib import Path
import sys
import unittest
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
import numpy as np
import trimesh
from split3mf.edge_index import FaceEdgeIndex
from split3mf.print_tolerance import PrintTolerance, tolerance_scope, small_patch_report
from split3mf.surface_preservation import audit_replaced_surface, face_key


class PrintToleranceTests(unittest.TestCase):
    def test_secondary_micro_loop_does_not_become_connector(self):
        from split3mf.micro_interfaces import filter_micro_interface_loops
        vertices=np.asarray([[0,0,0],[10,0,0],[10,10,0],[0,10,0],[1,1,0],[1.2,1,0],[1.2,1.2,0],[1,1.2,0]])
        records=[dict(loop_index=0,loop=[0,1,2,3]),dict(loop_index=1,loop=[4,5,6,7])]
        retained,ignored=filter_micro_interface_loops(vertices,records,part_index=14)
        self.assertEqual([r['loop_index'] for r in retained],[0])
        self.assertEqual([r['loop_index'] for r in ignored],[1])
        retained,ignored=filter_micro_interface_loops(vertices,records[1:],part_index=9)
        self.assertEqual(len(retained),1)
        self.assertEqual(ignored,[])
        with tolerance_scope(PrintTolerance(0)):
            retained,ignored=filter_micro_interface_loops(vertices,records,part_index=14)
        self.assertEqual(len(retained),2)

    def test_indexed_winding_matches_reference_and_preserves_geometry(self):
        from split3mf.winding import fix_winding_indexed
        rng = np.random.default_rng(43)
        for open_shell in (False, True):
            mesh = trimesh.creation.icosphere(subdivisions=2)
            if open_shell:
                mesh.update_faces(np.arange(len(mesh.faces)-7))
            faces = np.asarray(mesh.faces).copy()
            mask = rng.random(len(faces)) < .4
            faces[mask] = faces[mask, ::-1]
            mesh.faces = faces
            reference = mesh.copy()
            trimesh.repair.fix_winding(reference)
            fix_winding_indexed(mesh)
            np.testing.assert_array_equal(mesh.vertices, reference.vertices)
            np.testing.assert_array_equal(mesh.faces, reference.faces)

    def test_micro_shell_orientation_is_advisory_without_deleting_geometry(self):
        from split3mf.mesh import watertight_component_orientation_audit
        main = trimesh.creation.box(extents=[8,8,8])
        dust = trimesh.creation.box(extents=[.2,.3,.4]); dust.invert()
        mesh = trimesh.util.concatenate([main,dust])
        result = watertight_component_orientation_audit(mesh)
        self.assertEqual(len(result['print_tolerance_ignored_component_ids']),1)
        self.assertEqual(len(mesh.faces),24)
        with tolerance_scope(PrintTolerance(0)):
            strict = watertight_component_orientation_audit(mesh)
        self.assertEqual(strict['print_tolerance_ignored_component_ids'],[])
        single = watertight_component_orientation_audit(dust)
        self.assertEqual(single['inward_closed_component_count'],1)

    def test_touching_holes_keep_directed_cycles(self):
        from split3mf.micro_mesh_repair import directed_cycles
        edges=[(0,1),(1,2),(2,0),(0,3),(3,4),(4,0)]
        cycles=directed_cycles(edges)
        rebuilt=[(a,b) for loop in cycles for a,b in zip(loop,np.roll(loop,-1))]
        self.assertCountEqual(rebuilt,edges)
        self.assertTrue(all(len(loop)==len(set(loop)) for loop in cycles))

    def test_small_hole_repair_produces_closed_solid(self):
        from split3mf.micro_mesh_repair import repair_micro_mesh
        mesh=trimesh.creation.box(extents=[0.4,0.4,0.4])
        mesh.update_faces(np.arange(len(mesh.faces)-1))
        repaired,record=repair_micro_mesh(mesh)
        self.assertTrue(repaired.is_watertight)
        self.assertTrue(repaired.is_winding_consistent)
        self.assertEqual(record['closed_micro_loops'],1)

    def test_area_applies_to_whole_patch(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[2,0,0],[3,0,0],[2,1,0]])
        with tolerance_scope(PrintTolerance(1.0)):
            self.assertTrue(small_patch_report(v,[[0,1,2],[3,4,5]])['accepted'])
            self.assertFalse(small_patch_report(v,[[0,1,2],[3,4,5],[0,1,2]])['accepted'])

    def test_traced_subdivision_preserves_large_source(self):
        v=np.array([[0,0,0],[4,0,0],[0,4,0],[2,0,0]],dtype=float)
        f=np.array([[0,3,2],[3,1,2]])
        m=trimesh.Trimesh(vertices=v,faces=f,process=False)
        m.metadata['boundary_closure_subdivisions']=[dict(source_triangle=v[[0,1,2]].tolist(),replacement_triangles=v[f].tolist())]
        result=audit_replaced_surface(Counter({face_key(v[[0,1,2]]):1}),Counter(face_key(t) for t in v[f]),m)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['equivalent_subdivided_source_faces'],1)

    def test_displaced_patch_is_blocked(self):
        old=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=float)
        m=trimesh.Trimesh(vertices=old+[0,0,1],faces=[[0,1,2]],process=False)
        result=audit_replaced_surface(Counter({face_key(old):1}),Counter(face_key(t) for t in m.triangles),m)
        self.assertFalse(result['accepted'])

    def test_incremental_edges_match_full_rebuild(self):
        faces=[[0,1,2],[2,1,3]]
        index=FaceEdgeIndex(faces)
        index.remove(0,faces[0]);faces[0]=[0,4,2];index.add(0,faces[0])
        faces.append([4,1,2]);index.add(2,faces[2])
        self.assertEqual(index.within(range(5)),set(edge for face in faces for edge in index.edges(face)))


if __name__=='__main__':
    unittest.main()
