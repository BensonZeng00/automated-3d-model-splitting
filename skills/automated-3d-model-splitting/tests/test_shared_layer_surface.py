import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.domain import PlanarArcRetopologyConfig, PlanarArcRetopologyContext
from split3mf.interface_retopology import InterfaceRetopologyService
from split3mf.layer_surface_plan import build_layer_surface_plan
from split3mf.layer_seam_planning import (
    activate_layer_seams,
    layer_child_boundary_topology,
    prepare_layer_seams,
)
from split3mf.common import Component
from split3mf.assembly import component_boundary_neighbor_lookup
from split3mf.shared_seam_topology import classify_shared_seams
from split3mf.package_io import export_colored_parts_3mf, load_colored_mesh_objects_3mf
from split3mf.mesh import boundary_loops


def fixture():
    points = np.array([[-1,0,0],[0,0,0],[1,0,0],[-1,1,0],[0,1,0],[1,1,0]], dtype=float)
    faces = np.array([[0,1,4],[0,4,3],[1,2,5],[1,5,4]])
    loops = [[0,1,4,3], [1,2,5,4]]
    return points, faces, loops


class SharedLayerSurfaceTests(unittest.TestCase):
    def context(self, source=True):
        return PlanarArcRetopologyContext(PlanarArcRetopologyConfig(),
            inherited_frozen_vertex_ids=np.arange(6) if source else None)

    def test_valid_shared_edge_is_not_a_nonmanifold_edge(self):
        points, faces, loops = fixture()
        plan = build_layer_surface_plan(points, faces, loops, self.context())
        self.assertEqual(plan.report['shared_edge_count'], 1)
        self.assertEqual(plan.report['shared_vertex_count'], 2)
        self.assertFalse(plan.report['tolerance_waiver_applied'])
        np.testing.assert_array_equal(plan.vertices, points)

    def test_reversed_rotated_and_reordered_loops_have_same_plan(self):
        points, faces, loops = fixture()
        a = build_layer_surface_plan(points, faces, loops, self.context())
        variants = [list(reversed(loops[1][1:]+loops[1][:1])), loops[0][2:]+loops[0][:2]]
        b = build_layer_surface_plan(points, faces, variants, self.context())
        np.testing.assert_array_equal(a.vertices, b.vertices)
        np.testing.assert_array_equal(a.faces, b.faces)

    def test_parent_and_children_consume_identical_snapshot(self):
        points, faces, loops = fixture()
        context = self.context()
        plan = build_layer_surface_plan(points, faces, loops, context)
        context = replace(context, active_layer_seam=plan)
        parent, _ = InterfaceRetopologyService.retopologize_local_loops(
            plan.vertices, loops, np.arange(6), context, plan.faces)
        for loop in loops:
            ids = np.array(sorted(loop))
            local_loop = [int(np.where(ids==i)[0][0]) for i in loop]
            child, _ = InterfaceRetopologyService.retopologize_local_loops(
                plan.vertices[ids], [local_loop], ids, context)
            np.testing.assert_array_equal(child, parent[ids])

    def test_duplicate_loop_records_are_deduplicated(self):
        points, faces, loops = fixture()
        plan = build_layer_surface_plan(points, faces, [loops[0], loops[0][::-1]], self.context())
        self.assertEqual(plan.report['duplicate_loop_records'], 1)

    def test_real_nonmanifold_shared_edge_still_fails(self):
        points, faces, loops = fixture()
        with self.assertRaises(ValueError):
            classify_shared_seams(points, np.vstack((faces, [1,4,2])), loops)

    def test_nearby_vertices_are_not_merged_by_distance(self):
        points, faces, loops = fixture()
        points = np.vstack((points, points[[1,4]] + [0,0,1e-7]))
        faces[2:] = [[6,2,5],[6,5,7]]
        self.assertIsNone(build_layer_surface_plan(points, faces, [loops[0],[6,2,5,7]], self.context()))

    def test_stale_stage_input_is_rejected(self):
        points, faces, loops = fixture()
        context = self.context()
        context.layer_seams[7] = build_layer_surface_plan(points, faces, loops, context)
        moved = points.copy(); moved[0,0] += .01
        with self.assertRaisesRegex(ValueError, 'current recursive input'):
            activate_layer_seams(context, 7, moved, faces)

    def test_smooth_mode_runs_one_surface_audit_not_per_child(self):
        points, faces, loops = fixture()
        context = self.context(False)
        def proposal(p, ids, _):
            q=p.copy(); q[:,2] += .01 if 0 in ids else .02
            return q, {}
        def solver(p, f, boundary, targets, *args, **kwargs):
            q=p.copy(); q[boundary]=targets
            return q, {'valid':True, 'boundary_match_error_mm':0.}
        with patch.object(InterfaceRetopologyService, 'retopologize_loop', side_effect=proposal), \
                patch('split3mf.interface_retopology._surface_band_deformation', side_effect=solver) as audit:
            plan=build_layer_surface_plan(points, faces, loops, context)
            self.assertEqual(audit.call_count, 1)
        self.assertAlmostEqual(plan.report['maximum_proposal_disagreement_mm'], .01)
        # Three-way region junctions are source-locked, not averaged out of place.
        np.testing.assert_array_equal(plan.vertices[[1,4]], points[[1,4]])
        context=replace(context, active_layer_seam=plan)
        with self.assertRaisesRegex(ValueError, 'authoritative'):
            InterfaceRetopologyService.retopologize_local_loops(points,loops,np.arange(6),context,faces)

    def test_package_reload_rebuilds_plan_with_local_ids(self):
        points, faces, loops = fixture()
        part=dict(part_id='surface', mesh=trimesh.Trimesh(points,faces,process=False), color_hex='#FFFFFF')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'surface.3mf'
            export_colored_parts_3mf(path,[part],'stage')
            mesh=load_colored_mesh_objects_3mf(path)[0]['mesh']
        plan=build_layer_surface_plan(mesh.vertices,mesh.faces,loops,self.context())
        np.testing.assert_array_equal(plan.vertices,points)

    def test_actual_smooth_surface_solver_preserves_a_shared_chain(self):
        x,y=np.meshgrid(np.linspace(-2,2,9),np.linspace(-2,2,9))
        points=np.column_stack((x.ravel(),y.ravel(),(.03*x*x).ravel()))
        faces=[]; groups=[]
        for row in range(8):
            for col in range(8):
                a=row*9+col
                faces.extend([[a,a+1,a+10],[a,a+10,a+9]])
                groups.extend([int(col>=4)]*2)
        faces=np.asarray(faces); groups=np.asarray(groups)
        loops=[loop for group in (0,1) for loop in boundary_loops(faces[groups==group])]
        plan=build_layer_surface_plan(points,faces,loops,self.context(False),groups)
        self.assertTrue(plan.quality['valid'])
        self.assertEqual(plan.report['shared_edge_count'],8)
        self.assertEqual(plan.report['residual_shared_target_disagreement_mm'],0.)

    def test_unknown_inherited_surface_cannot_be_silently_reshaped(self):
        points,faces,loops=fixture()
        context=replace(self.context(False),inherited_surface_provenance_known=False)
        with self.assertRaisesRegex(ValueError,'provenance'):
            build_layer_surface_plan(points,faces,loops,context)

    def test_inherited_contact_vertices_stay_fixed(self):
        points,faces,loops=fixture()
        context=replace(self.context(False),inherited_frozen_vertex_ids=np.arange(6))
        plan=build_layer_surface_plan(points,faces,loops,context)
        np.testing.assert_array_equal(plan.vertices,points)
        np.testing.assert_array_equal(plan.faces,faces)

    def test_settings_change_invalidates_the_plan(self):
        points,faces,loops=fixture()
        context=self.context()
        context.layer_seams[7]=build_layer_surface_plan(points,faces,loops,context)
        changed=replace(context,config=replace(context.config,smooth_passes=3))
        with self.assertRaisesRegex(ValueError,'settings'):
            activate_layer_seams(changed,7,points,faces)

    def test_production_layer_preparation_and_parent_activation(self):
        points, top, loops = fixture()
        points = points * 10
        points = np.vstack((points, points + [0,0,-10]))
        faces = top.tolist() + (top[:, ::-1] + 6).tolist()
        rim = [0,1,2,5,4,3]
        for a,b in zip(rim, rim[1:]+rim[:1]):
            faces.extend([[b,a,a+6],[b,a+6,b+6]])
        faces = np.asarray(faces)
        components = []
        for indices in (np.arange(4,len(faces)), np.arange(2), np.arange(2,4)):
            local = trimesh.Trimesh(points, faces[indices], process=False)
            components.append(Component('1', indices, len(indices), float(local.area),
                                        local.bounds[0], local.bounds[1], local.centroid))
        lookup = component_boundary_neighbor_lookup(faces, components)
        context = self.context()
        child_points, child_faces, child_context = prepare_layer_seams(
            points, faces, components, 1, [2,3], {1:[2,3]}, lookup, context)
        self.assertIsNotNone(child_context.active_layer_seam)
        parent_points, parent_faces, parent_context = activate_layer_seams(context, 1, points, faces)
        self.assertIs(child_context.active_layer_seam, parent_context.active_layer_seam)
        np.testing.assert_array_equal(child_points, parent_points)
        np.testing.assert_array_equal(child_faces, parent_faces)
        for index, loop in enumerate(loops):
            actual, records = InterfaceRetopologyService.retopologize_local_loops(
                child_points, [loop], np.arange(len(points)), child_context, child_faces[index*2:index*2+2])
            np.testing.assert_array_equal(actual[loop], parent_points[loop])
            self.assertEqual(records[0]['shared_layer_surface']['shared_edge_count'],1)

    def test_subtree_boundary_topology_is_reused_and_connectivity_invalidates_it(self):
        points, faces, _ = fixture()
        local = trimesh.Trimesh(points, faces, process=False)
        component = Component(
            '1', np.arange(len(faces)), len(faces), float(local.area),
            local.bounds[0], local.bounds[1], local.centroid,
        )
        context = self.context()
        with patch('split3mf.mesh.boundary_loops', wraps=boundary_loops) as walk:
            first = layer_child_boundary_topology(faces, [component], [1], context)
            second = layer_child_boundary_topology(faces.copy(), [component], [1], context)
            self.assertIs(first, second)
            self.assertEqual(walk.call_count, 1)

            changed = faces.copy()
            changed[0] = changed[0, ::-1]
            third = layer_child_boundary_topology(changed, [component], [1], context)
            self.assertIsNot(first, third)
            self.assertEqual(walk.call_count, 2)


if __name__ == '__main__':
    unittest.main()
