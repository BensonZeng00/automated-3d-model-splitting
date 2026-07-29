from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common

common.load_core_dependencies()

from split3mf.assembly import load_visual_semantics
from split3mf.cli import build_parser
from split3mf.common import Component, cKDTree, trimesh
from split3mf.domain import CapDecision
from split3mf.inward import (
    ParentThicknessProbe,
    boundary_cap_distances,
    classify_body_cut_loop_references,
    effective_feature_clearance,
    matched_socket_bottom_geometry,
    remap_cap_decision,
)
from split3mf.validation import (
    exterior_visibility_basis,
    fibonacci_view_directions,
    localized_short_open_edge_acceptance,
    validate_multiview_visual_consistency,
)


def tetrahedron(offset=(0.0, 0.0, 0.0)):
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    ) + np.asarray(offset, dtype=np.float64)
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    return vertices, faces


def legacy_parent_thickness_hits(
    probe: ParentThicknessProbe,
    points: np.ndarray,
    directions: np.ndarray,
    search_limit_mm: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    directions /= np.maximum(
        np.linalg.norm(directions, axis=1)[:, None],
        1e-12,
    )
    search_limit = max(float(search_limit_mm), 0.0)
    misses = search_limit + 0.05
    hits = np.full(len(points), misses, dtype=np.float64)
    tree = cKDTree(probe.centroids)
    midpoints = points + directions * (search_limit * 0.5)
    candidate_groups = tree.query_ball_point(
        midpoints,
        search_limit * 0.5 + probe.maximum_radius + 1e-8,
    )
    for index, (origin, direction, candidates) in enumerate(
        zip(points, directions, candidate_groups)
    ):
        candidate_ids = np.asarray(candidates, dtype=np.int64)
        if not len(candidate_ids):
            continue
        centroid_delta = probe.centroids[candidate_ids] - origin
        ray_projection = centroid_delta @ direction
        clamped_projection = np.clip(ray_projection, 0.0, search_limit)
        closest = origin + clamped_projection[:, None] * direction
        sphere_distance = np.linalg.norm(
            probe.centroids[candidate_ids] - closest,
            axis=1,
        )
        candidate_ids = candidate_ids[
            sphere_distance <= probe.radii[candidate_ids] + 1e-7
        ]
        if not len(candidate_ids):
            continue

        triangles = probe.triangles[candidate_ids]
        edge_1 = triangles[:, 1] - triangles[:, 0]
        edge_2 = triangles[:, 2] - triangles[:, 0]
        h = np.cross(np.broadcast_to(direction, edge_2.shape), edge_2)
        determinant = np.einsum("ij,ij->i", edge_1, h)
        active = np.abs(determinant) > 1e-12
        inverse = np.zeros_like(determinant)
        inverse[active] = 1.0 / determinant[active]
        s = origin - triangles[:, 0]
        u = inverse * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, edge_1)
        v = inverse * (q @ direction)
        distance = inverse * np.einsum("ij,ij->i", edge_2, q)
        geometric_hit = (
            active
            & (u >= -1e-9)
            & (v >= -1e-9)
            & (u + v <= 1.0 + 1e-9)
            & (distance > 1e-4)
            & (distance <= search_limit + 1e-9)
        )
        if not np.any(geometric_hit):
            continue
        hit_distances = distance[geometric_hit]
        hit_facing = probe.normals[candidate_ids][geometric_hit] @ direction
        exit_distances = hit_distances[hit_facing > 1e-6]
        if len(exit_distances):
            exit_distance = float(exit_distances.min())
            entry_distances = hit_distances[
                (hit_facing < -1e-6)
                & (hit_distances < exit_distance - 1e-5)
            ]
            entry_distance = (
                float(entry_distances.max()) if len(entry_distances) else 0.0
            )
            hits[index] = exit_distance - entry_distance
    return hits


class GeneralizedV12Tests(unittest.TestCase):
    def test_recursive_body_preserves_unmatched_parent_contact_shell_loops(self) -> None:
        loops = [[0, 1, 2, 3], [4, 5, 6]]
        global_vertex_ids = np.asarray(
            [10, 11, 12, 13, 20, 21, 22],
            dtype=np.int64,
        )
        child_ref = {
            "component_index": 2,
            "global_vertices": {10, 11, 12, 13},
            "global_edges": {
                (10, 11),
                (11, 12),
                (12, 13),
                (10, 13),
            },
        }

        loop_refs, immutable = classify_body_cut_loop_references(
            loops,
            global_vertex_ids,
            [child_ref],
            preserve_unmatched_source_geometry=True,
        )

        self.assertIs(loop_refs[0], child_ref)
        self.assertIsNone(loop_refs[1])
        self.assertEqual(
            immutable,
            [
                {
                    "loop_index": 1,
                    "vertices": 3,
                    "reason": (
                        "inherited_parent_contact_shell_or_validated_source_boundary"
                    ),
                }
            ],
        )

    def test_v13_depth_defaults_use_one_to_five_mm_safety_policy(self) -> None:
        args = build_parser().parse_args(["--input", "placeholder.3mf"])
        self.assertEqual(args.max_extension_mm, 1.0)
        self.assertEqual(args.max_planar_travel_mm, 5.0)
        self.assertIsNone(args.planar_extra_limit_mm)
        self.assertEqual(args.max_topology_defect_ratio, 0.001)

    def test_adaptive_cap_uses_total_planar_travel_budget(self) -> None:
        inward = np.array([0.0, 0.0, 1.0])
        directions = np.tile(inward, (4, 1))
        shallow_points = np.array(
            [[0, 0, 0.0], [1, 0, 0.8], [1, 1, 0.0], [0, 1, 0.8]],
            dtype=np.float64,
        )
        distances, _, record = boundary_cap_distances(
            shallow_points,
            inward,
            directions,
            fixed_depth_mm=1.0,
            flat_clearance_mm=0.0,
            cap_mode="adaptive",
            planar_extra_limit_mm=4.0,
        )
        self.assertEqual(record["cap_mode"], "flat")
        self.assertLessEqual(float(distances.max()), 5.0)
        bottom = shallow_points + distances[:, None] * inward
        np.testing.assert_allclose(bottom[:, 2], bottom[0, 2])

        deep_points = shallow_points.copy()
        deep_points[[1, 3], 2] = 3.5
        distances, _, record = boundary_cap_distances(
            deep_points,
            inward,
            directions,
            fixed_depth_mm=1.0,
            flat_clearance_mm=0.0,
            cap_mode="adaptive",
            planar_extra_limit_mm=4.0,
        )
        self.assertEqual(record["cap_mode"], "flat")
        self.assertAlmostEqual(float(distances.max()), 4.5)
        self.assertAlmostEqual(float(distances.min()), 1.0)
        self.assertEqual(record["cap_depth_reference"], "common_plane_with_variable_point_depth")

        impossible_points = shallow_points.copy()
        impossible_points[[1, 3], 2] = 4.2
        distances, _, record = boundary_cap_distances(
            impossible_points,
            inward,
            directions,
            fixed_depth_mm=1.0,
            flat_clearance_mm=0.0,
            cap_mode="adaptive",
            planar_extra_limit_mm=4.0,
        )
        self.assertEqual(record["cap_mode"], "local-offset")
        np.testing.assert_allclose(distances, 5.0)
        self.assertEqual(
            record["local_fallback_reason"],
            "required_plane_depth_exceeds_safe_maximum",
        )

        with self.assertRaisesRegex(ValueError, "cannot fit"):
            boundary_cap_distances(
                impossible_points,
                inward,
                directions,
                fixed_depth_mm=1.0,
                flat_clearance_mm=0.0,
                cap_mode="flat",
                planar_extra_limit_mm=4.0,
            )

    def test_parent_thickness_lowers_minimum_only_below_one_point_zero_five_mm(self) -> None:
        slab = trimesh.creation.box(extents=[4.0, 4.0, 0.8])
        slab.apply_translation([0.0, 0.0, 0.4])
        probe = ParentThicknessProbe(
            np.asarray(slab.vertices),
            np.asarray(slab.faces),
        )
        points = np.asarray(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        inward = np.asarray([0.0, 0.0, 1.0])
        distances, _, record = boundary_cap_distances(
            points,
            inward,
            np.tile(inward, (len(points), 1)),
            fixed_depth_mm=1.0,
            flat_clearance_mm=0.0,
            cap_mode="adaptive",
            planar_extra_limit_mm=4.0,
            parent_thickness_probe=probe,
        )
        np.testing.assert_allclose(distances, 0.75, atol=1e-8)
        self.assertAlmostEqual(record["parent_thickness_min_mm"], 0.8)
        self.assertAlmostEqual(record["safe_maximum_inward_depth_mm"], 0.75)
        self.assertAlmostEqual(record["effective_minimum_inward_depth_mm"], 0.75)

    def test_parent_thickness_radius_buckets_match_legacy_global_query(self) -> None:
        slab = trimesh.creation.box(extents=[4.0, 4.0, 0.8])
        slab.apply_translation([0.0, 0.0, 0.4])
        vertices = np.asarray(slab.vertices, dtype=np.float64)
        faces = np.asarray(slab.faces, dtype=np.int64)
        giant_vertices = np.asarray(
            [
                [100.0, 100.0, 0.0],
                [160.0, 100.0, 0.0],
                [100.0, 160.0, 0.0],
            ],
            dtype=np.float64,
        )
        giant_face = np.asarray(
            [[len(vertices), len(vertices) + 1, len(vertices) + 2]],
            dtype=np.int64,
        )
        vertices = np.vstack([vertices, giant_vertices])
        faces = np.vstack([faces, giant_face])
        probe = ParentThicknessProbe(vertices, faces)
        self.assertGreater(len(probe.radius_buckets), 1)

        points = np.asarray(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        directions = np.tile(
            np.asarray([0.0, 0.0, 1.0]),
            (len(points), 1),
        )
        expected = legacy_parent_thickness_hits(
            probe,
            points,
            directions,
            5.05,
        )
        actual = probe.first_hit_distances(points, directions, 5.05)
        np.testing.assert_array_equal(actual, expected)

    def test_parent_thickness_excludes_child_subtree_and_reports_limiting_owner(self) -> None:
        vertices = np.asarray(
            [
                [-1.0, -1.0, 0.2],
                [1.0, -1.0, 0.2],
                [0.0, 1.0, 0.2],
                [-1.0, -1.0, 2.0],
                [1.0, -1.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [[0, 1, 2], [3, 4, 5]],
            dtype=np.int64,
        )
        probe = ParentThicknessProbe(
            vertices,
            faces,
            triangle_source_face_indices=np.asarray([100, 200]),
            triangle_owner_indices=np.asarray([2, 1]),
        )
        points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
        directions = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)

        unsafe_maximum, unsafe_record = probe.safety_limit(
            points,
            directions,
            5.0,
        )
        self.assertAlmostEqual(unsafe_maximum, 0.15)
        self.assertEqual(
            unsafe_record[
                "parent_thickness_limiting_exit_owner_counts"
            ],
            {"2": 1},
        )
        self.assertEqual(
            unsafe_record[
                "parent_thickness_limiting_exit_face_indices"
            ],
            [100],
        )

        filtered = probe.excluding_triangles(
            np.asarray([0], dtype=np.int64),
            filter_context={
                "parent_part_index": 1,
                "child_part_index": 2,
                "excluded_subtree_part_indices": [2],
            },
        )
        safe_maximum, record = filtered.safety_limit(
            points,
            directions,
            5.0,
        )
        self.assertAlmostEqual(safe_maximum, 1.95)
        self.assertEqual(
            record["parent_thickness_limiting_exit_owner_counts"],
            {"1": 1},
        )
        self.assertEqual(
            record["parent_thickness_limiting_exit_face_indices"],
            [200],
        )
        self.assertEqual(
            record["parent_thickness_probe_filter"],
            {
                "parent_part_index": 1,
                "child_part_index": 2,
                "excluded_subtree_part_indices": [2],
                "excluded_triangle_count": 1,
                "included_triangle_count": 1,
                "excluded_owner_indices": [2],
            },
        )
        self.assertEqual(probe.active_triangle_count, 2)

    def test_parent_thickness_discards_only_sparse_coincident_hits(self) -> None:
        class StubProbe(ParentThicknessProbe):
            def __init__(self, thicknesses):
                self.thicknesses = np.asarray(thicknesses, dtype=np.float64)

            def first_hit_distances(self, points, directions, search_limit_mm):
                return self.thicknesses.copy()

        points = np.zeros((200, 3), dtype=np.float64)
        directions = np.tile(np.asarray([0.0, 0.0, 1.0]), (len(points), 1))
        sparse_probe = StubProbe([0.001] + [3.0] * 199)
        safe_maximum, record = sparse_probe.safety_limit(points, directions, 5.0)
        self.assertAlmostEqual(safe_maximum, 2.95)
        self.assertTrue(record["parent_thickness_isolated_coincident_hits_discarded"])

        repeated_probe = StubProbe([0.001] * 2 + [3.0] * 198)
        safe_maximum, record = repeated_probe.safety_limit(points, directions, 5.0)
        self.assertEqual(safe_maximum, 0.0)
        self.assertFalse(record["parent_thickness_isolated_coincident_hits_discarded"])

    def test_parent_thickness_discards_clustered_unpaired_surface_hits_only(self) -> None:
        class ClassifiedStubProbe(ParentThicknessProbe):
            def __init__(self, thicknesses, preceding_entries):
                self.thicknesses = np.asarray(thicknesses, dtype=np.float64)
                self.preceding_entries = np.asarray(
                    preceding_entries,
                    dtype=bool,
                )

            def first_hit_distances(self, points, directions, search_limit_mm):
                self.last_hit_had_preceding_entry = (
                    self.preceding_entries.copy()
                )
                return self.thicknesses.copy()

        points = np.zeros((200, 3), dtype=np.float64)
        directions = np.tile(
            np.asarray([0.0, 0.0, 1.0]),
            (len(points), 1),
        )
        unpaired_probe = ClassifiedStubProbe(
            [0.001] * 20 + [3.0] * 180,
            [False] * 200,
        )
        safe_maximum, record = unpaired_probe.safety_limit(
            points,
            directions,
            5.0,
        )
        self.assertAlmostEqual(safe_maximum, 2.95)
        self.assertEqual(
            record[
                "parent_thickness_unpaired_surface_hit_vertices_discarded"
            ],
            20,
        )
        self.assertEqual(
            record["parent_thickness_remaining_coincident_hit_vertices"],
            0,
        )

        paired_probe = ClassifiedStubProbe(
            [0.001] * 20 + [3.0] * 180,
            [True] * 20 + [False] * 180,
        )
        safe_maximum, record = paired_probe.safety_limit(
            points,
            directions,
            5.0,
        )
        self.assertEqual(safe_maximum, 0.0)
        self.assertEqual(
            record[
                "parent_thickness_unpaired_surface_hit_vertices_discarded"
            ],
            0,
        )
        self.assertEqual(
            record["parent_thickness_remaining_coincident_hit_vertices"],
            20,
        )

    def test_localized_short_open_edges_are_ratio_accepted_but_not_other_defects(self) -> None:
        finding = {
            "open_edges": 187,
            "over_shared_edges": 0,
            "inconsistent_shared_edges": 0,
            "winding_consistent": True,
            "topology_defect_ratio": 0.0019635,
            "open_edge_metrics": {
                "max_length_mm": 0.181,
                "total_length_mm": 2.478,
            },
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [26.07, 18.77, 15.04],
        }
        accepted, limits = localized_short_open_edge_acceptance(finding, 0.001)
        self.assertTrue(accepted)
        self.assertEqual(limits["extended_ratio_limit"], 0.002)

        finding["inconsistent_shared_edges"] = 1
        accepted, _ = localized_short_open_edge_acceptance(finding, 0.001)
        self.assertFalse(accepted)

    def test_feature_adaptive_clearance_clamps_small_part(self) -> None:
        component = Component(
            color_code="pink",
            global_faces=np.arange(4),
            face_count=4,
            area=1.0,
            bbox_min=np.asarray([0.0, 0.0, 0.0]),
            bbox_max=np.asarray([10.0, 2.0, 5.0]),
            center=np.asarray([5.0, 1.0, 2.5]),
        )
        effective, record = effective_feature_clearance(component, 0.30)
        self.assertAlmostEqual(effective, 0.08)
        self.assertEqual(record["reason"], "feature_scale_clamp")

    def test_socket_reuses_common_child_plane_with_clearance_inside_safe_maximum(self) -> None:
        inward = np.array([0.0, 0.0, 1.0])
        points = np.array(
            [[0, 0, 0.0], [1, 0, 1.2], [1, 1, 0.0], [0, 1, 1.2]],
            dtype=np.float64,
        )
        bottom, record = matched_socket_bottom_geometry(
            source_boundary_points=points,
            socket_top_points=points,
            inward=inward,
            inward_directions=np.tile(inward, (4, 1)),
            child_insert_shrink_mm=0.0,
            max_extension_mm=1.0,
            flat_clearance_mm=0.0,
            cap_mode="adaptive",
            planar_extra_limit_mm=4.0,
            bottom_clearance_mm=0.1,
        )
        np.testing.assert_allclose(bottom[:, 2], 2.3)
        self.assertLessEqual(float(np.linalg.norm(bottom - points, axis=1).max()), 5.0)
        self.assertEqual(record["socket_cap_field_source"], "matched_child_cap")
        self.assertAlmostEqual(record["socket_bottom_clearance_applied_mm"], 0.1)

    def test_parent_socket_consumes_prevalidated_child_cap_without_replanning(self) -> None:
        class ReplanningMustNotRun:
            def safety_limit(self, *_args, **_kwargs):
                raise AssertionError("parent socket replanned an existing child cap")

        inward = np.array([0.0, 0.0, 1.0])
        child_points = np.array(
            [[0, 0, 0.0], [1, 0, 1.2], [1, 1, 0.0], [0, 1, 1.2]],
            dtype=np.float64,
        )
        source_vertex_ids = (10, 11, 12, 13)
        child_plane_z = 2.2
        decision = CapDecision(
            mode="flat",
            source_vertex_ids=source_vertex_ids,
            fit_points=child_points,
            directions=np.tile(inward, (4, 1)),
            distances=child_plane_z - child_points[:, 2],
            record={
                "requested_cap_mode": "adaptive",
                "cap_mode": "flat",
                "flat_bottom_plane_s": child_plane_z,
            },
        )

        reverse_order = np.asarray([3, 2, 1, 0], dtype=np.int64)
        socket_points = child_points[reverse_order]
        socket_vertex_ids = tuple(
            source_vertex_ids[index] for index in reverse_order
        )
        bottom, record = matched_socket_bottom_geometry(
            source_boundary_points=socket_points,
            socket_top_points=socket_points,
            inward=inward,
            inward_directions=np.tile(inward, (4, 1)),
            child_insert_shrink_mm=0.0,
            max_extension_mm=1.0,
            flat_clearance_mm=0.0,
            cap_mode="adaptive",
            planar_extra_limit_mm=4.0,
            bottom_clearance_mm=0.0,
            parent_thickness_probe=ReplanningMustNotRun(),
            cap_decision=decision,
            source_vertex_ids=socket_vertex_ids,
        )

        np.testing.assert_allclose(bottom[:, 2], child_plane_z)
        self.assertEqual(record["cap_mode"], "flat")
        self.assertEqual(
            record["socket_cap_field_source"],
            "prevalidated_shared_child_cap",
        )
        self.assertEqual(
            record["cap_decision_source"],
            "prevalidated_shared_child",
        )

    def test_shared_cap_decision_rejects_different_parent_boundary(self) -> None:
        decision = CapDecision(
            mode="flat",
            source_vertex_ids=(10, 11, 12),
            fit_points=np.zeros((3, 3), dtype=np.float64),
            directions=np.tile(np.array([0.0, 0.0, 1.0]), (3, 1)),
            distances=np.ones(3),
            record={"cap_mode": "flat"},
        )
        with self.assertRaisesRegex(
            ValueError,
            "matching parent socket and child cap source vertices differ",
        ):
            remap_cap_decision(decision, (10, 11, 99))

    def test_visual_validation_accepts_identical_surface(self) -> None:
        vertices, faces = tetrahedron()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        report = validate_multiview_visual_consistency(
            vertices,
            faces,
            np.ones(len(faces), dtype=np.int32),
            [{"mesh": mesh, "source_surface_face_count": len(faces)}],
            view_count=8,
            resolution=64,
            max_intrusion_ratio=0.0,
            max_material_mismatch_ratio=0.0,
            min_coverage_ratio=1.0,
        )
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["generated_surface_intrusion_ratio"], 0.0)
        self.assertEqual(report["front_material_mismatch_ratio"], 0.0)

    def test_visual_validation_rejects_front_intrusion(self) -> None:
        direction = fibonacci_view_directions(8)[0]
        u, v = exterior_visibility_basis(direction)
        if float(np.dot(np.cross(u, v), direction)) < 0.0:
            u, v = v, u
        vertices = np.asarray([u + v, -u + v, -v], dtype=np.float64)
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        shifted_vertices = vertices + direction * 0.5
        combined_vertices = np.vstack((vertices, shifted_vertices))
        combined_faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        shifted = trimesh.Trimesh(
            vertices=combined_vertices,
            faces=combined_faces,
            process=False,
        )
        report = validate_multiview_visual_consistency(
            vertices,
            faces,
            np.ones(len(faces), dtype=np.int32),
            [{"mesh": shifted, "source_surface_face_count": 1}],
            view_count=8,
            resolution=64,
            depth_tolerance_mm=0.01,
            max_intrusion_ratio=0.0,
            max_material_mismatch_ratio=1.0,
            min_coverage_ratio=0.0,
        )
        self.assertFalse(report["valid"])
        self.assertGreater(report["generated_surface_intrusion_ratio"], 0.0)

        advisory_report = validate_multiview_visual_consistency(
            vertices,
            faces,
            np.ones(len(faces), dtype=np.int32),
            [{"mesh": shifted, "source_surface_face_count": 1}],
            view_count=8,
            resolution=64,
            depth_tolerance_mm=0.01,
            max_intrusion_ratio=1.0,
            max_material_mismatch_ratio=1.0,
            min_coverage_ratio=0.0,
        )
        self.assertTrue(advisory_report["valid"], advisory_report["errors"])
        self.assertTrue(advisory_report["advisories"])

    def test_visual_validation_accepts_internal_generated_surface(self) -> None:
        vertices, faces = tetrahedron()
        inner_vertices = vertices * 0.5 + np.asarray([0.1, 0.1, 0.1])
        combined_vertices = np.vstack((vertices, inner_vertices))
        combined_faces = np.vstack((faces, faces + len(vertices)))
        mesh = trimesh.Trimesh(
            vertices=combined_vertices,
            faces=combined_faces,
            process=False,
        )
        report = validate_multiview_visual_consistency(
            vertices,
            faces,
            np.ones(len(faces), dtype=np.int32),
            [{"mesh": mesh, "source_surface_face_count": len(faces)}],
            view_count=8,
            resolution=64,
            depth_tolerance_mm=0.01,
            max_intrusion_ratio=0.0,
            max_material_mismatch_ratio=0.0,
            min_coverage_ratio=1.0,
        )
        self.assertTrue(report["valid"], report["errors"])

    def test_visual_validation_rejects_local_wrong_material_cover(self) -> None:
        source_vertices = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [3, 0, 0],
                [4, 0, 0],
                [3, 1, 0],
            ],
            dtype=np.float64,
        )
        source_faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        first = trimesh.Trimesh(
            vertices=source_vertices[:3],
            faces=np.asarray([[0, 1, 2]], dtype=np.int64),
            process=False,
        )
        second_vertices = np.vstack(
            (source_vertices[3:], source_vertices[:3] + np.asarray([0, 0, 0.5]))
        )
        second = trimesh.Trimesh(
            vertices=second_vertices,
            faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
            process=False,
        )
        report = validate_multiview_visual_consistency(
            source_vertices,
            source_faces,
            np.asarray([1, 2], dtype=np.int32),
            [
                {"mesh": first, "source_surface_face_count": 1},
                {"mesh": second, "source_surface_face_count": 1},
            ],
            view_count=16,
            resolution=64,
            depth_tolerance_mm=0.01,
            max_intrusion_ratio=1.0,
            max_material_mismatch_ratio=1.0,
            max_local_material_mismatch_ratio=0.0,
            min_coverage_ratio=0.0,
        )
        self.assertFalse(report["valid"])
        self.assertTrue(report["blocking_local_material_mismatch_pairs"])
        pair = report["blocking_local_material_mismatch_pairs"][0]
        self.assertEqual(pair["source_part_index"], 1)
        self.assertEqual(pair["generated_part_index"], 2)

    def test_visual_semantics_loads_direction_overrides(self) -> None:
        payload = {
            "parts": {
                "P04": {
                    "label": "tongue",
                    "confidence": "HIGH",
                    "force_inward_vector": [0, 1, 0],
                    "force_parent_direction": True,
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semantics.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_visual_semantics(path)
        self.assertEqual(loaded["parts"][4]["force_inward_vector"], [0, 1, 0])
        self.assertTrue(loaded["parts"][4]["force_parent_direction"])


if __name__ == "__main__":
    unittest.main()
