from __future__ import annotations

import math
import importlib
import collections
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common  # noqa: E402

common.load_core_dependencies()
inward_module = importlib.import_module("split3mf.inward")  # noqa: E402
mesh_module = importlib.import_module("split3mf.mesh")  # noqa: E402
local_connector_module = importlib.import_module("split3mf.local_connectors")  # noqa: E402
connector_topology_module = importlib.import_module(
    "split3mf.connector_topology"
)  # noqa: E402
connector_geometry_module = importlib.import_module(
    "split3mf.connector_geometry"
)  # noqa: E402
connector_surface_module = importlib.import_module(
    "split3mf.connector_surface"
)  # noqa: E402

from split3mf.local_connectors import (  # noqa: E402
    LocalConnectorSpec,
    assembly_interface_policy,
    authoritative_complete_child_clearance_mm,
    build_boolean_cutter_proxy_from_final_part,
    build_clearance_cutter_from_final_part,
    build_socket_cutter_from_plan,
    build_synthetic_local_connector_pair,
    connector_profile,
    plan_local_connector,
    subtract_socket_cutters,
)
from split3mf.boolean_cutters import assess_source_patch_reentry  # noqa: E402
from split3mf.inward import (  # noqa: E402
    add_local_female_boolean_closure,
    add_local_female_socket_and_backing,
    add_local_male_connector_and_backing,
    _add_constrained_connector_annulus,
    _line_preserving_inset_displacements,
    _printable_backing_rings,
    build_local_male_attachment_cutter,
    local_connector_safe_depth_at_footprint,
    local_connector_safe_depth_from_field,
    local_connector_spec_for_interface,
    preflight_local_connector_patch,
)
from split3mf.validation import (  # noqa: E402
    BooleanFinalizationPolicy,
    boolean_collapsed_face_audit,
    finalize_boolean_difference_mesh,
)


class LocalConnectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = LocalConnectorSpec(
            peg_width_mm=4.0,
            peg_length_mm=4.0,
            engagement_depth_mm=5.0,
            total_clearance_mm=0.50,
            socket_bottom_clearance_mm=0.25,
            socket_mouth_chamfer_mm=0.80,
            peg_tip_chamfer_mm=0.0,
            corner_radius_mm=0.45,
        )

    def test_profile_is_local_not_full_boundary_skirt(self) -> None:
        profile = connector_profile(self.spec, contact_width_mm=14.0, contact_length_mm=14.0)

        self.assertAlmostEqual(profile["peg_area_mm2"], 16.0, places=6)
        self.assertLess(profile["peg_area_ratio"], 0.10)
        self.assertAlmostEqual(profile["engagement_depth_mm"], 5.0, places=6)
        self.assertLessEqual(profile["full_boundary_backing_depth_mm"], 1.0)

    def test_shallow_minimal_closure_requires_both_user_reviewed_advisories(self) -> None:
        policy = connector_geometry_module._user_reviewed_shallow_minimal_closure_is_eligible
        reviewed = {
            "compact_peg_enabled": False,
            "backing_slope_validation_mode": "advisory",
            "backing_surface_validation_mode": "advisory",
        }

        self.assertTrue(
            policy(reviewed, boundary_vertex_count=4, backing_depth_mm=0.0213)
        )
        self.assertFalse(
            policy(reviewed, boundary_vertex_count=5, backing_depth_mm=0.0213)
        )
        self.assertFalse(
            policy(reviewed, boundary_vertex_count=4, backing_depth_mm=0.051)
        )
        self.assertFalse(
            policy(
                {**reviewed, "backing_slope_validation_mode": "strict"},
                boundary_vertex_count=4,
                backing_depth_mm=0.0213,
            )
        )

    def test_shallow_four_point_boundary_uses_locked_axial_closure(self) -> None:
        boundary = np.asarray(
            [
                [-0.01, -0.01, 0.0],
                [0.01, -0.01, 0.0],
                [0.01, 0.01, 0.0],
                [-0.01, 0.01, 0.0],
            ],
            dtype=np.float64,
        )
        plan = {
            "inward": np.asarray([0.0, 0.0, -1.0]),
            "center": np.zeros(3, dtype=np.float64),
            "u": np.asarray([1.0, 0.0, 0.0]),
            "v": np.asarray([0.0, 1.0, 0.0]),
            "full_boundary_backing_depth_mm": 0.0213,
            "compact_peg_enabled": False,
            "backing_slope_validation_mode": "advisory",
            "backing_surface_validation_mode": "advisory",
        }

        lead, backing, taper = connector_geometry_module.printable_backing_rings(
            boundary,
            plan,
            child_clearance=True,
        )

        np.testing.assert_allclose(lead, backing, atol=1e-12)
        np.testing.assert_allclose(lead[:, :2], boundary[:, :2], atol=1e-12)
        np.testing.assert_allclose(lead[:, 2], -0.0213, atol=1e-12)
        self.assertAlmostEqual(taper, 0.0213)
        self.assertTrue(plan["shallow_minimal_closure_advisory_accepted"])

    def test_shallow_minimal_closure_accepts_bounded_side_needle(self) -> None:
        vertices = [
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0.001, 0.0, 0.0]),
            np.asarray([3.847, 0.007182, -0.0019]),
        ]
        plan = {
            "compact_peg_enabled": False,
            "backing_slope_validation_mode": "advisory",
            "backing_surface_validation_mode": "advisory",
            "full_boundary_backing_depth_mm": 0.0019,
            "backing_source_ring_vertices": 3,
            "backing_strip_audits": [
                {
                    "valid": True,
                    "degenerate_face_count": 0,
                    "invalid_bridge_count": 0,
                    "maximum_fanout": 2,
                    "maximum_cross_edge_mm": 3.847,
                }
            ],
        }

        record = inward_module._validate_connector_wedge_faces(
            vertices,
            [[0, 1, 2]],
            0,
            shallow_minimal_closure_plan=plan,
        )

        self.assertEqual(
            record["backing_wedge_shallow_minimal_advisory_needle_faces"],
            1,
        )
        self.assertFalse(
            plan["shallow_minimal_closure_needle_advisory_checks"][
                "minimal_closure_preapproved"
            ]
        )

    def test_shallow_minimal_closure_does_not_waive_invalid_bridge(self) -> None:
        plan = {
            "compact_peg_enabled": False,
            "backing_slope_validation_mode": "advisory",
            "backing_surface_validation_mode": "advisory",
            "full_boundary_backing_depth_mm": 0.0019,
            "backing_source_ring_vertices": 3,
            "shallow_minimal_closure_advisory_accepted": True,
            "backing_strip_audits": [
                {
                    "valid": True,
                    "degenerate_face_count": 0,
                    "invalid_bridge_count": 1,
                    "maximum_fanout": 2,
                    "maximum_cross_edge_mm": 3.847,
                }
            ],
        }
        vertices = [
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0.001, 0.0, 0.0]),
            np.asarray([3.847, 0.007182, -0.0019]),
        ]

        with self.assertRaisesRegex(ValueError, "needle faces"):
            inward_module._validate_connector_wedge_faces(
                vertices,
                [[0, 1, 2]],
                0,
                shallow_minimal_closure_plan=plan,
            )

    def test_taper_audit_accepts_sparse_isolated_projection_outliers(self) -> None:
        angles = np.full(500, 45.0, dtype=np.float64)
        angles[[17, 263]] = np.asarray([0.304, 80.2])

        audit = connector_geometry_module.summarize_backing_taper_angle_samples(
            angles
        )

        self.assertTrue(audit["valid"])
        self.assertTrue(audit["accepted_sparse_outliers"])
        self.assertEqual(audit["outside_count"], 2)
        self.assertEqual(audit["maximum_cyclic_outlier_run"], 1)
        self.assertAlmostEqual(audit["median_degrees"], 45.0)

    def test_taper_audit_rejects_sustained_out_of_range_arc(self) -> None:
        angles = np.full(500, 45.0, dtype=np.float64)
        angles[120:131] = 12.0

        audit = connector_geometry_module.summarize_backing_taper_angle_samples(
            angles
        )

        self.assertFalse(audit["valid"])
        self.assertGreater(
            audit["maximum_cyclic_outlier_run"],
            audit["maximum_allowed_outlier_run"],
        )

    def test_connector_spec_rejects_unknown_slope_validation_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "slope_validation_mode"):
            LocalConnectorSpec(slope_validation_mode="hopeful")

    def test_connector_plan_preserves_advisory_slope_validation_mode(self) -> None:
        boundary = np.asarray(
            [
                [-5.0, -5.0, 0.0],
                [5.0, -5.0, 0.0],
                [5.0, 5.0, 0.0],
                [-5.0, 5.0, 0.0],
            ],
            dtype=np.float64,
        )
        plan = plan_local_connector(
            boundary,
            np.asarray([0.0, 0.0, -1.0]),
            LocalConnectorSpec(slope_validation_mode="advisory"),
        )

        self.assertEqual(plan["backing_slope_validation_mode"], "advisory")

    def test_tiny_interface_can_plan_backing_without_compact_peg(self) -> None:
        boundary = np.asarray(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ],
            dtype=np.float64,
        )
        spec = LocalConnectorSpec(
            engagement_depth_mm=0.0,
            socket_bottom_clearance_mm=0.0,
            peg_tip_chamfer_mm=0.0,
            full_boundary_backing_depth_mm=0.45,
            compact_peg_enabled=False,
        )

        plan = plan_local_connector(
            boundary,
            np.asarray([0.0, 0.0, -1.0]),
            spec,
            samples=32,
        )

        self.assertFalse(plan["compact_peg_enabled"])
        self.assertAlmostEqual(plan["full_boundary_backing_depth_mm"], 0.45)
        self.assertAlmostEqual(plan["peg_width_mm"], 0.0)

    def test_triangular_auxiliary_loop_can_plan_no_peg_backing(self) -> None:
        boundary = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.0, 0.6, 0.0],
            ],
            dtype=np.float64,
        )
        spec = LocalConnectorSpec(
            engagement_depth_mm=0.0,
            socket_bottom_clearance_mm=0.0,
            peg_tip_chamfer_mm=0.0,
            full_boundary_backing_depth_mm=0.45,
            compact_peg_enabled=False,
        )

        plan = plan_local_connector(
            boundary,
            np.asarray([0.0, 0.0, -1.0]),
            spec,
            samples=32,
        )

        self.assertFalse(plan["compact_peg_enabled"])
        self.assertGreater(plan["edge_clearance_mm"], 0.0)
        self.assertAlmostEqual(plan["full_boundary_backing_depth_mm"], 0.45)

        output_vertices = [point.copy() for point in boundary]
        output_faces = [[0, 1, 2]]
        record = add_local_male_connector_and_backing(
            output_vertices=output_vertices,
            output_faces=output_faces,
            boundary_ids=[0, 1, 2],
            boundary_points=boundary,
            inward=np.asarray([0.0, 0.0, -1.0]),
            spec=spec,
            boolean_cutters=[],
        )
        self.assertGreater(record["backing_faces_added"], 0)
        self.assertGreater(record["connector_cap_faces_added"], 0)

    def test_thin_triangular_auxiliary_loop_keeps_measured_no_peg_backing(self) -> None:
        boundary = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.0, 0.6, 0.0],
            ],
            dtype=np.float64,
        )
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=0.427,
            safe_backing_depth_mm=0.022,
            compact_peg_supported=False,
        )

        self.assertFalse(spec.compact_peg_enabled)
        self.assertAlmostEqual(spec.full_boundary_backing_depth_mm, 0.022)
        self.assertAlmostEqual(spec.minimum_elastic_backing_depth_mm, 0.0)

        output_vertices = [point.copy() for point in boundary]
        output_faces = [[0, 1, 2]]
        record = add_local_male_connector_and_backing(
            output_vertices=output_vertices,
            output_faces=output_faces,
            boundary_ids=[0, 1, 2],
            boundary_points=boundary,
            inward=np.asarray([0.0, 0.0, -1.0]),
            spec=spec,
            boolean_cutters=[],
        )
        self.assertGreater(record["backing_faces_added"], 0)
        self.assertGreater(record["connector_cap_faces_added"], 0)

    def test_tiny_interface_safety_omits_peg_but_keeps_backing(self) -> None:
        class Probe:
            def safety_limit(self, points, directions, requested_depth_mm):
                del points, directions, requested_depth_mm
                return 2.0, {"safe_maximum_inward_depth_mm": 2.0}

        boundary = np.asarray(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ],
            dtype=np.float64,
        )
        safety = local_connector_safe_depth_at_footprint(
            boundary_points=boundary,
            inward=np.asarray([0.0, 0.0, -1.0]),
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            boundary_distances=np.full(len(boundary), 1.0),
            parent_thickness_probe=Probe(),
        )
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=float(
                safety["local_connector_safety_budget_mm"]
            ),
            safe_backing_depth_mm=float(
                safety["local_connector_backing_safety_limit_mm"]
            ),
            compact_peg_supported=bool(
                safety["local_connector_compact_peg_supported"]
            ),
        )

        self.assertTrue(safety["local_connector_footprint_plan_fallback"])
        self.assertFalse(safety["local_connector_compact_peg_supported"])
        self.assertFalse(spec.compact_peg_enabled)
        self.assertAlmostEqual(spec.engagement_depth_mm, 0.0)
        self.assertAlmostEqual(spec.socket_bottom_clearance_mm, 0.0)
        self.assertAlmostEqual(spec.full_boundary_backing_depth_mm, 0.5)

    def test_grazing_rim_does_not_block_interior_footprint_measurement(self) -> None:
        class Probe:
            def safety_limit(self, points, directions, requested_depth_mm):
                del points, directions, requested_depth_mm
                return 2.0, {"safe_maximum_inward_depth_mm": 2.0}

        boundary = np.asarray(
            [
                [-3.0, -3.0, 0.0],
                [3.0, -3.0, 0.0],
                [3.0, 3.0, 0.0],
                [-3.0, 3.0, 0.0],
            ],
            dtype=np.float64,
        )

        safety = local_connector_safe_depth_at_footprint(
            boundary_points=boundary,
            inward=np.asarray([0.0, 0.0, -1.0]),
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            boundary_distances=np.asarray([0.033, 2.0, 2.0, 2.0]),
            parent_thickness_probe=Probe(),
        )

        self.assertAlmostEqual(safety["boundary_safety_minimum_mm"], 0.033)
        self.assertGreaterEqual(
            safety["local_connector_backing_safety_limit_mm"],
            0.45,
        )
        self.assertTrue(safety["local_connector_footprint_probe_applied"])

    def test_zero_engagement_backing_keeps_printable_sloped_wedge(self) -> None:
        samples = 320
        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        radius = 8.0 + 1.5 * np.cos(5.0 * angles)
        boundary = np.column_stack(
            (
                radius * np.cos(angles),
                radius * np.sin(angles),
                0.35 * np.sin(3.0 * angles),
            )
        )
        spec = LocalConnectorSpec(
            engagement_depth_mm=0.0,
            socket_bottom_clearance_mm=0.0,
            peg_tip_chamfer_mm=0.0,
            full_boundary_backing_depth_mm=0.75,
            compact_peg_enabled=False,
        )
        plan = plan_local_connector(
            boundary,
            np.asarray([0.0, 0.0, -1.0]),
            spec,
            samples=32,
        )

        lead, backing, taper = _printable_backing_rings(
            boundary,
            plan,
            child_clearance=True,
        )

        self.assertGreaterEqual(len(lead), 3)
        self.assertEqual(len(backing), len(lead))
        self.assertAlmostEqual(taper, 0.75)
        projected_source = inward_module._project_connector_points(boundary, plan)
        projected_backing = inward_module._project_connector_points(backing, plan)
        self.assertLess(
            float(np.ptp(projected_backing[:, 0])),
            float(np.ptp(projected_source[:, 0])),
        )
        backing_axial = (
            backing - np.asarray(plan["center"])[None, :]
        ) @ np.asarray(plan["inward"])
        self.assertLess(float(np.ptp(backing_axial)), 1e-9)
        minimum, _, maximum = (
            connector_geometry_module.measured_backing_taper_angles(
                boundary,
                backing,
                plan,
            )
        )
        self.assertGreaterEqual(minimum, 30.0 - 1e-6)
        self.assertLessEqual(maximum, 75.0 + 1e-6)
        self.assertTrue(plan["backing_planar_slope_bias_applied"])
        self.assertEqual(
            plan["backing_inset_method"],
            "clipper2_trimmed_constant_offset",
        )

    def test_interface_policy_separates_boolean_and_shared_template_modes(self) -> None:
        boundary_policy = assembly_interface_policy("boundary-extrusion")
        local_policy = assembly_interface_policy("local-connector")
        self.assertFalse(boundary_policy.complete_child_boolean)
        self.assertEqual(
            boundary_policy.geometry_source,
            "shared_source_patch_complementary_surfaces",
        )
        self.assertTrue(local_policy.complete_child_boolean)
        self.assertAlmostEqual(local_policy.exterior_overshoot_mm, 0.01)
        self.assertEqual(
            local_policy.geometry_source,
            "complete_emitted_child_boolean_proxy",
        )
        self.assertAlmostEqual(local_policy.authoritative_clearance_mm, 0.0)
        self.assertAlmostEqual(
            authoritative_complete_child_clearance_mm("boundary-extrusion"),
            0.0,
        )

    def test_vectorized_polygon_queries_match_scalar_contract(self) -> None:
        polygon = np.asarray(
            [
                [-5.0, -4.0],
                [4.0, -4.0],
                [5.0, 1.0],
                [1.0, 4.0],
                [-4.0, 3.0],
            ],
            dtype=np.float64,
        )
        query = np.asarray(
            [
                [-6.0, 0.0],
                [-3.0, 0.0],
                [0.0, 0.0],
                [3.5, 2.0],
                [6.0, 0.0],
            ],
            dtype=np.float64,
        )
        scalar_inside = np.asarray(
            [
                local_connector_module._point_in_polygon(point, polygon)
                for point in query
            ],
            dtype=bool,
        )
        batch_inside = local_connector_module._points_in_polygon_batch(
            query,
            polygon,
        )
        scalar_distances = np.asarray(
            [
                min(
                    local_connector_module._point_segment_distance(
                        point,
                        polygon[index],
                        polygon[(index + 1) % len(polygon)],
                    )
                    for index in range(len(polygon))
                )
                for point in query
            ]
        )
        batch_distances = (
            local_connector_module._minimum_point_to_polygon_segment_distances(
                query,
                polygon,
            )
        )

        np.testing.assert_array_equal(batch_inside, scalar_inside)
        np.testing.assert_allclose(batch_distances, scalar_distances, atol=1e-12)
        self.assertAlmostEqual(
            authoritative_complete_child_clearance_mm("local-connector"),
            0.0,
        )

    def test_boolean_finalizer_ratio_accepts_sparse_collapsed_faces(self) -> None:
        # Ratio mode may preserve one numerical seam face in a sufficiently
        # dense, otherwise closed mesh.  The same face remains rejected when
        # the explicit ratio budget is absent.
        sphere = common.trimesh.creation.icosphere(subdivisions=4, radius=8.0)
        sliver = common.trimesh.Trimesh(
            vertices=np.asarray(
                [[20.0, 0.0, 0.0], [21.0, 0.0, 0.0],
                 [22.0, 1e-15, 0.0], [20.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            faces=np.asarray(
                [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
                dtype=np.int64,
            ),
            process=False,
        )
        mesh = common.trimesh.util.concatenate((sphere, sliver))
        common.trimesh.repair.fix_winding(mesh)
        common.trimesh.repair.fix_normals(mesh)

        self.assertTrue(mesh.is_watertight)
        with self.assertRaises(ValueError):
            finalize_boolean_difference_mesh(mesh)

        preserved = finalize_boolean_difference_mesh(
            mesh,
            policy=BooleanFinalizationPolicy(
                ratio_accepted_collapsed_face_budget=1,
                maximum_collapsed_face_ratio=0.001,
            ),
        )
        audit = preserved.metadata["boolean_collapsed_face_audit"]
        self.assertTrue(audit["ratio_accepted"])
        self.assertEqual(audit["new_collapsed_face_count"], 1)

    def test_boolean_audit_ratio_accepts_sparse_collapsed_faces(self) -> None:
        sphere = common.trimesh.creation.icosphere(subdivisions=4, radius=8.0)
        sliver = common.trimesh.Trimesh(
            vertices=np.asarray(
                [[20.0, 0.0, 0.0], [21.0, 0.0, 0.0],
                 [22.0, 1e-15, 0.0], [20.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            faces=np.asarray(
                [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
                dtype=np.int64,
            ),
            process=False,
        )
        mesh = common.trimesh.util.concatenate((sphere, sliver))

        valid, record = local_connector_module._audit_boolean_candidate(
            mesh,
            reference_volume=abs(float(mesh.volume)),
            movement_tolerance_mm=0.0,
            maximum_new_collapsed_face_ratio=0.001,
        )

        self.assertTrue(valid)
        self.assertTrue(record["ratio_accepted_collapsed_faces"])
        self.assertEqual(record["new_collapsed_faces"], 1)

    def test_boolean_audit_bounds_micro_seam_volume_roundoff(self) -> None:
        sphere = common.trimesh.creation.icosphere(subdivisions=4, radius=8.0)
        sliver = common.trimesh.Trimesh(
            vertices=np.asarray(
                [
                    [20.0, 0.0, 0.0],
                    [21.0, 0.0, 0.0],
                    [22.0, 1e-15, 0.0],
                    [20.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray(
                [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
                dtype=np.int64,
            ),
            process=False,
        )
        mesh = common.trimesh.util.concatenate((sphere, sliver))
        common.trimesh.repair.fix_winding(mesh)
        common.trimesh.repair.fix_normals(mesh)
        volume = abs(float(mesh.volume))

        valid, record = local_connector_module._audit_boolean_candidate(
            mesh.copy(),
            reference_volume=volume + 4e-6,
            movement_tolerance_mm=0.0,
            maximum_new_collapsed_face_ratio=0.001,
        )
        rejected, rejected_record = local_connector_module._audit_boolean_candidate(
            mesh.copy(),
            reference_volume=volume + 2e-5,
            movement_tolerance_mm=0.0,
            maximum_new_collapsed_face_ratio=0.001,
        )
        reviewed_valid, reviewed_record = (
            local_connector_module._audit_boolean_candidate(
                mesh.copy(),
                reference_volume=volume + 2e-5,
                movement_tolerance_mm=0.0,
                maximum_new_collapsed_face_ratio=0.001,
                collapsed_seam_volume_envelope_cap_mm3=5e-5,
            )
        )

        self.assertTrue(valid)
        self.assertGreaterEqual(record["volume_tolerance_mm3"], 4e-6)
        self.assertLessEqual(record["collapsed_seam_volume_envelope_mm3"], 1e-5)
        self.assertFalse(rejected)
        self.assertTrue(reviewed_valid)
        self.assertLessEqual(
            reviewed_record["volume_delta_mm3"],
            reviewed_record["volume_tolerance_mm3"],
        )
        self.assertGreater(
            rejected_record["volume_delta_mm3"],
            rejected_record["volume_tolerance_mm3"],
        )

    def test_boolean_audit_allows_bounded_topology_healthy_export_roundoff(self) -> None:
        box = common.trimesh.creation.box(extents=[4.0, 4.0, 4.0])
        reference_volume = abs(float(box.volume)) + 0.004

        strict_valid, strict_record = local_connector_module._audit_boolean_candidate(
            box.copy(),
            reference_volume=reference_volume,
            movement_tolerance_mm=1e-9,
        )
        reviewed_valid, reviewed_record = (
            local_connector_module._audit_boolean_candidate(
                box.copy(),
                reference_volume=reference_volume,
                movement_tolerance_mm=1e-9,
                topology_healthy_export_volume_envelope_cap_mm3=5e-3,
            )
        )
        open_box = box.copy()
        open_box.update_faces(np.arange(len(open_box.faces) - 1))
        open_valid, open_record = local_connector_module._audit_boolean_candidate(
            open_box,
            reference_volume=abs(float(open_box.volume)),
            movement_tolerance_mm=1e-9,
            topology_healthy_export_volume_envelope_cap_mm3=5e-3,
        )

        self.assertFalse(strict_valid)
        self.assertEqual(strict_record["topology_healthy_export_volume_envelope_mm3"], 0.0)
        self.assertTrue(reviewed_valid)
        self.assertEqual(reviewed_record["degenerate_faces"], 0)
        self.assertLessEqual(
            reviewed_record["volume_delta_mm3"],
            reviewed_record["volume_tolerance_mm3"],
        )
        self.assertFalse(open_valid)
        self.assertFalse(open_record["watertight"])
        self.assertEqual(open_record["topology_healthy_export_volume_envelope_mm3"], 0.0)

    def test_boolean_cleanup_volume_reference_deducts_removed_micro_shells(self) -> None:
        reference = local_connector_module._reference_volume_after_redundant_shell_cleanup(
            12.5,
            {"applied": True, "removed_volume_mm3": 4.25e-6},
        )
        unchanged = local_connector_module._reference_volume_after_redundant_shell_cleanup(
            12.5,
            {"applied": False, "removed_volume_mm3": 0.0},
        )

        self.assertAlmostEqual(reference, 12.5 - 4.25e-6, places=14)
        self.assertEqual(unchanged, 12.5)

    def test_boolean_microface_exception_is_size_bounded(self) -> None:
        sphere = common.trimesh.creation.icosphere(subdivisions=3, radius=8.0)
        parts = [sphere]
        for index in range(6):
            origin = 20.0 + 0.4 * index
            parts.append(
                common.trimesh.Trimesh(
                    vertices=np.asarray(
                        [
                            [origin, 0.0, 0.0],
                            [origin + 0.1, 0.0, 0.0],
                            [origin + 0.2, 1e-15, 0.0],
                            [origin, 0.0, 0.1],
                        ],
                        dtype=np.float64,
                    ),
                    faces=np.asarray(
                        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
                        dtype=np.int64,
                    ),
                    process=False,
                )
            )
        mesh = common.trimesh.util.concatenate(parts)
        common.trimesh.repair.fix_winding(mesh)
        common.trimesh.repair.fix_normals(mesh)

        valid, record = local_connector_module._audit_boolean_candidate(
            mesh,
            reference_volume=abs(float(mesh.volume)),
            movement_tolerance_mm=0.0,
            maximum_new_collapsed_face_ratio=0.001,
        )

        self.assertTrue(valid)
        self.assertFalse(record["ratio_accepted_collapsed_faces"])
        self.assertTrue(record["micro_collapsed_faces_accepted"])
        self.assertLessEqual(record["collapsed_maximum_edge_mm"], 0.25)
        preserved = finalize_boolean_difference_mesh(
            mesh,
            policy=BooleanFinalizationPolicy(
                micro_accepted_collapsed_face_budget=6,
                maximum_micro_collapsed_face_ratio=0.005,
                maximum_micro_collapsed_face_edge_mm=0.25,
            ),
        )
        self.assertTrue(
            preserved.metadata["boolean_collapsed_face_audit"][
                "microfaces_accepted"
            ]
        )

    def test_default_boolean_ratio_accepts_closed_zero_volume_seams(self) -> None:
        sphere = common.trimesh.creation.icosphere(subdivisions=4, radius=8.0)
        parts = [sphere]
        collapsed_count = 20
        for index in range(collapsed_count):
            origin = 20.0 + 3.0 * index
            parts.append(
                common.trimesh.Trimesh(
                    vertices=np.asarray(
                        [
                            [origin, 0.0, 0.0],
                            [origin + 1.0, 0.0, 0.0],
                            [origin + 2.0, 1e-15, 0.0],
                            [origin, 0.0, 1.0],
                        ],
                        dtype=np.float64,
                    ),
                    faces=np.asarray(
                        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
                        dtype=np.int64,
                    ),
                    process=False,
                )
            )
        mesh = common.trimesh.util.concatenate(parts)
        common.trimesh.repair.fix_winding(mesh)
        common.trimesh.repair.fix_normals(mesh)

        valid, record = local_connector_module._audit_boolean_candidate(
            mesh,
            reference_volume=abs(float(mesh.volume)),
            movement_tolerance_mm=0.0,
        )

        self.assertTrue(valid)
        self.assertTrue(record["ratio_accepted_collapsed_faces"])
        self.assertFalse(record["micro_collapsed_faces_accepted"])
        self.assertEqual(record["new_collapsed_faces"], collapsed_count)
        self.assertGreater(record["collapsed_maximum_edge_mm"], 0.25)
        self.assertLess(record["new_collapsed_face_ratio"], 0.005)

        preserved = finalize_boolean_difference_mesh(
            mesh,
            policy=BooleanFinalizationPolicy(
                ratio_accepted_collapsed_face_budget=collapsed_count,
            ),
        )
        self.assertTrue(
            preserved.metadata["boolean_collapsed_face_audit"]["ratio_accepted"]
        )

    def test_boolean_finalizer_preserves_only_inherited_source_slivers(self) -> None:
        mesh = common.trimesh.Trimesh(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 1e-15, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray(
                [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
                dtype=np.int64,
            ),
            process=False,
        )
        common.trimesh.repair.fix_winding(mesh)
        common.trimesh.repair.fix_normals(mesh)
        inherited = boolean_collapsed_face_audit(mesh)

        self.assertTrue(mesh.is_watertight)
        self.assertGreater(inherited["collapsed_face_count"], 0)
        with self.assertRaisesRegex(ValueError, "inherited source budget"):
            finalize_boolean_difference_mesh(mesh)

        preserved = finalize_boolean_difference_mesh(
            mesh,
            policy=BooleanFinalizationPolicy(
                inherited_collapsed_face_budget=int(
                    inherited["collapsed_face_count"]
                ),
                inherited_source="unit_test_source",
            ),
        )
        audit = preserved.metadata["boolean_collapsed_face_audit"]
        self.assertTrue(preserved.is_watertight)
        self.assertEqual(audit["new_collapsed_face_count"], 0)
        self.assertEqual(audit["inherited_source"], "unit_test_source")

    def test_mesh64_array_contract_normalizes_views_and_signed_faces(self) -> None:
        vertices_source = np.arange(24, dtype=np.float32).reshape((4, 6))
        vertices_view = vertices_source[:, ::2]
        faces_view = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)[::-1]
        mesh = common.trimesh.Trimesh(
            vertices=vertices_view,
            faces=faces_view,
            process=False,
        )

        vertices, faces = local_connector_module._mesh64_arrays(mesh)

        self.assertEqual(vertices.shape, (4, 3))
        self.assertEqual(faces.shape, (2, 3))
        self.assertEqual(vertices.dtype, np.dtype(np.float64))
        self.assertEqual(faces.dtype, np.dtype(np.uint64))
        self.assertTrue(vertices.flags.c_contiguous)
        self.assertTrue(faces.flags.c_contiguous)
        # This is the exact call which previously failed only after the full
        # Lulu prebuild had already spent about fifteen minutes.
        manifold = local_connector_module._manifold64(mesh)
        self.assertIsNotNone(manifold.status())

    def test_nonplanar_cap_greedy_skips_repeated_zero_area_ear(self) -> None:
        # The first two positions intentionally coincide.  Old shortest-ear
        # selection emitted [last, first, second] with zero area, which made a
        # visually closed Lulu parent invalid before the mortise Boolean.
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.1],
                [2.0, 2.0, 0.0],
                [0.0, 2.0, -0.1],
            ],
            dtype=np.float64,
        )

        triangles, record = mesh_module.triangulate_nonplanar_loop_greedy(points)

        self.assertEqual(triangles, [])
        self.assertEqual(record["greedy_status"], "incomplete")

    def test_planar_transition_proof_rejects_a_curved_ring(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        outer = np.column_stack(
            (8.0 * np.cos(angles), 5.0 * np.sin(angles), np.zeros(len(angles)))
        )
        inner = np.column_stack(
            (
                7.5 * np.cos(angles),
                4.5 * np.sin(angles),
                np.full(len(angles), -0.25),
            )
        )
        curved = inner.copy()
        curved[:, 2] += 0.12 * np.sin(3.0 * angles)

        self.assertTrue(
            connector_topology_module._parallel_planar_ring_transition(
                outer,
                inner,
            )
        )
        self.assertFalse(
            connector_topology_module._parallel_planar_ring_transition(
                outer,
                curved,
            )
        )

    def test_long_bridge_local_plane_proof_rejects_a_fold(self) -> None:
        point_by_id = {
            0: np.asarray([0.0, 0.0, 0.0]),
            1: np.asarray([8.0, 0.0, 0.0]),
            2: np.asarray([0.0, 1.0, 0.0]),
            3: np.asarray([8.0, 1.0, 0.0]),
        }
        faces = [(0, 1, 2), (1, 3, 2)]
        long_edges = [(1, 2)]

        self.assertTrue(
            connector_topology_module._long_bridge_neighborhoods_are_coplanar(
                faces,
                point_by_id,
                long_edges,
            )
        )
        folded = dict(point_by_id)
        folded[3] = np.asarray([8.0, 1.0, 0.25])
        self.assertFalse(
            connector_topology_module._long_bridge_neighborhoods_are_coplanar(
                faces,
                folded,
                long_edges,
            )
        )

    def test_oblique_planar_cap_is_validated_against_its_fitted_plane(self) -> None:
        # Projection search is free to triangulate through a world-axis view,
        # but that viewing normal is not the physical normal of this tilted
        # cap.  Reusing it for the planarity audit caused false blade failures.
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 2.0],
                [4.0, 3.0, 2.0],
                [0.0, 3.0, 0.0],
            ],
            dtype=np.float64,
        )
        vertices = [point.copy() for point in points]
        faces: list[list[int]] = []
        wrong_projection_normal = np.asarray([0.0, 0.0, 1.0])

        with mock.patch.object(
            mesh_module,
            "triangulate_ordered_loop_3d",
            return_value=(
                [(0, 1, 2), (0, 2, 3)],
                wrong_projection_normal,
                {"status": "ready", "projection": "axis_z"},
            ),
        ):
            added, record = mesh_module.triangulate_boundary_cap_without_center(
                vertices,
                faces,
                [0, 1, 2, 3],
                np.asarray([-0.5, 0.0, 1.0]),
                require_planar_quality=True,
            )

        self.assertEqual(added, 2)
        self.assertTrue(record["surface_quality"]["valid"])
        self.assertLess(record["planarity_max_error_mm"], 1e-9)
        self.assertGreater(
            float(
                np.linalg.norm(
                    np.asarray(record["cap_plane_normal"])
                    - np.asarray(record["projection_normal"])
                )
            ),
            0.1,
        )

    def test_complete_male_negative_tool_has_only_female_side_boundary_overcut(self) -> None:
        boundary = np.asarray(
            [
                [-6.0, -5.0, 0.0],
                [6.0, -5.0, 0.0],
                [6.0, 5.0, 0.0],
                [-6.0, 5.0, 0.0],
            ],
            dtype=np.float64,
        )

        cutter = build_local_male_attachment_cutter(
            boundary_points=boundary,
            inward=np.asarray([0.0, 0.0, -1.0]),
            spec=self.spec,
            outside_extension_mm=2.0,
            boundary_overcut_mm=0.02,
        )
        source_min = boundary[:, :2].min(axis=0)
        source_max = boundary[:, :2].max(axis=0)
        cutter_boundary = np.asarray(cutter.vertices, dtype=np.float64)[4:8, :2]

        self.assertTrue(cutter.is_watertight)
        self.assertTrue(cutter.is_winding_consistent)
        self.assertTrue(np.all(cutter_boundary.min(axis=0) < source_min))
        self.assertTrue(np.all(cutter_boundary.max(axis=0) > source_max))
        self.assertAlmostEqual(cutter.metadata["boundary_overcut_mm"], 0.02, places=9)

    def test_socket_has_clearance_and_true_45_degree_mouth(self) -> None:
        profile = connector_profile(self.spec, contact_width_mm=14.0, contact_length_mm=14.0)

        self.assertAlmostEqual(profile["socket_throat_width_mm"], 4.50, places=6)
        self.assertAlmostEqual(profile["socket_throat_length_mm"], 4.50, places=6)
        self.assertAlmostEqual(profile["socket_depth_mm"], 5.25, places=6)
        self.assertAlmostEqual(profile["mouth_per_side_expansion_mm"], 0.80, places=6)
        self.assertAlmostEqual(profile["mouth_slope_degrees"], 45.0, places=6)

    def test_synthetic_pair_is_individually_watertight(self) -> None:
        pair = build_synthetic_local_connector_pair(self.spec)

        for name in ("child", "parent"):
            mesh = pair[name]
            self.assertTrue(mesh.is_watertight, name)
            self.assertTrue(mesh.is_winding_consistent, name)
            self.assertGreater(float(mesh.volume), 0.0, name)

    def test_peg_fits_socket_throat_without_using_chamfer_as_clearance(self) -> None:
        profile = connector_profile(self.spec, contact_width_mm=14.0, contact_length_mm=14.0)
        per_side = profile["total_clearance_mm"] / 2.0

        self.assertAlmostEqual(
            (profile["socket_throat_width_mm"] - profile["peg_width_mm"]) / 2.0,
            per_side,
            places=6,
        )
        self.assertAlmostEqual(
            (profile["socket_throat_length_mm"] - profile["peg_length_mm"]) / 2.0,
            per_side,
            places=6,
        )

    def test_chamfer_is_short_and_engagement_remains_near_five_mm(self) -> None:
        profile = connector_profile(self.spec, contact_width_mm=14.0, contact_length_mm=14.0)

        straight_depth = profile["socket_depth_mm"] - profile["socket_mouth_chamfer_mm"]
        self.assertGreater(straight_depth, 4.0)
        self.assertLess(profile["socket_mouth_chamfer_mm"], 1.0)
        self.assertTrue(math.isclose(profile["engagement_depth_mm"], 5.0))

    def test_curved_source_loop_gets_one_centered_local_connector(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        boundary = np.column_stack(
            (
                7.0 * np.cos(angles),
                6.5 * np.sin(angles),
                0.20 * np.sin(3.0 * angles),
            )
        )
        plan = plan_local_connector(boundary, np.asarray([0.0, 0.0, -1.0]), self.spec)

        self.assertLess(plan["footprint_area_ratio"], 0.15)
        self.assertAlmostEqual(plan["engagement_depth_mm"], 5.0, places=6)
        self.assertAlmostEqual(plan["socket_depth_mm"], 5.25, places=6)
        self.assertAlmostEqual(plan["mouth_slope_degrees"], 45.0, places=6)
        self.assertEqual(np.asarray(plan["peg_top"]).shape, (32, 3))
        self.assertEqual(np.asarray(plan["socket_mouth"]).shape, (32, 3))

    def test_small_interface_shortens_mouth_without_changing_45_degree_slope(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        boundary = np.column_stack(
            (
                3.0 * np.cos(angles),
                1.6 * np.sin(angles),
                np.zeros_like(angles),
            )
        )
        plan = plan_local_connector(
            boundary,
            np.asarray([0.0, 0.0, -1.0]),
            self.spec,
        )

        self.assertTrue(plan["adaptive_socket_mouth_chamfer_applied"])
        self.assertGreater(plan["socket_mouth_chamfer_mm"], 0.05 - 1e-6)
        self.assertLess(plan["socket_mouth_chamfer_mm"], 0.80)
        self.assertAlmostEqual(plan["mouth_slope_degrees"], 45.0, places=6)

    def test_full_depth_backing_rejects_vertical_skirt_fallback(self) -> None:
        width = 5.5
        boundary = np.asarray(
            [
                [-width / 2.0, -width / 2.0, 0.0],
                [width / 2.0, -width / 2.0, 0.0],
                [width / 2.0, width / 2.0, 0.0],
                [-width / 2.0, width / 2.0, 0.0],
            ]
        )
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=5.0,
        )

        with self.assertRaisesRegex(ValueError, "continuous full-depth 45-degree"):
            plan_local_connector(
                boundary,
                np.asarray([0.0, 0.0, -1.0]),
                spec,
            )

    def test_nonplanar_backing_records_measured_taper_angle_range(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        boundary = np.column_stack(
            (
                9.0 * np.cos(angles),
                7.5 * np.sin(angles),
                0.825 * np.sin(3.0 * angles),
            )
        )
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=5.0,
        )
        vertices = [point.copy() for point in boundary]
        faces: list[list[int]] = []

        record = add_local_male_connector_and_backing(
            output_vertices=vertices,
            output_faces=faces,
            boundary_ids=list(range(len(boundary))),
            boundary_points=boundary,
            inward=np.asarray([0.0, 0.0, -1.0]),
            spec=spec,
        )

        self.assertLess(record["backing_taper_measured_minimum_degrees"], 45.0)
        self.assertAlmostEqual(
            record["backing_taper_measured_median_degrees"],
            45.0,
            delta=2.0,
        )
        self.assertGreater(record["backing_taper_measured_maximum_degrees"], 45.0)
        self.assertEqual(record["backing_profile_layer_count"], 1)

    def test_final_refined_backing_ring_is_used_by_floor_annulus(self) -> None:
        boundary = np.asarray(
            [[-8.0, -8.0, 0.0], [8.0, -8.0, 0.0], [8.0, 8.0, 0.0], [-8.0, 8.0, 0.0]]
        )
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=5.0,
        )
        original_strip = inward_module.triangulate_bounded_ring_strip
        original_annulus = inward_module._add_constrained_connector_annulus
        strip_calls = 0

        def refine_last_strip(*args, **kwargs):
            nonlocal strip_calls
            strip_calls += 1
            result = original_strip(*args, **kwargs)
            if strip_calls != 1:
                return result
            inner_ids = list(result.inner_ids)
            inner_points = np.asarray(result.inner_points, dtype=np.float64)
            synthetic_id = max(max(args[0]), max(inner_ids)) + 1
            refined_ids = [inner_ids[0], synthetic_id, *inner_ids[1:]]
            refined_points = np.vstack(
                (
                    inner_points[0],
                    0.5 * (inner_points[0] + inner_points[1]),
                    inner_points[1:],
                )
            )
            face_map = []
            split_edge = tuple(sorted((inner_ids[0], inner_ids[1])))
            for face in result.faces:
                edges = (
                    (face[0], face[1]),
                    (face[1], face[2]),
                    (face[2], face[0]),
                )
                if split_edge not in {tuple(sorted(edge)) for edge in edges}:
                    face_map.append(face)
                    continue
                third = next(value for value in face if value not in split_edge)
                face_map.extend(
                    (
                        (inner_ids[0], synthetic_id, third),
                        (synthetic_id, inner_ids[1], third),
                    )
                )
            return type(result)(
                tuple(face_map),
                result.audit,
                tuple(refined_ids),
                refined_points,
            )

        captured: dict[str, int] = {}

        def capture_annulus(**kwargs):
            captured["id_count"] = len(kwargs["outer_ids"])
            captured["point_count"] = len(kwargs["outer_points"])
            return original_annulus(**kwargs)

        vertices = [point.copy() for point in boundary]
        faces: list[list[int]] = []
        with mock.patch.object(
            inward_module,
            "triangulate_bounded_ring_strip",
            side_effect=refine_last_strip,
        ), mock.patch.object(
            inward_module,
            "_add_constrained_connector_annulus",
            side_effect=capture_annulus,
        ):
            add_local_male_connector_and_backing(
                output_vertices=vertices,
                output_faces=faces,
                boundary_ids=list(range(len(boundary))),
                boundary_points=boundary,
                inward=np.asarray([0.0, 0.0, -1.0]),
                spec=spec,
            )

        self.assertEqual(captured["id_count"], 5)
        self.assertEqual(captured["point_count"], 5)

    def test_manifold_difference_opens_matching_female_socket(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        boundary = np.column_stack(
            (7.0 * np.cos(angles), 7.0 * np.sin(angles), np.zeros_like(angles))
        )
        plan = plan_local_connector(
            boundary,
            np.asarray([0.0, 0.0, -1.0]),
            self.spec,
            samples=64,
        )
        cutter = build_socket_cutter_from_plan(plan, outside_extension_mm=2.0)
        parent = common.trimesh.creation.box(extents=[20.0, 20.0, 8.0])
        parent.apply_translation([0.0, 0.0, -4.0])
        socketed, record = subtract_socket_cutters(parent, [cutter])

        self.assertTrue(cutter.is_watertight)
        self.assertTrue(socketed.is_watertight)
        self.assertTrue(socketed.is_winding_consistent)
        self.assertGreater(record["intersection_volume_mm3"], 1.0)
        self.assertEqual(
            record["cutter_steps"][0]["boolean_coordinate_precision"],
            "float64_mesh64",
        )
        self.assertLess(
            record["parent_volume_after_mm3"],
            record["parent_volume_before_mm3"],
        )

    def test_sequential_boolean_keeps_parent_manifold_resident(self) -> None:
        parent = common.trimesh.creation.box(extents=[20.0, 20.0, 8.0])
        parent.apply_translation([0.0, 0.0, -4.0])
        cutters = []
        for x_position in (-4.0, 4.0):
            cutter = common.trimesh.creation.box(extents=[3.0, 3.0, 4.0])
            cutter.apply_translation([x_position, 0.0, -1.0])
            cutters.append(cutter)

        with mock.patch.object(
            local_connector_module,
            "_manifold64",
            wraps=local_connector_module._manifold64,
        ) as manifold_import:
            result, record = subtract_socket_cutters(parent, cutters)

        self.assertTrue(result.is_watertight)
        self.assertTrue(result.is_winding_consistent)
        self.assertEqual(record["applied_socket_cutter_count"], 2)
        self.assertEqual(manifold_import.call_count, 3)

    def test_dense_aligned_ring_uses_one_strict_topology_audit(self) -> None:
        samples = 512
        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        outer_projected = np.column_stack(
            (12.0 * np.cos(angles), 8.0 * np.sin(angles))
        )
        inner_projected = np.column_stack(
            (11.0 * np.cos(angles), 7.0 * np.sin(angles))
        )
        outer_points = np.column_stack(
            (outer_projected, 0.05 * np.sin(3.0 * angles))
        )
        inner_points = np.column_stack(
            (inner_projected, -0.25 + 0.05 * np.sin(3.0 * angles))
        )
        outer_ids = list(range(samples))
        inner_ids = list(range(samples, 2 * samples))

        with mock.patch.object(
            connector_topology_module,
            "_audit_ring_strip",
            wraps=connector_topology_module._audit_ring_strip,
        ) as strict_audit:
            result = connector_topology_module.triangulate_bounded_ring_strip(
                outer_ids,
                outer_points,
                inner_ids,
                inner_points,
                outer_projected,
                inner_projected,
            )

        self.assertTrue(result.audit.valid)
        self.assertEqual(strict_audit.call_count, 1)
        self.assertEqual(
            result.audit.strategy,
            "aligned_equal_count_strict_audit",
        )

    def test_dense_mismatched_strip_uses_shared_parameter_grid(self) -> None:
        outer_count = 400
        inner_count = 101
        outer_angles = np.linspace(0.0, 2.0 * np.pi, outer_count, endpoint=False)
        inner_angles = np.linspace(0.0, 2.0 * np.pi, inner_count, endpoint=False)
        outer_projected = np.column_stack(
            (10.0 * np.cos(outer_angles), 8.0 * np.sin(outer_angles))
        )
        inner_projected = np.column_stack(
            (9.0 * np.cos(inner_angles), 7.0 * np.sin(inner_angles))
        )
        outer_points = np.column_stack((outer_projected, np.zeros(outer_count)))
        inner_points = np.column_stack((inner_projected, -np.ones(inner_count)))
        outer_ids = list(range(outer_count))
        inner_ids = list(range(outer_count, outer_count + inner_count))
        seam_roll = 37
        inner_projected = np.roll(inner_projected, seam_roll, axis=0)
        inner_points = np.roll(inner_points, seam_roll, axis=0)
        inner_ids = list(np.roll(np.asarray(inner_ids), seam_roll))

        with mock.patch.object(
            connector_topology_module,
            "triangulate_connector_annulus",
            side_effect=AssertionError("ranked strip unexpectedly used annulus fallback"),
        ):
            result = connector_topology_module.triangulate_bounded_ring_strip(
                outer_ids,
                outer_points,
                inner_ids,
                inner_points,
                outer_projected,
                inner_projected,
                maximum_fanout=4,
            )

        self.assertTrue(result.audit.valid)
        # Dense mismatches only need enough local refinement to satisfy the
        # bounded-fanout contract.  Expanding the inner ring to the full outer
        # count creates hundreds of unnecessary skinny triangles and was the
        # source of visible pits on otherwise smooth connector surfaces.
        self.assertGreaterEqual(len(result.inner_ids), inner_count)
        self.assertLess(len(result.inner_ids), outer_count)
        self.assertTrue(set(inner_ids).issubset(set(result.inner_ids)))
        self.assertLessEqual(result.audit.maximum_fanout, 8)
        self.assertEqual(
            result.audit.strategy,
            "balanced_arc_length_ranked_seam_bounded_fanout",
        )

    def test_extreme_ring_ratio_tries_one_audited_preconditioned_seam(self) -> None:
        outer_count = 256
        inner_count = 24
        outer_angles = np.linspace(0.0, 2.0 * np.pi, outer_count, endpoint=False)
        inner_angles = np.linspace(0.0, 2.0 * np.pi, inner_count, endpoint=False)
        outer_projected = np.column_stack(
            (10.0 * np.cos(outer_angles), 8.0 * np.sin(outer_angles))
        )
        inner_projected = np.column_stack(
            (7.0 * np.cos(inner_angles), 5.0 * np.sin(inner_angles))
        )
        outer_points = np.column_stack((outer_projected, np.zeros(outer_count)))
        inner_points = np.column_stack((inner_projected, -np.ones(inner_count)))
        outer_ids = list(range(outer_count))
        inner_ids = list(range(outer_count, outer_count + inner_count))

        original_balanced_strip = connector_topology_module._balanced_ring_strip_faces
        with mock.patch.object(
            connector_topology_module,
            "_balanced_ring_strip_faces",
            wraps=original_balanced_strip,
        ) as balanced_strip:
            result = connector_topology_module.triangulate_bounded_ring_strip(
                outer_ids,
                outer_points,
                inner_ids,
                inner_points,
                outer_projected,
                inner_projected,
                maximum_fanout=4,
            )

        self.assertTrue(result.audit.valid)
        self.assertEqual(balanced_strip.call_count, 1)
        self.assertTrue(set(inner_ids).issubset(set(result.inner_ids)))
        self.assertLess(len(result.inner_ids), len(outer_ids))
        self.assertEqual(
            result.audit.strategy,
            "balanced_arc_length_ranked_seam_bounded_fanout",
        )

    def test_five_vertex_inset_can_join_dense_vendor_ring(self) -> None:
        outer_count = 578
        outer_angles = np.linspace(0.0, 2.0 * np.pi, outer_count, endpoint=False)
        inner_angles = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
        outer_projected = np.column_stack(
            (10.0 * np.cos(outer_angles), 8.0 * np.sin(outer_angles))
        )
        inner_projected = np.column_stack(
            (7.0 * np.cos(inner_angles), 5.0 * np.sin(inner_angles))
        )
        outer_points = np.column_stack((outer_projected, np.zeros(outer_count)))
        inner_points = np.column_stack((inner_projected, -np.ones(5)))
        outer_ids = list(range(outer_count))
        inner_ids = list(range(outer_count, outer_count + 5))

        with mock.patch.object(
            connector_topology_module,
            "triangulate_connector_annulus",
            side_effect=AssertionError("accepted audited seam unexpectedly fell back"),
        ):
            result = connector_topology_module.triangulate_bounded_ring_strip(
                outer_ids,
                outer_points,
                inner_ids,
                inner_points,
                outer_projected,
                inner_projected,
                maximum_fanout=4,
            )

        self.assertTrue(result.audit.valid)
        self.assertEqual(len(result.inner_ids), 145)
        self.assertTrue(set(inner_ids).issubset(set(result.inner_ids)))
        self.assertEqual(
            result.audit.strategy,
            "balanced_arc_length_ranked_seam_bounded_fanout",
        )

    def test_preconditioned_extreme_ring_uses_visibility_path_fallback(self) -> None:
        outer_count = 256
        inner_count = 24
        outer_angles = np.linspace(0.0, 2.0 * np.pi, outer_count, endpoint=False)
        inner_angles = np.linspace(0.0, 2.0 * np.pi, inner_count, endpoint=False)
        outer_projected = np.column_stack(
            (10.0 * np.cos(outer_angles), 8.0 * np.sin(outer_angles))
        )
        inner_projected = np.column_stack(
            (7.0 * np.cos(inner_angles), 5.0 * np.sin(inner_angles))
        )
        outer_points = np.column_stack((outer_projected, np.zeros(outer_count)))
        inner_points = np.column_stack((inner_projected, -np.ones(inner_count)))

        with (
            mock.patch.object(
                connector_topology_module,
                "_balanced_ring_strip_faces",
                return_value=[],
            ),
            mock.patch.object(
                connector_topology_module,
                "triangulate_connector_annulus",
                side_effect=AssertionError("visibility path unexpectedly fell back"),
            ),
        ):
            result = connector_topology_module.triangulate_bounded_ring_strip(
                list(range(outer_count)),
                outer_points,
                list(range(outer_count, outer_count + inner_count)),
                inner_points,
                outer_projected,
                inner_projected,
                maximum_fanout=4,
            )

        self.assertTrue(result.audit.valid)
        self.assertEqual(
            result.audit.strategy,
            "visibility_dynamic_program_bounded_fanout",
        )

    def test_user_reviewed_projection_fallback_keeps_hard_topology_gates(self) -> None:
        outer_count = 256
        inner_count = 24
        outer_angles = np.linspace(0.0, 2.0 * np.pi, outer_count, endpoint=False)
        inner_angles = np.linspace(0.0, 2.0 * np.pi, inner_count, endpoint=False)
        outer_projected = np.column_stack(
            (10.0 * np.cos(outer_angles), 8.0 * np.sin(outer_angles))
        )
        # Deliberately move the projected inset across the outer outline.  It
        # models a valid spatial rim whose selected 2-D assembly projection
        # folds, so projection containment must not masquerade as 3-D damage.
        inner_projected = np.column_stack(
            (7.0 * np.cos(inner_angles) + 7.5, 5.0 * np.sin(inner_angles))
        )
        outer_points = np.column_stack((outer_projected, np.zeros(outer_count)))
        inner_points = np.column_stack((inner_projected, -np.ones(inner_count)))
        invalid_patch = connector_topology_module.PatchTopologyAudit(
            False, 0, 0, 0, 0, 0, 0.0, 0.0, "synthetic_failure", "test"
        )

        with (
            mock.patch.object(
                connector_topology_module,
                "_balanced_ring_strip_faces",
                return_value=[],
            ),
            mock.patch.object(
                connector_topology_module,
                "_bridge_visibility_matrix",
                return_value=np.zeros((outer_count, 64), dtype=bool),
            ),
            mock.patch.object(
                connector_topology_module,
                "triangulate_connector_annulus",
                return_value=([], invalid_patch),
            ),
        ):
            result = connector_topology_module.triangulate_bounded_ring_strip(
                list(range(outer_count)),
                outer_points,
                list(range(outer_count, outer_count + inner_count)),
                inner_points,
                outer_projected,
                inner_projected,
                maximum_fanout=4,
                allow_projection_bridge_passthrough=True,
            )

        self.assertTrue(result.audit.valid)
        self.assertEqual(result.audit.degenerate_face_count, 0)
        self.assertEqual(result.audit.over_shared_edge_count, 0)
        self.assertLessEqual(result.audit.maximum_fanout, 8)
        self.assertGreater(result.audit.invalid_bridge_count, 0)
        self.assertTrue(result.audit.projection_bridge_passthrough_accepted)
        self.assertEqual(
            result.audit.strategy,
            "bounded_parameter_path_user_reviewed_projection_advisory",
        )

    def test_user_reviewed_hidden_broad_fanout_has_finite_bounds(self) -> None:
        policy = connector_topology_module._user_reviewed_broad_fanout_is_bounded

        self.assertTrue(
            policy(
                enabled=True,
                maximum_fanout=37,
                maximum_cross_edge_mm=5.035085,
                invalid_bridge_count=0,
            )
        )
        self.assertFalse(
            policy(
                enabled=True,
                maximum_fanout=65,
                maximum_cross_edge_mm=5.0,
                invalid_bridge_count=0,
            )
        )
        self.assertFalse(
            policy(
                enabled=True,
                maximum_fanout=37,
                maximum_cross_edge_mm=8.1,
                invalid_bridge_count=0,
            )
        )
        self.assertFalse(
            policy(
                enabled=True,
                maximum_fanout=37,
                maximum_cross_edge_mm=5.0,
                invalid_bridge_count=1,
            )
        )

    def test_user_reviewed_parameter_path_does_not_require_preconditioning(self) -> None:
        outer_count = 74
        inner_count = 50
        outer_angles = np.linspace(0.0, 2.0 * np.pi, outer_count, endpoint=False)
        inner_angles = np.linspace(0.0, 2.0 * np.pi, inner_count, endpoint=False)
        outer_projected = np.column_stack(
            (5.0 * np.cos(outer_angles), 4.0 * np.sin(outer_angles))
        )
        inner_projected = np.column_stack(
            (3.5 * np.cos(inner_angles), 2.5 * np.sin(inner_angles))
        )
        outer_points = np.column_stack((outer_projected, np.zeros(outer_count)))
        inner_points = np.column_stack((inner_projected, -np.ones(inner_count)))
        invalid_patch = connector_topology_module.PatchTopologyAudit(
            False, 0, 0, 0, 0, 0, 0.0, 0.0, "synthetic_failure", "test"
        )

        with (
            mock.patch.object(
                connector_topology_module,
                "_balanced_ring_strip_faces",
                return_value=[],
            ),
            mock.patch.object(
                connector_topology_module,
                "triangulate_connector_annulus",
                return_value=([], invalid_patch),
            ),
            mock.patch.object(
                connector_topology_module,
                "_bridge_visibility_matrix",
                return_value=np.zeros((outer_count, inner_count), dtype=bool),
            ),
        ):
            result = connector_topology_module.triangulate_bounded_ring_strip(
                list(range(outer_count)),
                outer_points,
                list(range(outer_count, outer_count + inner_count)),
                inner_points,
                outer_projected,
                inner_projected,
                maximum_fanout=4,
                allow_projection_bridge_passthrough=True,
            )

        self.assertTrue(result.audit.valid)
        self.assertEqual(
            result.audit.strategy,
            "bounded_parameter_path_user_reviewed_projection_advisory",
        )

    def test_large_extreme_ring_ratio_tries_one_audited_linear_seam(self) -> None:
        outer_count = 1050
        inner_count = 100
        outer_angles = np.linspace(0.0, 2.0 * np.pi, outer_count, endpoint=False)
        inner_angles = np.linspace(0.0, 2.0 * np.pi, inner_count, endpoint=False)
        outer_projected = np.column_stack(
            (10.0 * np.cos(outer_angles), 8.0 * np.sin(outer_angles))
        )
        inner_projected = np.column_stack(
            (7.0 * np.cos(inner_angles), 5.0 * np.sin(inner_angles))
        )
        outer_points = np.column_stack((outer_projected, np.zeros(outer_count)))
        inner_points = np.column_stack((inner_projected, -np.ones(inner_count)))
        outer_ids = list(range(outer_count))
        inner_ids = list(range(outer_count, outer_count + inner_count))

        with mock.patch.object(
            connector_topology_module,
            "triangulate_connector_annulus",
            side_effect=AssertionError("accepted linear probe unexpectedly fell back"),
        ):
            result = connector_topology_module.triangulate_bounded_ring_strip(
                outer_ids,
                outer_points,
                inner_ids,
                inner_points,
                outer_projected,
                inner_projected,
                maximum_fanout=4,
            )

        self.assertTrue(result.audit.valid)
        self.assertEqual(
            result.audit.strategy,
            "balanced_arc_length_ranked_seam_bounded_fanout",
        )
        self.assertLessEqual(result.audit.maximum_fanout, 8)

    def test_dense_star_annulus_returns_before_constrained_solver_import(self) -> None:
        outer_count = 600
        hole_count = 500
        outer_angles = np.linspace(0.0, 2.0 * np.pi, outer_count, endpoint=False)
        hole_angles = np.linspace(0.0, 2.0 * np.pi, hole_count, endpoint=False)
        outer = np.column_stack(
            (10.0 * np.cos(outer_angles), 8.0 * np.sin(outer_angles))
        )
        hole = np.column_stack(
            (7.0 * np.cos(hole_angles), 5.0 * np.sin(hole_angles))
        )
        outer_ids = list(range(outer_count))
        hole_ids = list(range(outer_count, outer_count + hole_count))

        with mock.patch.dict(sys.modules, {"manifold3d": None}):
            faces, audit = connector_topology_module.triangulate_connector_annulus(
                outer_ids,
                outer,
                hole_ids,
                hole,
            )

        self.assertTrue(audit.valid)
        self.assertEqual(audit.strategy, "polar_monotone_zipper")
        self.assertEqual(len(faces), outer_count + hole_count)

    def test_dense_failed_solver_never_enters_visible_bridge_search(self) -> None:
        count = 130
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        outer = np.column_stack((10.0 * np.cos(angles), 8.0 * np.sin(angles)))
        hole = np.column_stack((7.0 * np.cos(angles), 5.0 * np.sin(angles)))
        outer_ids = list(range(count))
        hole_ids = list(range(count, 2 * count))
        invalid = connector_topology_module.PatchTopologyAudit(
            False,
            0,
            0,
            0,
            0,
            0,
            0.0,
            0.0,
            "synthetic_failure",
            "polar_monotone_zipper",
        )
        fake_manifold = mock.Mock()
        fake_manifold.triangulate.side_effect = ValueError("synthetic failure")

        with (
            mock.patch.dict(sys.modules, {"manifold3d": fake_manifold}),
            mock.patch.object(
                connector_topology_module,
                "triangulate_star_shaped_annulus",
                return_value=([], invalid),
            ),
            mock.patch.object(
                connector_topology_module,
                "triangulate_annulus_with_visible_bridges",
                side_effect=AssertionError("dense quadratic fallback was entered"),
            ),
        ):
            faces, audit = connector_topology_module.triangulate_connector_annulus(
                outer_ids,
                outer,
                hole_ids,
                hole,
            )

        self.assertEqual(faces, [])
        self.assertFalse(audit.valid)
        self.assertEqual(
            audit.reason,
            "constrained_solver_failed_for_dense_concave_annulus",
        )

    def test_dense_audited_hidden_surface_bypasses_optional_refinement(self) -> None:
        vertices = [
            np.asarray(point, dtype=np.float64)
            for point in (
                (-2.0, -2.0, 0.0),
                (2.0, -2.0, 0.0),
                (2.0, 2.0, 0.0),
                (-2.0, 2.0, 0.0),
                (-1.0, -1.0, -1.0),
                (1.0, -1.0, -1.0),
                (1.0, 1.0, -1.0),
                (-1.0, 1.0, -1.0),
            )
        ]
        faces = [
            [0, 1, 4],
            [1, 5, 4],
            [1, 2, 5],
            [2, 6, 5],
            [2, 3, 6],
            [3, 7, 6],
            [3, 0, 7],
            [0, 4, 7],
        ]
        original_vertices = [point.copy() for point in vertices]
        original_faces = [face.copy() for face in faces]
        plan = {
            "center": np.zeros(3),
            "u": np.asarray([1.0, 0.0, 0.0]),
            "v": np.asarray([0.0, 1.0, 0.0]),
            "inward": np.asarray([0.0, 0.0, -1.0]),
        }

        with mock.patch.object(
            connector_surface_module,
            "DENSE_AUDITED_PASSTHROUGH_FACE_LIMIT",
            1,
        ):
            refinement = connector_surface_module.refine_connector_annulus_heightfield(
                output_vertices=vertices,
                output_faces=faces,
                face_start=0,
                outer_ids=[0, 1, 2, 3],
                inner_ids=[4, 5, 6, 7],
                boundary_points=np.asarray(original_vertices[:4]),
                plan=plan,
                taper_depth_mm=1.0,
            )

        self.assertEqual(refinement.passes, 0)
        self.assertEqual(
            refinement.skipped_reason,
            "dense_hidden_annulus_already_audited",
        )
        self.assertEqual(faces, original_faces)
        np.testing.assert_allclose(np.asarray(vertices), np.asarray(original_vertices))

    def test_user_reviewed_hidden_surface_keeps_audited_strip_without_refinement(self) -> None:
        vertices = [
            np.asarray(point, dtype=np.float64)
            for point in (
                (-3.0, -3.0, 0.0),
                (3.0, -3.0, 0.0),
                (3.0, 3.0, 0.0),
                (-3.0, 3.0, 0.0),
                (-1.0, -1.0, -1.0),
                (1.0, -1.0, -1.0),
                (1.0, 1.0, -1.0),
                (-1.0, 1.0, -1.0),
            )
        ]
        faces = [
            [0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5],
            [2, 3, 6], [3, 7, 6], [3, 0, 7], [0, 4, 7],
        ]
        original_vertices = [point.copy() for point in vertices]
        original_faces = [face.copy() for face in faces]
        plan = {
            "center": np.zeros(3),
            "u": np.asarray([1.0, 0.0, 0.0]),
            "v": np.asarray([0.0, 1.0, 0.0]),
            "inward": np.asarray([0.0, 0.0, -1.0]),
        }

        refinement = connector_surface_module.refine_connector_annulus_heightfield(
            output_vertices=vertices,
            output_faces=faces,
            face_start=0,
            outer_ids=[0, 1, 2, 3],
            inner_ids=[4, 5, 6, 7],
            boundary_points=np.asarray(original_vertices[:4]),
            plan=plan,
            taper_depth_mm=1.0,
            allow_audited_resolution_passthrough=True,
        )

        self.assertEqual(refinement.passes, 0)
        self.assertEqual(
            refinement.skipped_reason,
            "user_reviewed_hidden_surface_resolution_advisory",
        )
        self.assertGreater(
            refinement.maximum_internal_edge_after_mm,
            refinement.target_maximum_internal_edge_mm,
        )
        self.assertEqual(faces, original_faces)
        np.testing.assert_allclose(np.asarray(vertices), np.asarray(original_vertices))

    def test_collapsed_boolean_cleanup_removes_only_seam_sliver(self) -> None:
        box = common.trimesh.creation.box(extents=[4.0, 4.0, 4.0])
        vertices = np.vstack(
            (
                np.asarray(box.vertices, dtype=np.float64),
                np.asarray(
                    [box.vertices[0], box.vertices[0], box.vertices[1]],
                    dtype=np.float64,
                ),
            )
        )
        faces = np.vstack(
            (
                np.asarray(box.faces, dtype=np.int64),
                np.asarray([[len(box.vertices), len(box.vertices) + 1, len(box.vertices) + 2]]),
            )
        )
        with_sliver = common.trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )

        repaired, record = local_connector_module._repair_collapsed_boolean_faces(
            with_sliver,
            minimum_double_area=1e-12,
            expected_degenerate_faces=1,
        )

        self.assertTrue(repaired.is_watertight)
        self.assertTrue(repaired.is_winding_consistent)
        self.assertTrue(record["applied"])
        self.assertEqual(record["policy"], "submicron_weld_without_hole_filling")
        self.assertAlmostEqual(abs(float(repaired.volume)), abs(float(box.volume)), places=9)

    def test_collapsed_boolean_cleanup_retriangulates_collinear_vertex_star(self) -> None:
        mesh = common.trimesh.creation.icosphere(subdivisions=2, radius=8.0)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        target_face = faces[0]
        removable_vertex = int(target_face[0])
        left = np.asarray(mesh.vertices[int(target_face[1])], dtype=np.float64)
        right = np.asarray(mesh.vertices[int(target_face[2])], dtype=np.float64)
        mesh.vertices[removable_vertex] = 0.5 * (left + right)
        mesh = common.trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float64),
            faces=faces,
            process=False,
        )
        triangles = np.asarray(mesh.vertices, dtype=np.float64)[faces]
        double_areas = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        threshold = 1e-12
        collapsed_before = int(np.count_nonzero(double_areas <= threshold))

        self.assertTrue(mesh.is_watertight)
        self.assertGreater(collapsed_before, 0)
        repaired, record = local_connector_module._repair_collapsed_boolean_faces(
            mesh,
            minimum_double_area=threshold,
            expected_degenerate_faces=collapsed_before,
        )

        repaired_triangles = np.asarray(repaired.vertices, dtype=np.float64)[
            np.asarray(repaired.faces, dtype=np.int64)
        ]
        repaired_double_areas = np.linalg.norm(
            np.cross(
                repaired_triangles[:, 1] - repaired_triangles[:, 0],
                repaired_triangles[:, 2] - repaired_triangles[:, 0],
            ),
            axis=1,
        )
        self.assertTrue(repaired.is_watertight)
        self.assertTrue(repaired.is_winding_consistent)
        self.assertEqual(int(np.count_nonzero(repaired_double_areas <= threshold)), 0)
        self.assertEqual(
            record["policy"],
            "local_collapsed_vertex_star_retriangulation_without_center_fan",
        )

    def test_boolean_export_uses_smallest_valid_manifold_simplification(self) -> None:
        clean = common.trimesh.creation.box(extents=[4.0, 4.0, 4.0])
        sliver = clean.copy()
        vertices = np.vstack((sliver.vertices, sliver.vertices[0], sliver.vertices[0]))
        faces = np.vstack((sliver.faces, [[len(vertices) - 2, len(vertices) - 1, 1]]))
        sliver = common.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        class FakeManifold:
            def __init__(self, tolerance: float = 0.0) -> None:
                self.tolerance = tolerance

            def volume(self) -> float:
                return 64.0

            def as_original(self):
                return self

            def simplify(self, tolerance: float):
                return FakeManifold(tolerance)

        def fake_export(manifold):
            return sliver.copy() if manifold.tolerance < 1e-8 else clean.copy()

        with mock.patch.object(
            local_connector_module,
            "_trimesh_from_manifold64",
            side_effect=fake_export,
        ):
            repaired, record = local_connector_module._validated_boolean_manifold64(
                FakeManifold()
            )

        self.assertTrue(repaired.is_watertight)
        self.assertEqual(
            record["policy"],
            "manifold_topology_simplify_without_hole_filling",
        )
        self.assertAlmostEqual(record["selected_tolerance_mm"], 1e-8, places=15)
        self.assertTrue(record["applied"])
        self.assertTrue(record["boolean_provenance_reset"])

    def test_boolean_export_removes_closed_zero_area_seam_face(self) -> None:
        clean = common.trimesh.creation.box(extents=[4.0, 4.0, 4.0])
        vertices = np.vstack((clean.vertices, clean.vertices[0], clean.vertices[0]))
        faces = np.vstack(
            (clean.faces, [[len(vertices) - 2, len(vertices) - 1, 1]])
        )
        closed_with_ghost = common.trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )

        class FakeManifold:
            def volume(self) -> float:
                return 64.0

            def as_original(self):
                return self

            def simplify(self, _tolerance: float):
                return self

        failed_audit = {
            "faces": int(len(closed_with_ghost.faces)),
            "watertight": True,
            "winding_consistent": True,
            "boundary_edges": 0,
            "over_shared_edges": 0,
            "degenerate_faces": 1,
            "inherited_collapsed_face_budget": 0,
            "new_collapsed_faces": 1,
            "minimum_double_area_mm2": 0.0,
            "volume_delta_mm3": 0.0,
            "volume_tolerance_mm3": 1e-6,
        }
        repaired_audit = {
            **failed_audit,
            "faces": int(len(clean.faces)),
            "degenerate_faces": 0,
            "new_collapsed_faces": 0,
            "minimum_double_area_mm2": 8.0,
        }
        with mock.patch.object(
            local_connector_module,
            "_trimesh_from_manifold64",
            return_value=closed_with_ghost.copy(),
        ), mock.patch.object(
            local_connector_module,
            "_audit_boolean_candidate",
            side_effect=[(False, failed_audit), (True, repaired_audit)],
        ):
            repaired, record, _kernel = (
                local_connector_module._validated_boolean_manifold64_with_kernel(
                    FakeManifold()
                )
            )

        self.assertTrue(repaired.is_watertight)
        self.assertTrue(repaired.is_winding_consistent)
        self.assertEqual(
            record["policy"],
            "manifold_topology_simplify_then_collapsed_seam_face_removal_without_hole_filling",
        )
        self.assertEqual(len(repaired.faces), len(clean.faces))

    def test_boolean_audit_rejects_closed_mesh_with_collapsed_face(self) -> None:
        box = common.trimesh.creation.box(extents=[4.0, 4.0, 4.0])
        vertices = np.vstack((box.vertices, box.vertices[0], box.vertices[0]))
        faces = np.vstack((box.faces, [[len(vertices) - 2, len(vertices) - 1, 1]]))
        sliver = common.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        valid, record = local_connector_module._audit_boolean_candidate(
            sliver,
            reference_volume=abs(float(box.volume)),
            movement_tolerance_mm=0.0,
        )

        self.assertFalse(valid)
        self.assertGreater(record["degenerate_faces"], 0)

    def test_final_part_clearance_cutter_is_uniform_minkowski_dilation(self) -> None:
        part = common.trimesh.creation.box(extents=[4.0, 6.0, 8.0])

        cutter, record = build_clearance_cutter_from_final_part(
            part,
            radial_clearance_mm=0.05,
        )

        self.assertTrue(cutter.is_watertight)
        self.assertTrue(cutter.is_winding_consistent)
        np.testing.assert_allclose(
            cutter.bounds[0], part.bounds[0] - 0.05, atol=1e-9
        )
        np.testing.assert_allclose(
            cutter.bounds[1], part.bounds[1] + 0.05, atol=1e-9
        )
        self.assertGreater(abs(float(cutter.volume)), abs(float(part.volume)))
        self.assertEqual(record["dilation_method"], "manifold_minkowski_sphere")
        self.assertAlmostEqual(record["total_added_fit_clearance_mm"], 0.10, places=9)

    def test_complete_child_boolean_proxy_moves_only_source_patch_outward(self) -> None:
        vertices = np.asarray(
            [
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
                [-1.0, -1.0, -2.0],
                [1.0, -1.0, -2.0],
                [1.0, 1.0, -2.0],
                [-1.0, 1.0, -2.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [
                [0, 1, 2], [0, 2, 3],
                [0, 4, 5], [0, 5, 1],
                [1, 5, 6], [1, 6, 2],
                [2, 6, 7], [2, 7, 3],
                [3, 7, 4], [3, 4, 0],
                [4, 7, 6], [4, 6, 5],
            ],
            dtype=np.int64,
        )
        child = common.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        common.trimesh.repair.fix_winding(child)
        common.trimesh.repair.fix_normals(child)
        child.metadata["protected_source_face_count"] = 2
        child_vertices_before = np.asarray(child.vertices).copy()
        child_faces_before = np.asarray(child.faces).copy()

        cutter, record = build_boolean_cutter_proxy_from_final_part(
            child,
            radial_clearance_mm=0.0,
            exterior_overshoot_mm=0.01,
        )

        np.testing.assert_array_equal(child.vertices, child_vertices_before)
        np.testing.assert_array_equal(child.faces, child_faces_before)
        self.assertTrue(cutter.is_watertight)
        self.assertTrue(cutter.is_winding_consistent)
        self.assertAlmostEqual(float(cutter.bounds[1, 2]), 0.01, places=9)
        self.assertFalse(record["exact_source_rim_preserved"])
        self.assertTrue(record["proxy_source_rim_shifted_outward"])
        self.assertEqual(record["collar_face_count"], 0)
        self.assertFalse(record["printable_child_mutated"])
        self.assertEqual(record["proxy_degenerate_faces"], 0)

        parent = common.trimesh.creation.box(extents=[10.0, 10.0, 4.0])
        parent.apply_translation([0.0, 0.0, -2.0])
        socketed, boolean_record = subtract_socket_cutters(parent, [cutter])
        audit = boolean_collapsed_face_audit(socketed)
        self.assertTrue(socketed.is_watertight)
        self.assertTrue(socketed.is_winding_consistent)
        self.assertEqual(audit["collapsed_face_count"], 0)
        self.assertGreater(boolean_record["intersection_volume_mm3"], 1.0)

    def test_curved_source_proxy_uses_local_outward_direction_field(self) -> None:
        vertices = np.asarray(
            [
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.12],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, -0.12],
                [0.0, 0.0, 0.45],
                [-1.0, -1.0, -2.0],
                [1.0, -1.0, -2.0],
                [1.0, 1.0, -2.0],
                [-1.0, 1.0, -2.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [
                [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
                [0, 5, 6], [0, 6, 1],
                [1, 6, 7], [1, 7, 2],
                [2, 7, 8], [2, 8, 3],
                [3, 8, 5], [3, 5, 0],
                [5, 8, 7], [5, 7, 6],
            ],
            dtype=np.int64,
        )
        child = common.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        common.trimesh.repair.fix_winding(child)
        common.trimesh.repair.fix_normals(child)
        child.metadata["protected_source_face_count"] = 4

        proxy, record = build_boolean_cutter_proxy_from_final_part(
            child,
            radial_clearance_mm=0.0,
            exterior_overshoot_mm=0.01,
        )

        self.assertTrue(proxy.is_watertight)
        self.assertTrue(proxy.is_winding_consistent)
        self.assertEqual(record["proxy_degenerate_faces"], 0)
        self.assertGreater(
            max(record["source_patch_outward_direction_span"][:2]),
            0.05,
        )

    def test_wraparound_source_patch_uses_interface_cap_proxy(self) -> None:
        """A remote inward source lobe must not become a second exit cutter."""

        vertices = np.asarray(
            [
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
                [0.0, 0.0, -0.75],
                [-1.0, -1.0, -2.0],
                [1.0, -1.0, -2.0],
                [1.0, 1.0, -2.0],
                [-1.0, 1.0, -2.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [
                [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
                [0, 5, 6], [0, 6, 1],
                [1, 6, 7], [1, 7, 2],
                [2, 7, 8], [2, 8, 3],
                [3, 8, 5], [3, 5, 0],
                [5, 8, 7], [5, 7, 6],
            ],
            dtype=np.int64,
        )
        child = common.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        common.trimesh.repair.fix_winding(child)
        common.trimesh.repair.fix_normals(child)
        child.metadata["protected_source_face_count"] = 4

        proxy, record = build_boolean_cutter_proxy_from_final_part(
            child,
            radial_clearance_mm=0.0,
            exterior_overshoot_mm=0.01,
        )

        self.assertTrue(proxy.is_watertight)
        self.assertTrue(proxy.is_winding_consistent)
        self.assertEqual(
            record["proxy_method"],
            "interface_cap_and_generated_attachment_overshoot",
        )
        self.assertTrue(record["source_patch_replaced_by_interface_cap"])
        self.assertTrue(record["interface_cap_triangulated_after_rim_overshoot"])
        self.assertTrue(record["interface_cap_occupied_edge_guard"])
        self.assertFalse(
            record["source_patch_reentry"]["complete_source_patch_proxy_safe"]
        )
        self.assertGreater(
            record["source_patch_reentry"]["maximum_remote_inward_reentry_mm"],
            0.70,
        )
        self.assertEqual(record["interface_cap_face_count"], 2)
        self.assertLess(len(proxy.faces), len(child.faces))

    def test_symmetric_attachment_uses_oriented_rim_for_inward_axis(self) -> None:
        """Radial attachment vectors may cancel without making the rim ambiguous."""

        vertices = np.asarray(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ],
            dtype=np.float64,
        )
        source_faces = np.asarray(
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
            dtype=np.int64,
        )
        generated_faces = np.asarray(
            [
                [0, 5, 1], [1, 5, 6],
                [1, 6, 2], [2, 6, 7],
                [2, 7, 3], [3, 7, 8],
                [3, 8, 0], [0, 8, 5],
            ],
            dtype=np.int64,
        )

        record = assess_source_patch_reentry(
            vertices,
            source_faces,
            generated_faces,
            overshoot_mm=0.01,
        )

        self.assertEqual(
            record["inward_axis_inference"],
            "oriented_source_rim_area_fallback",
        )
        np.testing.assert_allclose(record["inward_axis"], [0.0, 0.0, -1.0])

    def test_complete_child_audit_accepts_parent_already_disjoint_after_socket(self) -> None:
        parent = common.trimesh.creation.box(extents=[20.0, 20.0, 8.0])
        cutter = common.trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        cutter.apply_translation([0.0, 0.0, 10.0])

        unchanged, record = subtract_socket_cutters(
            parent,
            [cutter],
            allow_empty_intersection=True,
        )

        self.assertTrue(unchanged.is_watertight)
        self.assertTrue(record["boolean_skipped"])
        self.assertEqual(
            record["boolean_skip_reason"],
            "already_disjoint_after_clearance_socket",
        )
        self.assertAlmostEqual(
            record["parent_volume_after_mm3"],
            record["parent_volume_before_mm3"],
            places=6,
        )

    def test_production_connector_reserves_three_mm_backing_inside_five_mm_total(self) -> None:
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=5.0,
        )

        self.assertAlmostEqual(spec.full_boundary_backing_depth_mm, 3.0, places=6)
        self.assertAlmostEqual(spec.engagement_depth_mm, 1.75, places=6)
        self.assertAlmostEqual(
            spec.full_boundary_backing_depth_mm
            + spec.engagement_depth_mm
            + spec.socket_bottom_clearance_mm,
            5.0,
            places=6,
        )
        self.assertAlmostEqual(spec.peg_tip_chamfer_mm, 0.0, places=6)

    def test_zero_tip_chamfer_makes_vertical_peg_with_flat_tip(self) -> None:
        square = np.asarray(
            [
                [-6.0, -6.0, 0.0],
                [6.0, -6.0, 0.0],
                [6.0, 6.0, 0.0],
                [-6.0, 6.0, 0.0],
            ],
            dtype=np.float64,
        )
        plan = plan_local_connector(
            square,
            np.asarray([0.0, 0.0, -1.0]),
            self.spec,
            samples=32,
        )
        u = np.asarray(plan["u"])
        v = np.asarray(plan["v"])
        center = np.asarray(plan["center"])

        def projected(points):
            relative = np.asarray(points) - center[None, :]
            return np.column_stack((relative @ u, relative @ v))

        np.testing.assert_allclose(
            projected(plan["peg_top"]),
            projected(plan["peg_tip"]),
            atol=1e-9,
        )
        axial_depth = (
            (np.asarray(plan["peg_tip"]) - np.asarray(plan["peg_top"]))
            @ np.asarray(plan["inward"])
        )
        np.testing.assert_allclose(
            axial_depth,
            np.full(len(axial_depth), self.spec.engagement_depth_mm),
            atol=1e-9,
        )

    def test_square_inset_keeps_each_straight_edge_straight(self) -> None:
        samples_per_edge = 16
        parameter = np.linspace(-6.0, 6.0, samples_per_edge, endpoint=False)
        square = np.vstack(
            (
                np.column_stack((parameter, np.full_like(parameter, -6.0), np.zeros_like(parameter))),
                np.column_stack((np.full_like(parameter, 6.0), parameter, np.zeros_like(parameter))),
                np.column_stack((-parameter, np.full_like(parameter, 6.0), np.zeros_like(parameter))),
                np.column_stack((np.full_like(parameter, -6.0), -parameter, np.zeros_like(parameter))),
            )
        )
        plan = plan_local_connector(
            square,
            np.asarray([0.0, 0.0, -1.0]),
            self.spec,
            samples=32,
        )
        displacement = _line_preserving_inset_displacements(square, plan, 2.0)
        inset = square + displacement

        # Ignore the first corner sample on each run; every non-corner sample
        # along one source edge must land on one exact straight inset line.
        edge_coordinates = (
            inset[1:samples_per_edge, 1],
            inset[samples_per_edge + 1 : 2 * samples_per_edge, 0],
            inset[2 * samples_per_edge + 1 : 3 * samples_per_edge, 1],
            inset[3 * samples_per_edge + 1 :, 0],
        )
        for coordinates in edge_coordinates:
            self.assertLess(float(np.ptp(coordinates)), 1e-9)

    def test_square_backing_annulus_has_no_folded_corner_faces(self) -> None:
        samples_per_edge = 24
        parameter = np.linspace(-6.0, 6.0, samples_per_edge, endpoint=False)
        square = np.vstack(
            (
                np.column_stack((parameter, np.full_like(parameter, -6.0), np.zeros_like(parameter))),
                np.column_stack((np.full_like(parameter, 6.0), parameter, np.zeros_like(parameter))),
                np.column_stack((-parameter, np.full_like(parameter, 6.0), np.zeros_like(parameter))),
                np.column_stack((np.full_like(parameter, -6.0), -parameter, np.zeros_like(parameter))),
            )
        )
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=5.0,
        )
        plan = plan_local_connector(
            square,
            np.asarray([0.0, 0.0, -1.0]),
            spec,
            samples=64,
        )
        _lead, outer, _depth = _printable_backing_rings(
            square,
            plan,
            child_clearance=True,
        )
        vertices = [point.copy() for point in outer]
        faces = []
        _inner_ids, _order, added = _add_constrained_connector_annulus(
            output_vertices=vertices,
            output_faces=faces,
            outer_ids=list(range(len(outer))),
            outer_points=outer,
            inner_points=np.asarray(plan["peg_top"]),
            plan=plan,
            outward_normal=np.asarray(plan["inward"]),
        )
        points = np.asarray(vertices)
        triangles = points[np.asarray(faces)]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        dots = normals @ np.asarray(plan["inward"])

        def polygon_area(projected):
            projected = np.asarray(projected)
            return 0.5 * abs(
                float(
                    np.sum(
                        projected[:, 0] * np.roll(projected[:, 1], -1)
                        - np.roll(projected[:, 0], -1) * projected[:, 1]
                    )
                )
            )

        u = np.asarray(plan["u"])
        v = np.asarray(plan["v"])
        center = np.asarray(plan["center"])
        project = lambda ring: np.column_stack(
            ((np.asarray(ring) - center[None, :]) @ u, (np.asarray(ring) - center[None, :]) @ v)
        )
        projected_triangles = np.stack(
            [project(triangle) for triangle in triangles], axis=0
        )
        first_edges = projected_triangles[:, 1] - projected_triangles[:, 0]
        second_edges = projected_triangles[:, 2] - projected_triangles[:, 0]
        triangle_area = 0.5 * np.abs(
            first_edges[:, 0] * second_edges[:, 1]
            - first_edges[:, 1] * second_edges[:, 0]
        ).sum()
        expected_area = polygon_area(project(outer)) - polygon_area(
            project(plan["peg_top"])
        )

        self.assertGreater(added, 0)
        self.assertTrue(np.all(dots > 1e-12))
        self.assertAlmostEqual(float(triangle_area), expected_area, places=6)

    def test_concave_v_backing_keeps_curve_inside_and_faces_consistent(self) -> None:
        def cubic(p0, p1, p2, p3, samples=128):
            parameter = np.linspace(0.0, 1.0, samples, endpoint=False)
            remaining = 1.0 - parameter
            return (
                (remaining**3)[:, None] * np.asarray(p0)[None, :]
                + (3.0 * remaining**2 * parameter)[:, None] * np.asarray(p1)[None, :]
                + (3.0 * remaining * parameter**2)[:, None] * np.asarray(p2)[None, :]
                + (parameter**3)[:, None] * np.asarray(p3)[None, :]
            )

        left_top = (-7.5, 4.0, 0.0)
        bottom = (0.0, -7.0, 0.0)
        right_top = (7.5, 4.0, 0.0)
        boundary = np.vstack(
            (
                cubic(left_top, (-5.6, 0.0, 0.0), (-2.4, -7.0, 0.0), bottom),
                cubic(bottom, (2.4, -7.0, 0.0), (5.6, 0.0, 0.0), right_top),
                cubic(right_top, (3.2, 1.0, 0.0), (-3.2, 1.0, 0.0), left_top),
            )
        )
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=5.0,
        )
        plan = plan_local_connector(
            boundary,
            np.asarray([0.0, 0.0, -1.0]),
            spec,
            samples=64,
        )
        lead, backing, depth = _printable_backing_rings(
            boundary,
            plan,
            child_clearance=True,
        )
        vertices = [point.copy() for point in backing]
        faces = []
        _inner_ids, _order, added = _add_constrained_connector_annulus(
            output_vertices=vertices,
            output_faces=faces,
            outer_ids=list(range(len(backing))),
            outer_points=backing,
            inner_points=np.asarray(plan["peg_top"]),
            plan=plan,
            outward_normal=np.asarray(plan["inward"]),
        )
        triangles = np.asarray(vertices)[np.asarray(faces)]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )

        # The previous global corner correction pulled the rounded V below its
        # source outline and created one reversed triangle.  Both are explicit
        # regressions because slicer previews make them look like corner fans.
        self.assertGreater(added, 0)
        self.assertGreaterEqual(float(backing[:, 1].min()), float(boundary[:, 1].min()) - 1e-9)
        self.assertTrue(np.all(normals @ np.asarray(plan["inward"]) > 1e-12))
        # The visible backing wall must be the complete black-line profile:
        # one continuous outer-large/inner-small wedge from the source rim to
        # the 3 mm backing floor.  A triangulation fallback must never shorten
        # the wedge and append a visually vertical skirt below it.
        self.assertAlmostEqual(depth, 3.0, places=5)
        np.testing.assert_allclose(backing, lead, atol=1e-8)
        self.assertFalse(plan["backing_taper_shape_backoff_applied"])
        self.assertEqual(
            plan["backing_inset_method"],
            "clipper2_trimmed_constant_offset",
        )

        # Preserve the simplified inset's real topology and sew the dense
        # source ring to it with a geometry-aware unequal-count strip.  The
        # historical 384->384 resampling produced hundreds of collinear
        # points, needle triangles, and the inverted spikes seen in slicers.
        male_vertices = [point.copy() for point in boundary]
        male_faces = []
        male_record = add_local_male_connector_and_backing(
            output_vertices=male_vertices,
            output_faces=male_faces,
            boundary_ids=list(range(len(boundary))),
            boundary_points=boundary,
            inward=np.asarray([0.0, 0.0, -1.0]),
            spec=spec,
        )
        male_points = np.asarray(male_vertices, dtype=np.float64)
        male_triangles = male_points[np.asarray(male_faces, dtype=np.int64)]
        edge_lengths = np.stack(
            (
                np.linalg.norm(male_triangles[:, 1] - male_triangles[:, 0], axis=1),
                np.linalg.norm(male_triangles[:, 2] - male_triangles[:, 1], axis=1),
                np.linalg.norm(male_triangles[:, 0] - male_triangles[:, 2], axis=1),
            ),
            axis=1,
        )
        double_areas = np.linalg.norm(
            np.cross(
                male_triangles[:, 1] - male_triangles[:, 0],
                male_triangles[:, 2] - male_triangles[:, 0],
            ),
            axis=1,
        )
        aspects = edge_lengths.max(axis=1) ** 2 / double_areas
        self.assertEqual(male_record["backing_source_ring_vertices"], len(boundary))
        self.assertLess(male_record["backing_floor_ring_vertices"], len(boundary) // 2)
        wedge_face_count = (
            male_record["backing_source_ring_vertices"]
            + male_record["backing_floor_ring_vertices"]
        )
        self.assertGreater(float(double_areas[:wedge_face_count].min()), 1e-8)
        self.assertLess(float(aspects[:wedge_face_count].max()), 500.0)
        self.assertGreater(float(double_areas.min()), 1e-8)
        # The coplanar annulus may contain slender ears along the original
        # densely sampled planar-arc boundary. They preserve that real curve and
        # cannot become 3-D spikes; the constrained triangulator separately
        # audits every ring edge, orientation, and exact annulus area.
        planar_exception_faces = (
            male_record["backing_annulus_dense_sampling_faces"]
            + male_record["backing_annulus_planar_boundary_ear_faces"]
        )
        if planar_exception_faces == 0:
            # The layered wall can retain very short vendor perimeter edges;
            # fanout and bridge geometry, rather than one global aspect number,
            # decide whether those triangles can form a visible 3-D spike.
            self.assertLessEqual(male_record["backing_strip_maximum_fanout"], 8)
            self.assertLess(
                male_record["backing_strip_maximum_cross_edge_mm"],
                2.5,
            )
        self.assertIn(
            male_record["backing_annulus_triangulation_method"],
            {
                "manifold3d_constrained_polygon_with_hole",
                "audited_visible_bridge_polygon_with_hole",
            },
        )

        # The matching parent pre-boolean closure must use the same trimmed
        # offset policy.  Repairing only the male side leaves the dense mother
        # annulus self-crossing and the real split still fails before boolean
        # subtraction.
        parent_plan = plan_local_connector(
            boundary,
            np.asarray([0.0, 0.0, -1.0]),
            spec,
            samples=64,
        )
        _parent_lead, parent_backing, _parent_depth = _printable_backing_rings(
            boundary,
            parent_plan,
            child_clearance=False,
        )
        parent_vertices = [point.copy() for point in parent_backing]
        parent_faces = []
        _mouth_ids, _mouth_order, parent_added = _add_constrained_connector_annulus(
            output_vertices=parent_vertices,
            output_faces=parent_faces,
            outer_ids=list(range(len(parent_backing))),
            outer_points=parent_backing,
            inner_points=np.asarray(parent_plan["socket_mouth"]),
            plan=parent_plan,
            outward_normal=-np.asarray(parent_plan["inward"]),
        )
        self.assertGreater(parent_added, 0)
        self.assertEqual(
            parent_plan["backing_inset_method"],
            "clipper2_trimmed_constant_offset",
        )

    def test_concave_v_preflight_builds_the_actual_local_connector(self) -> None:
        def cubic(p0, p1, p2, p3, samples=128):
            parameter = np.linspace(0.0, 1.0, samples, endpoint=False)
            remaining = 1.0 - parameter
            return (
                (remaining**3)[:, None] * np.asarray(p0)[None, :]
                + (3.0 * remaining**2 * parameter)[:, None]
                * np.asarray(p1)[None, :]
                + (3.0 * remaining * parameter**2)[:, None]
                * np.asarray(p2)[None, :]
                + (parameter**3)[:, None] * np.asarray(p3)[None, :]
            )

        left_top = (-7.5, 4.0, 0.0)
        bottom = (0.0, -7.0, 0.0)
        right_top = (7.5, 4.0, 0.0)
        boundary = np.vstack(
            (
                cubic(left_top, (-5.6, 0.0, 0.0), (-2.4, -7.0, 0.0), bottom),
                cubic(bottom, (2.4, -7.0, 0.0), (5.6, 0.0, 0.0), right_top),
                cubic(right_top, (3.2, 1.0, 0.0), (-3.2, 1.0, 0.0), left_top),
            )
        )
        triangles, _normal, _record = inward_module.triangulate_ordered_loop_3d(
            boundary,
            np.asarray([0.0, 0.0, 1.0]),
        )
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.30,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=6.70,
            safe_backing_depth_mm=3.0,
        )

        quality = preflight_local_connector_patch(
            source_vertices=boundary,
            source_faces=np.asarray(triangles, dtype=np.int64),
            boundary_ids=list(range(len(boundary))),
            boundary_points=boundary,
            inward=np.asarray([0.0, 0.0, -1.0]),
            inward_directions=np.tile(
                np.asarray([0.0, 0.0, -1.0]),
                (len(boundary), 1),
            ),
            spec=spec,
        )

        self.assertEqual(len(boundary), 384)
        self.assertTrue(quality["valid"])
        self.assertEqual(quality["invalid_faces"], 0)
        self.assertEqual(quality["interface_geometry"], "local-connector")
        self.assertEqual(
            quality["side_wall_quality"]["strategy"],
            "geometry_aware_unequal_count_connector_strips",
        )
        self.assertEqual(quality["private_boolean_cutter_count"], 0)
        self.assertLess(
            quality["connector_record"]["backing_floor_ring_vertices"],
            len(boundary),
        )

    def test_dense_aligned_backing_strip_is_not_misclassified_as_a_spike(self) -> None:
        vertices = [
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0.001, 0.0, 0.0]),
            np.asarray([0.0, 0.0, -3.0]),
            np.asarray([0.001, 0.0, -3.0]),
        ]
        faces = [[0, 1, 3], [0, 3, 2]]
        record = inward_module._validate_connector_wedge_faces(
            vertices,
            faces,
            0,
            perimeter_edges={(0, 1), (2, 3)},
        )

        self.assertGreater(record["backing_wedge_max_aspect"], 750.0)
        self.assertEqual(record["backing_wedge_dense_sampling_faces"], 2)
        self.assertEqual(
            record["backing_wedge_aspect_policy"],
            "strict_750_or_aligned_perimeter_dense_sampling",
        )

    def test_near_collinear_perimeter_needle_is_still_rejected(self) -> None:
        vertices = [
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0.001, 0.0, 0.0]),
            np.asarray([3.0, 0.001, 0.0]),
        ]
        from split3mf.print_tolerance import PrintTolerance, tolerance_scope
        with tolerance_scope(PrintTolerance(micro_area_mm2=0)), self.assertRaisesRegex(ValueError, "needle faces"):
            inward_module._validate_connector_wedge_faces(
                vertices,
                [[0, 1, 2]],
                0,
                perimeter_edges={(0, 1)},
            )

    def test_ratio_mode_accepts_only_sparse_print_neutral_needles(self) -> None:
        vertices: list[np.ndarray] = []
        faces: list[list[int]] = []
        for index in range(1000):
            base = 3 * index
            x = 3.0 * index
            vertices.extend(
                (
                    np.asarray([x, 0.0, 0.0]),
                    np.asarray([x + 1.0, 0.0, 0.0]),
                    np.asarray([x, 1.0, 0.0]),
                )
            )
            faces.append([base, base + 1, base + 2])
        vertices[-3:] = (
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0.01, 0.0, 0.0]),
            np.asarray([1.95, 0.011, 0.0]),
        )

        record = inward_module._validate_connector_wedge_faces(
            vertices,
            faces,
            0,
        )

        self.assertEqual(record["backing_wedge_ratio_accepted_needle_faces"], 1)
        self.assertAlmostEqual(record["backing_wedge_needle_face_ratio"], 0.001)

    def test_balanced_cross_layer_coincident_ear_is_not_a_spike(self) -> None:
        vertices = [
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0.0003, 0.0, 0.0]),
            np.asarray([0.0, 0.25, -0.25]),
        ]
        record = inward_module._validate_connector_wedge_faces(
            vertices,
            [[0, 1, 2]],
            0,
            coherent_boundary_pairs=[{0, 1, 2}],
        )

        self.assertGreater(record["backing_wedge_max_aspect"], 750.0)
        self.assertEqual(
            record["backing_wedge_cross_layer_coincident_ear_faces"],
            1,
        )

    def test_shallow_dense_perimeter_ear_requires_adjacent_layers(self) -> None:
        vertices = [
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0.00001, 0.0, 0.0]),
            np.asarray([0.000005, 0.02, -0.001]),
        ]
        record = inward_module._validate_connector_wedge_faces(
            vertices,
            [[0, 1, 2]],
            0,
            perimeter_edges={(0, 1)},
            coherent_boundary_rings=[{0, 1}, {2, 3}],
            coherent_boundary_pairs=[{0, 1, 2, 3}],
        )

        self.assertEqual(
            record["backing_wedge_cross_layer_dense_perimeter_ear_faces"],
            1,
        )

    def test_sub_layer_perimeter_sliver_is_rejected(self) -> None:
        vertices = [
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0.00001, 0.0, 0.0]),
            np.asarray([0.000005, 0.02, 0.0]),
        ]
        from split3mf.print_tolerance import PrintTolerance, tolerance_scope
        with tolerance_scope(PrintTolerance(micro_area_mm2=0)), self.assertRaisesRegex(ValueError, "needle faces"):
            inward_module._validate_connector_wedge_faces(
                vertices,
                [[0, 1, 2]],
                0,
                perimeter_edges={(0, 1)},
            )

    def test_dense_concave_annulus_preserves_both_ring_boundaries(self) -> None:
        def sample_segment(start, end, count):
            parameter = np.linspace(0.0, 1.0, count, endpoint=False)
            return (
                np.asarray(start)[None, :] * (1.0 - parameter[:, None])
                + np.asarray(end)[None, :] * parameter[:, None]
            )

        outline_2d = np.vstack(
            (
                sample_segment((-10.0, -8.0), (10.0, -8.0), 150),
                sample_segment((10.0, -8.0), (10.0, 8.0), 120),
                sample_segment((10.0, 8.0), (2.0, 2.0), 100),
                sample_segment((2.0, 2.0), (0.0, 7.0), 70),
                sample_segment((0.0, 7.0), (-2.0, 2.0), 70),
                sample_segment((-2.0, 2.0), (-10.0, 8.0), 100),
                sample_segment((-10.0, 8.0), (-10.0, -8.0), 120),
            )
        )
        inner_parameter = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        inner_2d = np.column_stack(
            (2.0 * np.cos(inner_parameter), 1.0 * np.sin(inner_parameter) - 3.0)
        )
        outer_ids = list(range(len(outline_2d)))
        inner_ids = list(range(len(outline_2d), len(outline_2d) + len(inner_2d)))
        triangles = inward_module.triangulate_loop_group_with_bridges(
            [outer_ids, inner_ids],
            [outline_2d, inner_2d],
            [0, 1],
        )

        edge_counts = collections.Counter(
            tuple(sorted(edge))
            for face in triangles
            for edge in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            )
        )
        required = {
            tuple(sorted((ring[position], ring[(position + 1) % len(ring)])))
            for ring in (outer_ids, inner_ids)
            for position in range(len(ring))
        }
        self.assertTrue(triangles)
        self.assertEqual(
            {edge for edge, count in edge_counts.items() if count == 1},
            required,
        )
        self.assertTrue(all(edge_counts[edge] == 1 for edge in required))

        # Use a representative down-sample for the forced-path check; the
        # full 700-point case above already protects dense production input,
        # while deliberately recomputing every possible visible bridge at the
        # same density would turn a unit test into a coffee break.
        fallback_outer = outline_2d[::10]
        fallback_inner = inner_2d[::4]
        fallback_outer_ids = list(range(len(fallback_outer)))
        fallback_inner_ids = list(
            range(len(fallback_outer), len(fallback_outer) + len(fallback_inner))
        )
        fallback_required = {
            tuple(sorted((ring[position], ring[(position + 1) % len(ring)])))
            for ring in (fallback_outer_ids, fallback_inner_ids)
            for position in range(len(ring))
        }
        original_ear_clip = mesh_module.triangulate_polygon_ear_clip
        ear_clip_calls = 0

        def skip_weak_polygon_once(points):
            nonlocal ear_clip_calls
            ear_clip_calls += 1
            if ear_clip_calls == 1:
                return []
            return original_ear_clip(points)

        # Force the conventional duplicate-bridge representation to fail so
        # the two-distinct-bridge partition is exercised directly.
        with mock.patch.object(
            mesh_module,
            "triangulate_polygon_ear_clip",
            side_effect=skip_weak_polygon_once,
        ):
            split_triangles = mesh_module.triangulate_loop_group_with_bridges(
                [fallback_outer_ids, fallback_inner_ids],
                [fallback_outer, fallback_inner],
                [0, 1],
            )
        split_edge_counts = collections.Counter(
            tuple(sorted(edge))
            for face in split_triangles
            for edge in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            )
        )
        self.assertGreater(ear_clip_calls, 1)
        self.assertTrue(split_triangles)
        self.assertEqual(
            {edge for edge, count in split_edge_counts.items() if count == 1},
            fallback_required,
        )
        self.assertTrue(
            all(split_edge_counts[edge] == 1 for edge in fallback_required)
        )
        self.assertTrue(all(count <= 2 for count in edge_counts.values()))

    def test_annulus_triangulation_does_not_shorten_the_visible_wedge(self) -> None:
        square = np.asarray(
            [
                [-6.0, -6.0, 0.0],
                [6.0, -6.0, 0.0],
                [6.0, 6.0, 0.0],
                [-6.0, 6.0, 0.0],
            ]
        )
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=5.0,
        )
        plan = plan_local_connector(
            square,
            np.asarray([0.0, 0.0, -1.0]),
            spec,
            samples=64,
        )

        with mock.patch.object(
            inward_module,
            "triangulate_loop_group_with_bridges",
            return_value=[],
        ) as triangulate:
            lead, backing, taper_depth = _printable_backing_rings(
                square,
                plan,
                child_clearance=True,
            )

        triangulate.assert_not_called()
        self.assertAlmostEqual(taper_depth, 3.0, places=6)
        np.testing.assert_allclose(backing, lead, atol=1e-8)
        self.assertFalse(plan["backing_taper_shape_backoff_applied"])

    def test_high_density_annulus_keeps_requested_45_degree_taper(self) -> None:
        samples_per_edge = 130
        parameter = np.linspace(-6.0, 6.0, samples_per_edge, endpoint=False)
        boundary = np.vstack(
            (
                np.column_stack((parameter, np.full_like(parameter, -6.0), np.zeros_like(parameter))),
                np.column_stack((np.full_like(parameter, 6.0), parameter, np.zeros_like(parameter))),
                np.column_stack((-parameter, np.full_like(parameter, 6.0), np.zeros_like(parameter))),
                np.column_stack((np.full_like(parameter, -6.0), -parameter, np.zeros_like(parameter))),
            )
        )
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=5.0,
        )
        plan = plan_local_connector(
            boundary,
            np.asarray([0.0, 0.0, -1.0]),
            spec,
            samples=64,
        )

        with mock.patch.object(
            inward_module,
            "triangulate_loop_group_with_bridges",
            return_value=[[0, 1, 2]],
        ) as triangulate:
            lead, backing, taper_depth = _printable_backing_rings(
                boundary,
                plan,
                child_clearance=True,
            )

        triangulate.assert_not_called()
        self.assertAlmostEqual(taper_depth, 3.0, places=6)
        np.testing.assert_allclose(backing, lead, atol=1e-8)
        self.assertFalse(plan["backing_taper_shape_backoff_applied"])

    def test_backing_sweep_uses_one_planar_axis_not_per_vertex_normals(self) -> None:
        samples = 64
        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        boundary = np.column_stack(
            (8.0 * np.cos(angles), 8.0 * np.sin(angles), np.zeros(samples))
        )
        directions = np.column_stack(
            (
                0.12 * np.cos(angles),
                0.12 * np.sin(angles),
                -np.ones(samples),
            )
        )
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=5.0,
        )
        plan = plan_local_connector(
            boundary,
            np.asarray([0.0, 0.0, -1.0]),
            spec,
            samples=64,
        )

        lead, backing, taper_depth = _printable_backing_rings(
            boundary,
            plan,
            child_clearance=True,
            inward_directions=directions,
        )

        self.assertGreater(taper_depth, 0.0)
        self.assertAlmostEqual(taper_depth, 3.0, places=6)
        np.testing.assert_allclose(backing, lead, atol=1e-8)
        self.assertEqual(
            plan["backing_inward_direction_mode"],
            "single_global_planar_direction_from_safe_axis",
        )
        self.assertTrue(plan["backing_safe_direction_field_received"])

    def test_nonplanar_source_boundary_produces_a_planar_backing_floor(self) -> None:
        samples = 96
        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        boundary = np.column_stack(
            (
                8.0 * np.cos(angles),
                6.0 * np.sin(angles),
                0.8 * np.sin(3.0 * angles),
            )
        )
        directions = np.column_stack(
            (
                0.15 * np.cos(angles),
                0.15 * np.sin(angles),
                -np.ones(samples),
            )
        )
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=5.0,
        )
        plan = plan_local_connector(
            boundary,
            np.asarray([0.0, 0.0, -1.0]),
            spec,
            samples=64,
        )

        lead, backing, taper_depth = _printable_backing_rings(
            boundary,
            plan,
            child_clearance=True,
            inward_directions=directions,
        )

        axis = np.asarray(plan["inward"], dtype=np.float64)
        self.assertGreater(taper_depth, 0.0)
        self.assertLess(float(np.ptp(lead @ axis)), 1e-8)
        self.assertLess(float(np.ptp(backing @ axis)), 1e-8)

    def test_rotated_inset_seam_keeps_source_height_at_physical_location(self) -> None:
        samples = 192
        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        source = np.column_stack(
            (
                9.0 * np.cos(angles),
                6.5 * np.sin(angles),
                1.25 * np.sin(2.0 * angles) + 0.30 * np.cos(5.0 * angles),
            )
        )
        inner_angles = np.linspace(0.0, 2.0 * np.pi, 137, endpoint=False)
        planar = np.column_stack(
            (
                8.25 * np.cos(inner_angles),
                5.75 * np.sin(inner_angles),
                np.zeros(len(inner_angles)),
            )
        )
        # Clipper may legally return any cyclic seam.  Rolling must not rotate
        # the source height field around the physical connector.
        planar = np.roll(planar, 53, axis=0)
        plan = {
            "center": np.zeros(3, dtype=np.float64),
            "u": np.asarray([1.0, 0.0, 0.0]),
            "v": np.asarray([0.0, 1.0, 0.0]),
            "inward": np.asarray([0.0, 0.0, 1.0]),
        }

        carried = connector_geometry_module._carry_source_height_into_transition(
            source,
            planar,
            plan,
            depth_mm=0.25,
            backing_depth_mm=3.0,
        )
        source_axial = source[:, 2]
        expected_source_residual = connector_geometry_module._project_ring_scalar_field(
            planar[:, :2],
            source[:, :2],
            source_axial - float(np.mean(source_axial)),
        )
        expected = 0.25 + (1.0 - 0.25 / 3.0) * (
            expected_source_residual
        )
        np.testing.assert_allclose(carried[:, 2], expected, atol=1e-9)

        unrolled = np.roll(carried, -53, axis=0)
        self.assertLess(float(np.max(np.abs(np.diff(unrolled[:, 2])))), 0.20)

    def test_zero_taper_male_attachment_cutter_reuses_boundary_seam(self) -> None:
        boundary = np.asarray(
            [
                [-6.0, -6.0, 0.0],
                [6.0, -6.0, 0.0],
                [6.0, 6.0, 0.0],
                [-6.0, 6.0, 0.0],
            ]
        )
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=5.0,
        )

        def zero_taper_backing(points, plan, *, child_clearance):
            del child_clearance
            points = np.asarray(points, dtype=np.float64)
            axis = np.asarray(plan["inward"], dtype=np.float64)
            return points.copy(), points + axis[None, :] * 3.0, 0.0

        with mock.patch.object(
            inward_module,
            "_printable_backing_rings",
            side_effect=zero_taper_backing,
        ):
            cutter = build_local_male_attachment_cutter(
                boundary_points=boundary,
                inward=np.asarray([0.0, 0.0, -1.0]),
                spec=spec,
                outside_extension_mm=4.0,
            )

        self.assertTrue(cutter.is_watertight)
        self.assertTrue(cutter.is_winding_consistent)

    def test_male_attachment_cutter_preserves_prebuilt_topology(self) -> None:
        boundary = np.asarray(
            [
                [-6.0, -6.0, 0.0],
                [0.0, -6.0, 0.0],
                [6.0, -6.0, 0.0],
                [6.0, 6.0, 0.0],
                [-6.0, 6.0, 0.0],
            ]
        )
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=5.0,
        )
        real_constructor = inward_module.trimesh.Trimesh

        with mock.patch.object(
            inward_module.trimesh,
            "Trimesh",
            wraps=real_constructor,
        ) as constructor:
            cutter = build_local_male_attachment_cutter(
                boundary_points=boundary,
                inward=np.asarray([0.0, 0.0, -1.0]),
                spec=spec,
                outside_extension_mm=4.0,
            )

        self.assertFalse(constructor.call_args.kwargs["process"])
        self.assertTrue(cutter.is_watertight)
        self.assertTrue(cutter.is_winding_consistent)

    def test_isolated_thin_rim_points_do_not_shorten_the_compact_center_connector(self) -> None:
        distances = np.full(100, 5.0, dtype=np.float64)
        distances[:6] = 3.272
        safety = local_connector_safe_depth_from_field(distances)

        self.assertAlmostEqual(safety["boundary_safety_minimum_mm"], 3.272)
        self.assertAlmostEqual(safety["local_connector_safety_budget_mm"], 5.0)
        self.assertTrue(safety["thin_boundary_override_applied"])

    def test_thin_rim_shrinks_backing_then_uses_remaining_center_depth(self) -> None:
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=5.0,
            safe_backing_depth_mm=1.095,
        )

        self.assertTrue(spec.elastic_shrink_applied)
        self.assertAlmostEqual(spec.full_boundary_backing_depth_mm, 1.095)
        self.assertAlmostEqual(spec.engagement_depth_mm, 3.655)
        self.assertAlmostEqual(
            spec.full_boundary_backing_depth_mm
            + spec.engagement_depth_mm
            + spec.socket_bottom_clearance_mm,
            5.0,
        )
        self.assertLess(spec.peg_width_mm, 4.0)
        self.assertGreaterEqual(spec.peg_width_mm, 1.20)
        self.assertAlmostEqual(
            spec.elastic_lateral_scale,
            1.095 / 3.0,
        )

    def test_zero_engagement_builds_closed_backing_without_degenerate_peg(self) -> None:
        samples = 64
        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        boundary = np.column_stack(
            (8.0 * np.cos(angles), 8.0 * np.sin(angles), np.zeros(samples))
        )
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.60,
            safe_engagement_depth_mm=3.25,
            safe_backing_depth_mm=5.0,
        )
        vertices = [point.copy() for point in boundary]
        vertices.append(np.asarray([0.0, 0.0, 6.0]))
        faces = [
            [index, (index + 1) % samples, samples]
            for index in range(samples)
        ]

        record = add_local_male_connector_and_backing(
            output_vertices=vertices,
            output_faces=faces,
            boundary_ids=list(range(samples)),
            boundary_points=boundary,
            inward=np.asarray([0.0, 0.0, -1.0]),
            spec=spec,
        )
        mesh = common.trimesh.Trimesh(
            vertices=np.asarray(vertices),
            faces=np.asarray(faces),
            process=False,
        )
        triangles = mesh.vertices[mesh.faces]
        double_areas = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )

        self.assertTrue(mesh.is_watertight)
        common.trimesh.repair.fix_winding(mesh)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertGreater(float(double_areas.min()), 1e-12)
        self.assertTrue(record["compact_peg_omitted"])
        self.assertEqual(record["connector_side_faces_added"], 0)
        self.assertGreater(record["connector_cap_faces_added"], 0)

    def test_zero_engagement_boolean_closure_builds_only_backing_cutter(self) -> None:
        samples = 64
        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        boundary = np.column_stack(
            (8.0 * np.cos(angles), 8.0 * np.sin(angles), np.zeros(samples))
        )
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.60,
            safe_engagement_depth_mm=3.25,
            safe_backing_depth_mm=5.0,
        )
        vertices = [point.copy() for point in boundary]
        faces: list[list[int]] = []

        record, cutters = add_local_female_boolean_closure(
            output_vertices=vertices,
            output_faces=faces,
            boundary_ids=list(range(samples)),
            boundary_points=boundary,
            inward=np.asarray([0.0, 0.0, -1.0]),
            spec=spec,
            build_boolean_cutters=True,
        )

        self.assertEqual(len(cutters), 1)
        self.assertTrue(cutters[0].is_watertight)
        self.assertFalse(record["boolean_socket_required"])
        self.assertEqual(record["planned_cavity_cutter_count"], 1)
        self.assertEqual(
            record["boolean_cutter_scope"],
            "complete_male_backing_only_zero_engagement",
        )

    def test_compact_footprint_probe_can_override_uniform_rim_budget(self) -> None:
        class Probe:
            def safety_limit(self, points, directions, maximum):
                self.points = np.asarray(points)
                self.directions = np.asarray(directions)
                return 5.0, {"safe_maximum_inward_depth_mm": 5.0}

        samples = 64
        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        boundary = np.column_stack(
            (8.0 * np.cos(angles), 8.0 * np.sin(angles), np.zeros(samples))
        )
        probe = Probe()
        safety = local_connector_safe_depth_at_footprint(
            boundary_points=boundary,
            inward=np.asarray([0.0, 0.0, -1.0]),
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            boundary_distances=np.full(samples, 3.272),
            parent_thickness_probe=probe,
        )

        self.assertAlmostEqual(safety["boundary_safety_minimum_mm"], 3.272)
        self.assertAlmostEqual(safety["local_connector_safety_budget_mm"], 5.0)
        self.assertTrue(safety["local_connector_footprint_probe_applied"])
        self.assertEqual(probe.points.shape, (33, 3))

    def test_internal_footprint_not_grazing_rim_controls_backing_depth(self) -> None:
        class Probe:
            def safety_limit(self, points, directions, maximum):
                self.points = np.asarray(points)
                self.directions = np.asarray(directions)
                return 2.838, {"safe_maximum_inward_depth_mm": 2.838}

        # A narrow curved-strip analogue: rim rays report 1.095 mm after
        # grazing a sidewall, while its interior has about 2.8 mm of usable
        # axial and lateral room.  The false rim minimum remains diagnostic but
        # must no longer collapse the complete backing.
        boundary = np.asarray(
            [
                [-6.0, -2.82, 0.0],
                [6.0, -2.82, 0.0],
                [6.0, 2.82, 0.0],
                [-6.0, 2.82, 0.0],
            ],
            dtype=np.float64,
        )
        probe = Probe()
        safety = local_connector_safe_depth_at_footprint(
            boundary_points=boundary,
            inward=np.asarray([0.0, 0.0, -1.0]),
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.0,
            lead_in_mm=0.60,
            boundary_distances=np.full(len(boundary), 1.095),
            parent_thickness_probe=probe,
        )
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.0,
            lead_in_mm=0.60,
            safe_engagement_depth_mm=float(
                safety["local_connector_safety_budget_mm"]
            ),
            safe_backing_depth_mm=float(
                safety["local_connector_backing_safety_limit_mm"]
            ),
        )

        self.assertAlmostEqual(safety["boundary_safety_minimum_mm"], 1.095)
        self.assertAlmostEqual(
            safety["local_connector_lateral_backing_limit_mm"],
            2.82,
        )
        self.assertAlmostEqual(
            safety["local_connector_backing_safety_limit_mm"],
            2.82,
        )
        self.assertTrue(safety["thin_boundary_override_applied"])
        self.assertAlmostEqual(spec.full_boundary_backing_depth_mm, 2.82)
        self.assertAlmostEqual(spec.engagement_depth_mm, 0.0)
        self.assertFalse(spec.compact_peg_enabled)

    def test_production_backing_is_outer_large_inner_small_at_45_degrees(self) -> None:
        samples = 64
        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        ring = np.column_stack(
            (8.0 * np.cos(angles), 8.0 * np.sin(angles), np.zeros(samples))
        )
        vertices = [point.copy() for point in ring]
        vertices.append(np.asarray([0.0, 0.0, 6.0]))
        faces = [
            [index, (index + 1) % samples, samples]
            for index in range(samples)
        ]
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=5.0,
        )
        record = add_local_male_connector_and_backing(
            output_vertices=vertices,
            output_faces=faces,
            boundary_ids=list(range(samples)),
            boundary_points=ring,
            inward=np.asarray([0.0, 0.0, -1.0]),
            spec=spec,
        )
        layer_count = int(record["backing_profile_layer_count"])
        final_layer_start = samples + 1 + (layer_count - 1) * samples
        lead = np.asarray(vertices[final_layer_start : final_layer_start + samples])
        taper_depth = float(record["backing_taper_depth_mm"])
        outer_radius = np.linalg.norm(ring[:, :2], axis=1).mean()
        lead_radius = np.linalg.norm(lead[:, :2], axis=1).mean()

        # A sampled curved loop uses the exact polygon-edge miter; its radial
        # travel differs from the continuous-circle value by the tiny secant
        # factor of the sampling angle.
        self.assertAlmostEqual(
            outer_radius - lead_radius,
            taper_depth,
            delta=0.01,
        )
        self.assertAlmostEqual(abs(float(lead[:, 2].mean())), taper_depth, places=5)
        self.assertAlmostEqual(abs(float(lead[:, 2].mean())), 3.0, places=5)
        self.assertEqual(record["backing_profile"], "outer_large_inner_small")
        self.assertAlmostEqual(record["backing_taper_target_degrees"], 45.0)
        self.assertEqual(layer_count, 1)
        self.assertLessEqual(record["backing_strip_maximum_fanout"], 4)

    def test_production_male_defers_boolean_cutters_to_complete_child(self) -> None:
        samples = 64
        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        ring = np.column_stack(
            (8.0 * np.cos(angles), 8.0 * np.sin(angles), np.zeros(samples))
        )
        vertices = [point.copy() for point in ring]
        vertices.append(np.asarray([0.0, 0.0, 6.0]))
        faces = [
            [index, (index + 1) % samples, samples]
            for index in range(samples)
        ]
        cutters: list[common.trimesh.Trimesh] = []
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.60,
            bottom_clearance_mm=0.30,
            lead_in_mm=0.60,
            safe_engagement_depth_mm=8.25,
        )

        record = add_local_male_connector_and_backing(
            output_vertices=vertices,
            output_faces=faces,
            boundary_ids=list(range(samples)),
            boundary_points=ring,
            inward=np.asarray([0.0, 0.0, -1.0]),
            spec=spec,
            boolean_cutters=cutters,
        )

        # Complete child solids are the sole production Boolean cutters.
        # Do not rebuild private cutters which the recursive executor discards.
        self.assertEqual(len(cutters), 0)
        self.assertGreater(record['backing_faces_added'], 0)

    def test_production_parent_boolean_closure_cuts_real_socket(self) -> None:
        samples = 64
        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        top = np.column_stack(
            (8.0 * np.cos(angles), 8.0 * np.sin(angles), np.zeros(samples))
        )
        bottom = top.copy()
        bottom[:, 2] = -7.0
        vertices = [point.copy() for point in top] + [point.copy() for point in bottom]
        vertices.append(np.asarray([0.0, 0.0, -7.0]))
        faces = []
        for index in range(samples):
            nxt = (index + 1) % samples
            faces.extend(
                ([index, samples + nxt, nxt], [index, samples + index, samples + nxt])
            )
            faces.append([samples + index, 2 * samples, samples + nxt])
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.80,
            safe_engagement_depth_mm=5.0,
        )
        record, cutters = add_local_female_boolean_closure(
            output_vertices=vertices,
            output_faces=faces,
            boundary_ids=list(range(samples)),
            boundary_points=top,
            inward=np.asarray([0.0, 0.0, -1.0]),
            spec=spec,
            occupied_parent_edges={(0, 2)},
        )
        closed_parent = common.trimesh.Trimesh(
            vertices=np.asarray(vertices),
            faces=np.asarray(faces),
            process=True,
        )
        socketed, boolean_record = subtract_socket_cutters(closed_parent, cutters)
        preserved = finalize_boolean_difference_mesh(socketed)

        child_vertices = [point.copy() for point in top]
        child_vertices.append(np.asarray([0.0, 0.0, 5.5]))
        child_faces = [
            [index, (index + 1) % samples, samples]
            for index in range(samples)
        ]
        add_local_male_connector_and_backing(
            output_vertices=child_vertices,
            output_faces=child_faces,
            boundary_ids=list(range(samples)),
            boundary_points=top,
            inward=np.asarray([0.0, 0.0, -1.0]),
            spec=spec,
        )
        child = common.trimesh.Trimesh(
            vertices=np.asarray(child_vertices),
            faces=np.asarray(child_faces),
            process=True,
        )
        common.trimesh.repair.fix_winding(child)
        common.trimesh.repair.fix_normals(child)
        overlap = common.trimesh.boolean.intersection(
            [preserved, child],
            engine="manifold",
            check_volume=True,
        )
        overlap_volume = 0.0 if overlap is None else abs(float(overlap.volume))

        self.assertTrue(closed_parent.is_watertight)
        self.assertTrue(socketed.is_watertight)
        self.assertTrue(preserved.is_watertight)
        self.assertTrue(child.is_watertight)
        self.assertTrue(preserved.is_winding_consistent)
        self.assertEqual(len(preserved.faces), len(socketed.faces))
        self.assertAlmostEqual(abs(float(preserved.volume)), abs(float(socketed.volume)), places=6)
        self.assertEqual(
            preserved.metadata["boolean_finalize_policy"],
            "preserve_inherited_source_slivers_without_hole_filling",
        )
        self.assertEqual(record["connector_role"], "female_boolean")
        self.assertEqual(
            record["pre_boolean_closure_forbidden_internal_edge_count"],
            1,
        )
        cap_faces = np.asarray(faces[-(samples - 2) :], dtype=np.int64)
        cap_edges = {
            tuple(sorted(edge))
            for face in cap_faces
            for edge in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            )
        }
        self.assertNotIn((0, 2), cap_edges)
        self.assertEqual(
            record["boolean_cutter_scope"],
            "sequential_complete_male_backing_then_compact_socket",
        )
        self.assertEqual(
            boolean_record["boolean_strategy"],
            "sequential_difference_without_cutter_union",
        )
        self.assertEqual(boolean_record["applied_socket_cutter_count"], 2)
        self.assertGreater(boolean_record["intersection_volume_mm3"], 1.0)

        self.assertLess(overlap_volume, 1e-6)
        triangles = np.asarray(preserved.vertices)[np.asarray(preserved.faces)]
        double_areas = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        self.assertGreater(float(double_areas.min()), 1e-12)
        section_profiles = []
        for section_depth in (0.25, 0.75, 1.5, 2.5, 3.25, 4.0, 4.75):
            section = preserved.section(
                plane_origin=np.asarray([0.0, 0.0, -section_depth]),
                plane_normal=np.asarray([0.0, 0.0, 1.0]),
            )
            loop_areas = []
            if section is not None:
                for loop in section.discrete:
                    points = np.asarray(loop, dtype=np.float64)[:, :2]
                    loop_areas.append(
                        0.5
                        * abs(
                            float(
                                np.dot(points[:, 0], np.roll(points[:, 1], -1))
                                - np.dot(points[:, 1], np.roll(points[:, 0], -1))
                            )
                        )
                    )
            loop_areas = sorted(loop_areas, reverse=True)
            self.assertEqual(
                len(loop_areas),
                2,
                msg=f"depth {section_depth} must contain one body loop and one cavity loop",
            )
            section_profiles.append((section_depth, loop_areas))

        outer_areas = [profile[1][0] for profile in section_profiles]
        cavity_areas = [profile[1][1] for profile in section_profiles]
        self.assertLess(max(outer_areas) - min(outer_areas), 1e-6)
        self.assertTrue(
            all(
                cavity_areas[index] > cavity_areas[index + 1]
                for index in range(4)
            )
        )
        for depth, (_, cavity_area) in section_profiles[:4]:
            equivalent_radius = float(np.sqrt(cavity_area / np.pi))
            self.assertAlmostEqual(equivalent_radius, 8.0 - depth, delta=0.02)
        self.assertAlmostEqual(cavity_areas[-2], cavity_areas[-1], places=6)

    def test_deferred_parent_closure_skips_discarded_cutter_construction(self) -> None:
        boundary = np.asarray(
            [
                [-8.0, -6.0, 0.0],
                [8.0, -6.0, 0.0],
                [8.0, 6.0, 0.0],
                [-8.0, 6.0, 0.0],
            ],
            dtype=np.float64,
        )
        vertices = [point.copy() for point in boundary]
        faces: list[list[int]] = []

        with mock.patch.object(
            inward_module,
            "build_socket_cutter_from_plan",
            side_effect=AssertionError("deferred compact cutter was built"),
        ), mock.patch.object(
            inward_module,
            "build_local_male_attachment_cutter",
            side_effect=AssertionError("deferred male cutter was built"),
        ):
            record, cutters = add_local_female_boolean_closure(
                output_vertices=vertices,
                output_faces=faces,
                boundary_ids=list(range(len(boundary))),
                boundary_points=boundary,
                inward=np.asarray([0.0, 0.0, -1.0]),
                spec=self.spec,
                build_boolean_cutters=False,
            )

        self.assertEqual(cutters, [])
        self.assertGreater(record["pre_boolean_closure_faces_added"], 0)
        self.assertFalse(record["boolean_cutters_built"])
        self.assertEqual(record["cavity_cutter_count"], 0)
        self.assertEqual(record["planned_cavity_cutter_count"], 2)

    def test_local_interfaces_close_open_source_patches_without_full_loop_walls(self) -> None:
        samples = 64
        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        ring = np.column_stack(
            (7.0 * np.cos(angles), 7.0 * np.sin(angles), np.zeros(samples))
        )

        child_vertices = [point.copy() for point in ring]
        child_vertices.append(np.asarray([0.0, 0.0, 5.5]))
        child_faces = [
            [index, (index + 1) % samples, samples]
            for index in range(samples)
        ]
        child_record = add_local_male_connector_and_backing(
            output_vertices=child_vertices,
            output_faces=child_faces,
            boundary_ids=list(range(samples)),
            boundary_points=ring,
            inward=np.asarray([0.0, 0.0, -1.0]),
            spec=self.spec,
        )
        child = common.trimesh.Trimesh(
            vertices=np.asarray(child_vertices),
            faces=np.asarray(child_faces),
            process=False,
        )

        lower_ring = ring.copy()
        lower_ring[:, 2] = -7.0
        parent_vertices = [point.copy() for point in ring] + [point.copy() for point in lower_ring]
        parent_vertices.append(np.asarray([0.0, 0.0, -7.0]))
        parent_faces = []
        for index in range(samples):
            nxt = (index + 1) % samples
            parent_faces.extend(
                ([index, samples + nxt, nxt], [index, samples + index, samples + nxt])
            )
            parent_faces.append([samples + index, 2 * samples, samples + nxt])
        parent_record = add_local_female_socket_and_backing(
            output_vertices=parent_vertices,
            output_faces=parent_faces,
            boundary_ids=list(range(samples)),
            boundary_points=ring,
            inward=np.asarray([0.0, 0.0, -1.0]),
            spec=self.spec,
        )
        parent = common.trimesh.Trimesh(
            vertices=np.asarray(parent_vertices),
            faces=np.asarray(parent_faces),
            process=False,
        )

        self.assertTrue(child.is_watertight)
        self.assertTrue(parent.is_watertight)
        self.assertEqual(child_record["connector_role"], "male")
        self.assertEqual(parent_record["connector_role"], "female")
        self.assertLess(child_record["footprint_area_ratio"], 0.15)


if __name__ == "__main__":
    unittest.main()
