from __future__ import annotations

from dataclasses import FrozenInstanceError
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from split3mf import common  # noqa: E402

common.load_core_dependencies()

from split3mf.connector_planning import (  # noqa: E402
    ConnectorDepthPolicy,
    LocalConnectorPlanningService,
)
from split3mf.connector_topology import (  # noqa: E402
    ConnectorTopologyService,
    _audit_ring_strip,
    _balanced_ring_strip_faces,
)
from tools.run_connector_harness import (  # noqa: E402
    run_case,
    run_mismatched_ripple_strip_case,
)


class ConnectorArchitectureTests(unittest.TestCase):
    def test_depth_policy_is_immutable_and_service_reuses_it(self) -> None:
        policy = ConnectorDepthPolicy()
        with self.assertRaises(FrozenInstanceError):
            policy.maximum_total_depth_mm = 7.0  # type: ignore[misc]

        spec = LocalConnectorPlanningService.spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.60,
            safe_engagement_depth_mm=5.0,
            policy=policy,
        )
        self.assertAlmostEqual(spec.full_boundary_backing_depth_mm, 3.0)
        self.assertAlmostEqual(
            spec.engagement_depth_mm
            + spec.full_boundary_backing_depth_mm
            + spec.socket_bottom_clearance_mm,
            5.0,
        )

    def test_depth_service_keeps_rim_and_footprint_budgets_separate(self) -> None:
        spec = LocalConnectorPlanningService.spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.60,
            safe_engagement_depth_mm=5.0,
            safe_backing_depth_mm=1.10,
        )

        self.assertTrue(spec.elastic_shrink_applied)
        self.assertAlmostEqual(spec.full_boundary_backing_depth_mm, 1.10)
        self.assertAlmostEqual(spec.engagement_depth_mm, 3.65)
        self.assertLess(spec.elastic_lateral_scale, 1.0)
        self.assertAlmostEqual(spec.total_safety_limit_mm, 5.0)

    def test_depth_service_emits_full_three_plus_five_when_wall_allows(self) -> None:
        spec = LocalConnectorPlanningService.spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.60,
            safe_engagement_depth_mm=8.25,
            safe_backing_depth_mm=8.25,
        )

        self.assertAlmostEqual(spec.full_boundary_backing_depth_mm, 3.0)
        self.assertAlmostEqual(spec.engagement_depth_mm, 5.0)
        self.assertFalse(spec.elastic_shrink_applied)
        self.assertTrue(spec.compact_peg_enabled)

    def test_depth_service_shrinks_engagement_to_zero_before_backing(self) -> None:
        spec = LocalConnectorPlanningService.spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.60,
            safe_engagement_depth_mm=3.25,
            safe_backing_depth_mm=5.0,
        )

        self.assertAlmostEqual(spec.full_boundary_backing_depth_mm, 3.0)
        self.assertAlmostEqual(spec.engagement_depth_mm, 0.0)
        self.assertFalse(spec.compact_peg_enabled)
        self.assertTrue(spec.engagement_elastic_shrink_applied)
        self.assertFalse(spec.backing_elastic_shrink_applied)

    def test_depth_service_shrinks_backing_only_after_zero_engagement(self) -> None:
        spec = LocalConnectorPlanningService.spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.60,
            safe_engagement_depth_mm=2.0,
            safe_backing_depth_mm=5.0,
        )

        self.assertAlmostEqual(spec.full_boundary_backing_depth_mm, 2.0)
        self.assertAlmostEqual(spec.engagement_depth_mm, 0.0)
        self.assertAlmostEqual(spec.socket_bottom_clearance_mm, 0.0)
        self.assertFalse(spec.compact_peg_enabled)
        self.assertTrue(spec.engagement_elastic_shrink_applied)
        self.assertTrue(spec.backing_elastic_shrink_applied)

    def test_annulus_service_preserves_collinear_boundary_edges(self) -> None:
        outer = np.asarray(
            [
                [-5.0, -5.0],
                [0.0, -5.0],
                [5.0, -5.0],
                [5.0, 5.0],
                [0.0, 5.0],
                [-5.0, 5.0],
            ],
            dtype=np.float64,
        )
        hole = np.asarray(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
            dtype=np.float64,
        )
        outer_ids = list(range(len(outer)))
        hole_ids = list(range(len(outer), len(outer) + len(hole)))
        faces, audit = ConnectorTopologyService.triangulate_annulus(
            outer_ids,
            outer,
            hole_ids,
            hole,
        )
        self.assertTrue(audit.valid, audit.reason)
        self.assertEqual(audit.degenerate_face_count, 0)
        self.assertEqual(audit.over_shared_edge_count, 0)
        self.assertEqual(len(faces), len(outer) + len(hole))

    def test_concave_v_harness_uses_production_modules(self) -> None:
        result = run_case("concave-v")
        self.assertTrue(result.passed, result.error)
        self.assertAlmostEqual(result.backing_depth_mm, 3.0)
        self.assertEqual(result.connector_edges_outside_backing, 0)
        self.assertTrue(result.topology["valid"])
        self.assertEqual(
            result.plan["production_backing_profile_layer_count"], 1
        )
        self.assertAlmostEqual(
            result.plan["production_backing_taper_measured_median_degrees"],
            45.0,
            delta=2.0,
        )

    def test_mismatched_ripple_strip_has_bounded_fanout(self) -> None:
        result = run_mismatched_ripple_strip_case()
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["outer_vertices"], 996)
        self.assertEqual(result["inner_vertices"], 433)
        self.assertAlmostEqual(result["source_axial_spread_mm"], 1.65, places=3)
        self.assertLessEqual(result["audit"]["maximum_fanout"], 4)
        self.assertLess(result["audit"]["maximum_cross_edge_mm"], 2.0)

    def test_ring_strip_rejects_even_one_bridge_outside_annulus(self) -> None:
        outer = np.asarray(
            [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]]
        )
        inner = np.asarray(
            [[-1.0, -1.0, -0.5], [1.0, -1.0, -0.5], [1.0, 1.0, -0.5], [-1.0, 1.0, -0.5]]
        )
        inner = np.roll(inner, 2, axis=0)
        outer_ids = list(range(4))
        inner_ids = [6, 7, 4, 5]
        faces = _balanced_ring_strip_faces(
            outer_ids,
            outer,
            inner_ids,
            inner,
            maximum_consecutive_advances=3,
        )
        points = {
            **{index: point for index, point in zip(outer_ids, outer)},
            **{index: point for index, point in zip(inner_ids, inner)},
        }
        projected = {index: point[:2] for index, point in points.items()}
        audit = _audit_ring_strip(
            faces,
            points,
            projected,
            outer_ids,
            inner_ids,
            outer[:, :2],
            inner[:, :2],
            maximum_fanout=4,
        )

        self.assertEqual(audit.invalid_bridge_count, 1)
        self.assertFalse(audit.valid)
        self.assertIn("bridge_leaves_annulus", audit.reason)


if __name__ == "__main__":
    unittest.main()
