from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common

common.load_core_dependencies()

from split3mf.guided_internal_cut import (  # noqa: E402
    GuidedInternalCutPlanner,
    GuidedInternalCutSpec,
    adaptive_guided_entry_ring,
)
from split3mf.inward import cap_side_wall_quality  # noqa: E402


class LateralSensitiveProbe:
    def __init__(self) -> None:
        self.triangles = np.asarray(
            [
                [[-2.0, -2.0, 5.0], [2.0, -2.0, 5.0], [2.0, 2.0, 5.0]],
                [[-2.0, -2.0, 5.0], [2.0, 2.0, 5.0], [-2.0, 2.0, 5.0]],
            ],
            dtype=np.float64,
        )
        self.active_triangle_mask = np.ones(2, dtype=bool)

    def first_hit_distances(self, points, directions, global_ceiling_mm):
        lateral = np.abs(np.asarray(directions, dtype=np.float64)[:, 0])
        return np.where(lateral >= 0.08, float(global_ceiling_mm), 0.30)

    def safety_limit(self, points, directions, global_ceiling_mm):
        raw = self.first_hit_distances(points, directions, global_ceiling_mm + 0.05)
        safe = min(float(global_ceiling_mm), float(np.min(raw)) - 0.05)
        return max(safe, 0.0), {"safe_maximum_inward_depth_mm": max(safe, 0.0)}


class LocalizedThinProbe(LateralSensitiveProbe):
    def first_hit_distances(self, points, directions, global_ceiling_mm):
        points = np.asarray(points, dtype=np.float64)
        thin = (points[:, 0] > 0.85) & (points[:, 1] > -0.10)
        return np.where(thin, 0.30, float(global_ceiling_mm))


class GuidedInternalCutTests(unittest.TestCase):
    def test_lateral_search_preserves_marked_plane_and_avoids_thin_shell(self) -> None:
        points = np.asarray(
            [
                [-1.0, -0.5, 0.0],
                [1.0, -0.5, 0.0],
                [1.0, 0.5, 0.0],
                [-1.0, 0.5, 0.0],
            ],
            dtype=np.float64,
        )
        local = np.asarray(
            [
                [0.7, 0.0, 0.7],
                [-0.7, 0.0, 0.7],
                [-0.7, 0.0, 0.7],
                [0.7, 0.0, 0.7],
            ],
            dtype=np.float64,
        )
        spec = GuidedInternalCutSpec.from_mapping(
            {
                "entry_direction": [0.0, 0.18, 0.984],
                "target_plane_normal": [0.0, -0.20, 1.0],
                "target_plane_point_mm": [0.0, 0.0, 3.0],
                "minimum_depth_mm": 2.5,
                "maximum_depth_mm": 4.0,
            }
        )
        plan = GuidedInternalCutPlanner.plan(
            points=points,
            local_inward_normals=local,
            parent_thickness_probe=LateralSensitiveProbe(),
            spec=spec,
        )
        residual = np.abs(
            (plan.bottom_points - spec.target_plane_point_mm) @ spec.target_plane_normal
        )
        self.assertLess(float(residual.max()), 1e-9)
        self.assertNotEqual(
            plan.record["guided_internal_cut_selected_candidate"],
            "uniform_section_entry",
        )
        self.assertGreaterEqual(float(plan.distances.min()), 2.5)
        self.assertLessEqual(float(plan.distances.max()), 4.0)

    def test_dense_preflight_rejects_long_circumferential_spike(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 600, endpoint=False)
        top = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
        bottom = top.copy()
        bottom[:, 2] = 3.0
        bottom[241, 0] += 12.0
        quality = cap_side_wall_quality([top, bottom])
        self.assertFalse(quality["valid"])
        self.assertGreater(quality["long_circumferential_edges"], 0)

    def test_entry_inset_is_localized_to_thin_arc(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
        boundary = np.column_stack(
            (np.cos(angles), np.sin(angles), np.zeros_like(angles))
        )
        conormals = -boundary.copy()
        spec = GuidedInternalCutSpec.from_mapping(
            {
                "entry_direction": [0.0, 0.18, 0.984],
                "target_plane_normal": [0.0, -0.20, 1.0],
                "target_plane_point_mm": [0.0, 0.0, 3.0],
                "minimum_depth_mm": 1.0,
                "maximum_depth_mm": 8.0,
                "entry_inset_mm": 0.8,
            }
        )
        fit, insets, record = adaptive_guided_entry_ring(
            boundary_points=boundary,
            interior_conormals=conormals,
            parent_thickness_probe=LocalizedThinProbe(),
            spec=spec,
            base_inset_mm=0.1,
        )
        self.assertEqual(record["guided_entry_inset_policy"], "localized_parent_thickness_avoidance")
        self.assertGreater(float(insets.max()), 0.79)
        self.assertLess(float(insets.min()), 0.11)
        np.testing.assert_allclose(
            np.linalg.norm(fit - boundary, axis=1),
            insets,
            atol=1e-9,
        )

    def test_spec_rejects_near_tangent_entry(self) -> None:
        with self.assertRaisesRegex(ValueError, "nearly parallel"):
            GuidedInternalCutSpec.from_mapping(
                {
                    "entry_direction": [1.0, 0.0, 0.0],
                    "target_plane_normal": [0.0, 0.0, 1.0],
                    "target_plane_point_mm": [0.0, 0.0, 1.0],
                }
            )


if __name__ == "__main__":
    unittest.main()
