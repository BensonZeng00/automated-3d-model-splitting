from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common

common.load_core_dependencies()

from split3mf.hidden_interface import (  # noqa: E402
    HIDDEN_INTERFACE_MINIMUM_LOAD_BEARING_DEPTH_MM,
    HiddenInterfacePlanner,
)


class DirectionSensitiveProbe:
    def __init__(self) -> None:
        self.triangles = np.asarray(
            [
                [[-2.0, 4.0, 4.0], [2.0, 4.0, 4.0], [0.0, 6.0, 5.0]],
                [[-2.0, 4.0, 6.0], [0.0, 6.0, 5.0], [2.0, 4.0, 6.0]],
            ],
            dtype=np.float64,
        )
        self.active_triangle_mask = np.ones(len(self.triangles), dtype=bool)

    def safety_limit(self, points, directions, global_ceiling_mm):
        mean_direction = np.asarray(directions, dtype=np.float64).mean(axis=0)
        safe_depth = min(
            float(global_ceiling_mm),
            max(0.0, float(mean_direction[1])) * 4.0,
        )
        return safe_depth, {
            "safe_maximum_inward_depth_mm": safe_depth,
            "parent_thickness_min_mm": safe_depth + 0.05,
        }


class HiddenInterfacePlannerTests(unittest.TestCase):
    def test_parent_interior_search_improves_grazing_baseline(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        points = np.column_stack(
            (np.cos(angles), np.zeros_like(angles), np.sin(angles))
        )
        initial = np.tile(np.asarray([0.0, -0.2, 0.98]), (len(points), 1))
        local = np.tile(np.asarray([0.0, 0.7, 0.7]), (len(points), 1))
        plan = HiddenInterfacePlanner.plan(
            points=points,
            initial_directions=initial,
            fallback_axis=np.asarray([0.0, 0.0, 1.0]),
            local_inward_normals=local,
            parent_thickness_probe=DirectionSensitiveProbe(),
            baseline_safe_depth_mm=0.008,
            preferred_depth_mm=3.0,
        )
        self.assertTrue(plan.candidates)
        selected = plan.candidates[0]
        self.assertGreaterEqual(
            selected.safe_depth_mm,
            HIDDEN_INTERFACE_MINIMUM_LOAD_BEARING_DEPTH_MM,
        )
        self.assertGreater(selected.minimum_local_inward_dot, 0.0)
        self.assertGreater(float(selected.directions[:, 1].mean()), 0.0)
        self.assertEqual(
            plan.record["hidden_interface_policy"],
            "deterministic_parent_interior_direction_search",
        )

    def test_candidate_order_is_deterministic(self) -> None:
        points = np.asarray(
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        initial = np.tile(np.asarray([0.0, -0.1, 0.99]), (3, 1))
        local = np.tile(np.asarray([0.0, 1.0, 1.0]), (3, 1))
        arguments = dict(
            points=points,
            initial_directions=initial,
            fallback_axis=np.asarray([0.0, 0.0, 1.0]),
            local_inward_normals=local,
            parent_thickness_probe=DirectionSensitiveProbe(),
            baseline_safe_depth_mm=0.01,
            preferred_depth_mm=3.0,
        )
        first = HiddenInterfacePlanner.plan(**arguments)
        second = HiddenInterfacePlanner.plan(**arguments)
        self.assertEqual(
            [candidate.name for candidate in first.candidates],
            [candidate.name for candidate in second.candidates],
        )


if __name__ == "__main__":
    unittest.main()


