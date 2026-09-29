from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split3mf import common

common.load_core_dependencies()

from split3mf.application.recognized_boundaries import RecognizedBoundaries
from split3mf.application.boundary_snapshot_builder import _filter_unpaired_component_loops
from split3mf.common import Component
from split3mf.contact_interface_planner import plan_contact_interfaces
from split3mf.boundary_matching import matching_seam


class ContactInterfacePlannerTests(unittest.TestCase):
    def _plan(self, separation: float, areas=(3.0, 4.0), **options) -> dict:
        angles_a = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        angles_b = np.linspace(0.03, 2 * np.pi + 0.03, 26, endpoint=False)
        first = np.column_stack((np.cos(angles_a), np.sin(angles_a), np.zeros(len(angles_a))))
        second = np.column_stack((np.cos(angles_b) + separation, np.sin(angles_b), np.zeros(len(angles_b))))
        boundaries = RecognizedBoundaries(
            component_loops=((tuple(range(30)),), (tuple(range(100, 126)),)),
            component_loop_points=((tuple(map(tuple, first)),), (tuple(map(tuple, second)),)),
            source_vertex_count=126, source_face_count=0, fingerprint="test",
        )
        components = [
            Component("A", np.array([], dtype=np.int64), 0, areas[0],
                      first.min(axis=0), first.max(axis=0), np.array([0.0, 0.0, 0.0])),
            Component("B", np.array([], dtype=np.int64), 0, areas[1],
                      second.min(axis=0), second.max(axis=0), np.array([0.0, 0.0, 1.0])),
        ]
        with patch(
            "split3mf.contact_interface_planner._component_inward_direction",
            return_value=np.array([0.0, 0.0, 1.0]),
        ):
            return plan_contact_interfaces(
                np.zeros((1, 3)), np.zeros((0, 3), dtype=np.int64),
                components, boundaries, np.zeros(3), **options,
            )

    def test_large_area_difference_overrides_stronger_normal(self):
        for areas in ((12.0, 4.0), (120.0, 4.0), (1.2, 0.4)):
            relation = self._plan(0.0, areas=areas)["interfaces"][0]
            self.assertEqual(relation["tenon_part"], "P02")
            self.assertEqual(relation["direction_evidence"]["method"], "large_area_difference")

    def test_area_threshold_preserves_normal_below_cutoff(self):
        relation = self._plan(0.0, areas=(11.99, 4.0))["interfaces"][0]
        self.assertEqual(relation["tenon_part"], "P01")
        self.assertEqual(relation["direction_evidence"]["method"], "inward_alignment")
        relation = self._plan(0.0, areas=(12.0, 4.0), area_priority_ratio=4.0)["interfaces"][0]
        self.assertEqual(relation["tenon_part"], "P01")

    def test_explicit_area_policy_still_has_priority(self):
        relation = self._plan(0.0, areas=(8.0, 4.0),
                              direction_policy_overrides={"I001": "smaller_area_first"})["interfaces"][0]
        self.assertEqual(relation["tenon_part"], "P02")
        self.assertEqual(relation["direction_evidence"]["method"], "user_requested_smaller_area_first")

    def test_invalid_area_threshold_rejected(self):
        for value in (0, 1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self._plan(0.0, area_priority_ratio=value)

    def test_independent_simplified_samples_form_one_interface(self):
        plan = self._plan(0.0)
        self.assertEqual(plan["interface_count"], 1)
        self.assertEqual(len(plan["interfaces"][0]["contact"]["shared_boundary_loops"]), 1)

    def test_insertion_uses_contact_normal_after_role_selection(self):
        relation = self._plan(0.0, areas=(12.0, 4.0))["interfaces"][0]
        self.assertEqual(relation["tenon_part"], "P02")
        self.assertEqual(
            relation["direction_evidence"]["insertion_axis_source"],
            "contact_boundary_area_normal",
        )
        np.testing.assert_allclose(relation["insertion_direction"], [0, 0, -1])
        np.testing.assert_allclose(relation["mating_axis_toward_mortise"], [0, 0, -1])

    def test_nearby_distinct_contours_do_not_form_interface(self):
        self.assertEqual(self._plan(0.3)["interface_count"], 0)

    def test_sparse_simplified_copies_match_at_model_scale(self):
        left_angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        right_angles = np.linspace(0.18, 2 * np.pi + 0.18, 10, endpoint=False)
        left = np.column_stack((5 * np.cos(left_angles), 5 * np.sin(left_angles), np.zeros(9)))
        right = np.column_stack((5 * np.cos(right_angles), 5 * np.sin(right_angles), np.zeros(10)))
        self.assertTrue(matching_seam(left, right))

    def test_unpaired_inner_loop_is_only_removed_from_logical_boundaries(self):
        angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        outer = tuple(map(tuple, np.column_stack((np.cos(angles), np.sin(angles), np.zeros(30)))))
        inner = tuple(map(tuple, np.column_stack((0.2 * np.cos(angles), 0.2 * np.sin(angles), np.zeros(30)))))
        standalone = tuple(map(tuple, np.column_stack((np.cos(angles) + 4, np.sin(angles), np.zeros(30)))))
        records = [
            {"component_index": 1, "loop_index": 1, "status": "simplified"},
            {"component_index": 1, "loop_index": 2, "status": "simplified"},
        ]
        loops, points = _filter_unpaired_component_loops(
            [(tuple(range(30)), tuple(range(30, 60))), (tuple(range(60, 90)),), (tuple(range(90, 120)),)],
            [(outer, inner), (outer,), (standalone,)],
            [1, 2, 3], records,
        )
        self.assertEqual([len(value) for value in loops], [1, 1, 1])
        self.assertEqual([len(value) for value in points], [1, 1, 1])
        self.assertEqual(records[1]["recognition_status"], "unpaired_boundary_ignored")


if __name__ == "__main__":
    unittest.main()
