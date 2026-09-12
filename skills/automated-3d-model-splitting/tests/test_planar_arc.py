from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf.planar_arc import (
    PlanarArcError,
    build_planar_arc_boundary,
)
from split3mf import common

common.load_core_dependencies()

from split3mf.mesh import triangulate_polygon_ear_clip


def noisy_ellipse(samples: int = 240) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    noise = 0.10 * np.sin(29.0 * angle) + 0.04 * np.sin(53.0 * angle)
    return np.column_stack(
        (
            (8.0 + noise) * np.cos(angle),
            (5.5 + noise) * np.sin(angle),
            0.08 * np.sin(3.0 * angle),
        )
    )


class PlanarArcTests(unittest.TestCase):
    def test_three_vertex_loop_is_preserved_without_inventing_control_point(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.1],
                [0.5, 1.0, -0.1],
            ],
            dtype=np.float64,
        )
        result = build_planar_arc_boundary(
            points,
            np.asarray([30, 10, 20], dtype=np.int64),
        )

        np.testing.assert_allclose(result.target_points, points, atol=0.0)
        self.assertEqual(result.record["status"], "exact-minimal-loop")
        self.assertTrue(result.record["minimal_loop_preserved"])
        self.assertEqual(len(result.guide_points), 384)

    def test_confirmed_384_sample_28_pass_profile_reduces_ripple(self) -> None:
        points = noisy_ellipse()
        result = build_planar_arc_boundary(
            points,
            np.arange(len(points), dtype=np.int64),
            target_samples=384,
            smooth_passes=28,
            maximum_target_offset_mm=0.75,
        )

        self.assertEqual(len(result.guide_points), 384)
        self.assertEqual(result.record["mode"], "planar-arc-retopology")
        self.assertLess(
            result.record["projected_ripple_after_mm"],
            result.record["projected_ripple_before_mm"],
        )
        self.assertLessEqual(result.record["maximum_target_offset_mm"], 0.75)
        self.assertFalse(result.record["projected_extents_restored"])
        self.assertEqual(
            result.record["guide_model"],
            "least-squares-periodic-cubic-b-spline",
        )
        self.assertEqual(
            result.record["guide_sampling_parameterization"],
            "equal-source-arc-phase",
        )
        self.assertEqual(
            result.record["guide_interpolation"],
            "direct-periodic-cubic-b-spline",
        )

    def test_source_id_canonicalization_is_orientation_independent(self) -> None:
        points = noisy_ellipse(96)
        ids = np.arange(1000, 1096, dtype=np.int64)
        forward = build_planar_arc_boundary(points, ids)
        reverse_order = np.concatenate(([0], np.arange(len(points) - 1, 0, -1)))
        reverse = build_planar_arc_boundary(points[reverse_order], ids[reverse_order])
        reverse_by_id = {
            int(source_id): point
            for source_id, point in zip(ids[reverse_order], reverse.target_points)
        }
        reordered = np.asarray([reverse_by_id[int(source_id)] for source_id in ids])
        np.testing.assert_allclose(forward.target_points, reordered, atol=1e-9)

    def test_isolated_short_rim_edge_is_locally_redistributed(self) -> None:
        angle = 2.0 * np.pi * np.arange(160, dtype=np.float64) / 160.0
        angle[40] = angle[39] + 0.01 * (2.0 * np.pi / 160.0)
        points = np.column_stack(
            (8.0 * np.cos(angle), 5.5 * np.sin(angle), np.zeros_like(angle))
        )
        result = build_planar_arc_boundary(
            points,
            np.arange(len(points), dtype=np.int64),
            maximum_target_offset_mm=0.75,
        )

        source_edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        target_edges = np.linalg.norm(
            np.roll(result.target_points, -1, axis=0) - result.target_points,
            axis=1,
        )
        self.assertGreater(float(target_edges.min()), 5.0 * float(source_edges.min()))
        self.assertLess(float(np.std(target_edges)), float(np.std(source_edges)))
        self.assertEqual(
            result.record["boundary_vertex_distribution"],
            "locally-regularized-physical-arc",
        )
        self.assertFalse(result.record["source_arc_parameters_preserved"])

    def test_out_of_plane_boundary_ripple_is_smoothed_in_three_dimensions(self) -> None:
        samples = 192
        angle = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        points = np.column_stack(
            (
                8.0 * np.cos(angle),
                5.5 * np.sin(angle),
                0.30 * np.sin(37.0 * angle) + 0.04 * np.sin(3.0 * angle),
            )
        )
        result = build_planar_arc_boundary(
            points,
            np.arange(samples, dtype=np.int64),
            maximum_target_offset_mm=1.0,
        )

        self.assertEqual(
            result.record["surface_restore_policy"],
            "stable-plane-full-3d-smoothed-height",
        )
        self.assertLess(
            result.record["normal_height_ripple_after_mm"],
            0.70 * result.record["normal_height_ripple_before_mm"],
        )
        source_height = (points - result.plane_origin) @ result.plane_normal
        target_height = (
            result.target_points - result.plane_origin
        ) @ result.plane_normal
        self.assertLess(float(np.std(target_height)), 0.65 * float(np.std(source_height)))

    def test_dense_loop_uses_low_order_height_spline(self) -> None:
        samples = 405
        angle = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        points = np.column_stack(
            (
                8.0 * np.cos(angle),
                5.5 * np.sin(angle),
                0.12 * np.sin(73.0 * angle),
            )
        )
        result = build_planar_arc_boundary(
            points,
            np.arange(samples, dtype=np.int64),
            maximum_target_offset_mm=0.75,
        )

        self.assertEqual(result.record["planar_control_points"], 24)
        self.assertEqual(result.record["height_control_points"], 8)
        self.assertEqual(result.record["arc_parameter_smooth_passes"], 2)
        self.assertFalse(result.record["source_arc_parameters_preserved"])
        self.assertLess(
            result.record["normal_height_ripple_after_mm"],
            result.record["normal_height_ripple_before_mm"],
        )

    def test_dense_low_frequency_height_waves_are_not_restored(self) -> None:
        samples = 878
        angle = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        points = np.column_stack(
            (
                18.0 * np.cos(angle),
                14.0 * np.sin(angle),
                0.20 * np.sin(18.0 * angle),
            )
        )
        result = build_planar_arc_boundary(
            points,
            np.arange(samples, dtype=np.int64),
            maximum_target_offset_mm=0.75,
        )

        source_height = (points - result.plane_origin) @ result.plane_normal
        target_height = (
            result.target_points - result.plane_origin
        ) @ result.plane_normal
        self.assertLess(float(np.std(target_height)), 0.10 * float(np.std(source_height)))
        self.assertFalse(result.record["projected_extents_restored"])

    def test_target_outside_retopology_band_blocks_instead_of_backing_off(self) -> None:
        points = noisy_ellipse(96)
        points[8, :2] += np.asarray([2.5, -1.5])
        with self.assertRaisesRegex(PlanarArcError, "leaves its retopology band"):
            build_planar_arc_boundary(
                points,
                np.arange(len(points), dtype=np.int64),
                maximum_target_offset_mm=0.54,
            )

    def test_near_straight_samples_remain_nondegenerate_after_3mf_precision_weld(self) -> None:
        side_samples = 24
        t = np.arange(side_samples, dtype=np.float64) / side_samples
        wobble = 5e-9 * np.sin(np.pi * t)
        points = np.vstack(
            (
                np.column_stack((-6.0 + 12.0 * t, -6.0 + wobble)),
                np.column_stack((6.0 - wobble, -6.0 + 12.0 * t)),
                np.column_stack((6.0 - 12.0 * t, 6.0 - wobble)),
                np.column_stack((-6.0 + wobble, 6.0 - 12.0 * t)),
            )
        )

        triangles = triangulate_polygon_ear_clip(points)
        self.assertEqual(len(triangles), len(points) - 2)
        welded = np.round(points, 6)
        triangle_points = welded[np.asarray(triangles, dtype=np.int64)]
        first = triangle_points[:, 1] - triangle_points[:, 0]
        second = triangle_points[:, 2] - triangle_points[:, 0]
        double_areas = np.abs(
            first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        )
        self.assertGreater(float(double_areas.min()), 1e-12)


if __name__ == "__main__":
    unittest.main()
