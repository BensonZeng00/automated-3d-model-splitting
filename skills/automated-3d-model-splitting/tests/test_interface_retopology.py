from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common

common.load_core_dependencies()

from split3mf.cli import build_parser
from split3mf.domain import PlanarArcRetopologyConfig, PlanarArcRetopologyContext
from split3mf.debug_export import export_retopology_failure_diagnostics
from split3mf.planar_arc import PlanarArcError
from split3mf.interface_retopology import (
    InterfaceRetopologyService,
    _directed_edge_topology_issues,
    _repair_flipped_boundary_ears,
    _locally_inverted_face_mask,
    _surface_band_deformation,
    _triangle_shape_quality,
    _untangle_interior_surface_vertices,
    _visual_displacement_advisory,
)
from split3mf.surface_quality import sparse_local_inversion_audit


class InterfaceRetopologyTests(unittest.TestCase):
    def test_visual_advisory_uses_physical_displacement_not_coverage(self) -> None:
        policy = PlanarArcRetopologyConfig().smoothing_policy
        broad_submillimeter = np.full(100_000, 0.4, dtype=np.float64)

        strict, _maximum, _p95 = _visual_displacement_advisory(
            "strict", broad_submillimeter, policy
        )
        advisory, maximum, p95 = _visual_displacement_advisory(
            "advisory", broad_submillimeter, policy
        )
        visible_drift, drift_maximum, _drift_p95 = _visual_displacement_advisory(
            "advisory", np.r_[broad_submillimeter, 5.975], policy
        )

        self.assertFalse(strict)
        self.assertTrue(advisory)
        self.assertAlmostEqual(maximum, 0.4)
        self.assertAlmostEqual(p95, 0.4)
        self.assertFalse(visible_drift)
        self.assertAlmostEqual(drift_maximum, 5.975)

    def test_user_reviewed_surface_band_lowers_only_sparse_angle_floor(self) -> None:
        face_count = 36402
        faces = np.arange(face_count * 3, dtype=np.int64).reshape((-1, 3))
        mask = np.zeros(face_count, dtype=bool)
        mask[[13, 29001]] = True
        result_angles = np.asarray([3.89583, 1.20601])

        strict = sparse_local_inversion_audit(
            faces,
            mask,
            audited_face_count=face_count,
            result_minimum_angles_degrees=result_angles,
        )
        advisory = sparse_local_inversion_audit(
            faces,
            mask,
            audited_face_count=face_count,
            result_minimum_angles_degrees=result_angles,
            minimum_result_angle_degrees=1.0,
        )

        self.assertFalse(strict["accepted"])
        self.assertTrue(advisory["accepted"])
        self.assertEqual(advisory["maximum_edge_connected_cluster"], 1)
        self.assertAlmostEqual(
            advisory["required_minimum_result_angle_degrees"], 1.0
        )

    def test_large_band_accepts_only_sparse_isolated_normal_outliers(self) -> None:
        face_count = 12000
        faces = np.arange(face_count * 3, dtype=np.int64).reshape((-1, 3))
        mask = np.zeros(face_count, dtype=bool)
        mask[[11, 3100, 9002]] = True

        audit = sparse_local_inversion_audit(
            faces,
            mask,
            audited_face_count=face_count,
            result_minimum_angles_degrees=np.asarray([8.0, 12.0, 16.0]),
        )

        self.assertTrue(audit["accepted"])
        self.assertEqual(audit["maximum_edge_connected_cluster"], 1)

    def test_small_mesh_keeps_single_local_inversion_blocking(self) -> None:
        faces = np.asarray([[0, 1, 2], [2, 1, 3]], dtype=np.int64)
        mask = np.asarray([True, False])

        audit = sparse_local_inversion_audit(
            faces,
            mask,
            audited_face_count=2,
            result_minimum_angles_degrees=np.asarray([20.0]),
        )

        self.assertFalse(audit["accepted"])

    def test_connected_local_inversion_cluster_remains_blocking(self) -> None:
        face_count = 12000
        faces = np.arange(face_count * 3, dtype=np.int64).reshape((-1, 3))
        faces[0] = np.asarray([0, 1, 2])
        faces[1] = np.asarray([1, 0, 3])
        faces[2] = np.asarray([0, 3, 4])
        mask = np.zeros(face_count, dtype=bool)
        mask[:3] = True

        audit = sparse_local_inversion_audit(
            faces,
            mask,
            audited_face_count=face_count,
            result_minimum_angles_degrees=np.asarray([10.0, 10.0, 10.0]),
        )

        self.assertFalse(audit["accepted"])
        self.assertEqual(audit["maximum_edge_connected_cluster"], 3)

    def test_vendor_sliver_normal_change_is_advisory(self) -> None:
        source = np.asarray(
            [
                [190.2488191, 151.8225431, 70.5516930],
                [190.3132655, 151.6680603, 70.6023998],
                [190.2724616, 151.7625198, 70.5673866],
            ],
            dtype=np.float64,
        )
        result = np.asarray(
            [
                [190.95301694, 151.48187744, 69.89210061],
                [190.90359837, 151.37742565, 69.97216320],
                [190.92110079, 151.47505549, 69.92040263],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        source_normal = np.cross(source[1] - source[0], source[2] - source[0])
        result_normal = np.cross(result[1] - result[0], result[2] - result[0])
        normal_cosine = float(
            np.dot(source_normal, result_normal)
            / (np.linalg.norm(source_normal) * np.linalg.norm(result_normal))
        )

        self.assertLess(normal_cosine, 0.0)
        self.assertLess(
            float(_triangle_shape_quality(source[faces])[0]),
            0.08,
        )
        self.assertEqual(
            int(np.count_nonzero(_locally_inverted_face_mask(source, result, faces))),
            0,
        )

    def test_directed_edge_audit_distinguishes_real_winding_error(self) -> None:
        consistent = np.asarray([[0, 1, 2], [1, 0, 3]], dtype=np.int64)
        inconsistent = np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
        over_shared = np.asarray(
            [[0, 1, 2], [1, 0, 3], [0, 1, 4]],
            dtype=np.int64,
        )

        self.assertEqual(_directed_edge_topology_issues(consistent), (set(), set()))
        self.assertEqual(
            _directed_edge_topology_issues(inconsistent),
            (set(), {(0, 1)}),
        )
        self.assertEqual(
            _directed_edge_topology_issues(over_shared),
            ({(0, 1)}, set()),
        )

    def test_interior_untangle_repairs_fold_without_moving_boundary(self) -> None:
        source = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
            dtype=np.int64,
        )
        folded = source.copy()
        folded[4] = np.asarray([0.5, -0.25, 0.0])
        self.assertGreater(
            int(np.count_nonzero(_locally_inverted_face_mask(source, folded, faces))),
            0,
        )

        repaired, repair_count, maximum_correction = (
            _untangle_interior_surface_vertices(
                source,
                folded,
                faces,
                np.arange(4, dtype=np.int64),
            )
        )

        self.assertGreater(repair_count, 0)
        self.assertGreater(maximum_correction, 0.0)
        np.testing.assert_allclose(repaired[:4], folded[:4], atol=0.0)
        self.assertEqual(
            int(np.count_nonzero(_locally_inverted_face_mask(source, repaired, faces))),
            0,
        )

    def test_flipped_all_boundary_ear_uses_safe_local_diagonal(self) -> None:
        source = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        result = source.copy()
        result[1] = np.asarray([0.2, 0.5, 0.0])
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

        repaired, repair_count = _repair_flipped_boundary_ears(
            source,
            result,
            faces,
            np.arange(4, dtype=np.int64),
        )

        self.assertEqual(repair_count, 1)
        self.assertFalse(np.array_equal(repaired, faces))
        source_triangles = source[repaired]
        result_triangles = result[repaired]
        source_normals = np.cross(
            source_triangles[:, 1] - source_triangles[:, 0],
            source_triangles[:, 2] - source_triangles[:, 0],
        )
        result_normals = np.cross(
            result_triangles[:, 1] - result_triangles[:, 0],
            result_triangles[:, 2] - result_triangles[:, 0],
        )
        self.assertTrue(np.all(np.einsum("ij,ij->i", source_normals, result_normals) > 0.0))

    def test_dense_independent_boundary_ears_converge_past_legacy_cap(self) -> None:
        source_patches = []
        result_patches = []
        face_patches = []
        for patch_index in range(300):
            offset = np.asarray([2.0 * patch_index, 0.0, 0.0])
            base = np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                 [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ) + offset
            folded = base.copy()
            folded[1] = offset + np.asarray([0.2, 0.5, 0.0])
            vertex_offset = 4 * patch_index
            source_patches.append(base)
            result_patches.append(folded)
            face_patches.append(
                np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
                + vertex_offset
            )
        source = np.vstack(source_patches)
        result = np.vstack(result_patches)
        faces = np.vstack(face_patches)

        started = time.perf_counter()
        repaired, repair_count = _repair_flipped_boundary_ears(
            source,
            result,
            faces,
            np.arange(len(source), dtype=np.int64),
        )
        elapsed = time.perf_counter() - started

        self.assertEqual(repair_count, 300)
        self.assertEqual(
            int(np.count_nonzero(_locally_inverted_face_mask(source, result, repaired))),
            0,
        )
        self.assertLess(elapsed, 5.0)

    def test_cli_has_only_planar_arc_parameters(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--input", "placeholder.3mf"])
        self.assertEqual(args.boundary_target_samples, 384)
        self.assertEqual(args.boundary_smooth_passes, 28)
        self.assertEqual(args.boundary_retopology_band_mm, 3.0)
        self.assertEqual(args.seam_smoothing_profile, "print-balanced")
        config = PlanarArcRetopologyConfig.from_namespace(args)
        self.assertEqual(config.smoothing_policy.profile, "print-balanced")
        self.assertEqual(config.smoothing_policy.maximum_displacement_mm, 0.5)
        self.assertEqual(config.smoothing_policy.maximum_affected_area_ratio, 0.01)
        self.assertEqual(config.smoothing_policy.maximum_topology_layers, 8)
        self.assertEqual(config.smoothing_policy.maximum_introduced_reversed_ratio, 0.001)

    def test_conservative_and_smooth_profiles_have_ordered_budgets(self) -> None:
        parser = build_parser()
        conservative = PlanarArcRetopologyConfig.from_namespace(parser.parse_args([
            "--input", "placeholder.3mf", "--seam-smoothing-profile", "source-conservative"])
        ).smoothing_policy
        smooth = PlanarArcRetopologyConfig.from_namespace(parser.parse_args([
            "--input", "placeholder.3mf", "--seam-smoothing-profile", "print-smooth"])
        ).smoothing_policy
        self.assertEqual(conservative.maximum_displacement_mm, smooth.maximum_displacement_mm)
        self.assertEqual(conservative.maximum_affected_area_ratio, smooth.maximum_affected_area_ratio)
        self.assertLess(
            conservative.maximum_introduced_reversed_ratio,
            smooth.maximum_introduced_reversed_ratio,
        )
        self.assertLess(conservative.maximum_edge_stretch_ratio, smooth.maximum_edge_stretch_ratio)

    def test_user_reviewed_surface_band_can_use_sixty_percent_of_real_band(self) -> None:
        parser = build_parser()
        strict = PlanarArcRetopologyConfig.from_namespace(
            parser.parse_args(["--input", "placeholder.3mf"])
        )
        advisory = PlanarArcRetopologyConfig.from_namespace(
            parser.parse_args(
                [
                    "--input",
                    "placeholder.3mf",
                    "--boundary-retopology-band-mm",
                    "12.5",
                    "--surface-band-validation",
                    "advisory",
                ]
            )
        )

        self.assertAlmostEqual(strict.maximum_band_fraction, 0.45)
        self.assertAlmostEqual(advisory.maximum_band_fraction, 0.60)
        self.assertAlmostEqual(advisory.maximum_safe_target_offset_mm, 7.5)

    def test_every_loop_uses_same_service_and_preserves_order(self) -> None:
        angle = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        first = np.column_stack((5.0 * np.cos(angle), 3.0 * np.sin(angle), np.zeros(64)))
        second = np.column_stack((2.0 * np.cos(angle), 1.5 * np.sin(angle), np.ones(64)))
        vertices = np.vstack((first, second))
        context = PlanarArcRetopologyContext(
            config=PlanarArcRetopologyConfig(),
        )
        result, records = InterfaceRetopologyService.retopologize_local_loops(
            vertices,
            [list(range(64)), list(range(64, 128))],
            np.arange(128, dtype=np.int64),
            context,
        )
        self.assertEqual(result.shape, vertices.shape)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(item["source_order_preserved"] for item in records))
        self.assertTrue(all(item["target_samples"] == 384 for item in records))

    def test_advisory_surface_band_locks_isolated_shared_loop_junction(self) -> None:
        source = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [-2.0, 0.0, 0.0],
                [-2.0, -2.0, 0.0],
                [0.0, -2.0, 0.0],
            ],
            dtype=np.float64,
        )
        loops = [[0, 1, 2, 3], [0, 4, 5, 6]]
        context = PlanarArcRetopologyContext(
            config=PlanarArcRetopologyConfig(
                retopology_band_mm=10.0,
                surface_band_validation="advisory",
            )
        )

        result, records = InterfaceRetopologyService.retopologize_local_loops(
            source,
            loops,
            np.arange(len(source), dtype=np.int64),
            context,
        )

        np.testing.assert_allclose(result[0], source[0], atol=1e-12)
        self.assertTrue(
            all(item["shared_boundary_vertex_advisory_accepted"] for item in records)
        )
        self.assertTrue(all(item["shared_boundary_vertex_count"] == 1 for item in records))

    def test_strict_surface_band_rejects_isolated_shared_loop_junction(self) -> None:
        source = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [-2.0, 0.0, 0.0],
                [-2.0, -2.0, 0.0],
                [0.0, -2.0, 0.0],
            ],
            dtype=np.float64,
        )
        with self.assertRaisesRegex(PlanarArcError, "share a boundary vertex"):
            InterfaceRetopologyService.retopologize_local_loops(
                source,
                [[0, 1, 2, 3], [0, 4, 5, 6]],
                np.arange(len(source), dtype=np.int64),
                PlanarArcRetopologyContext(config=PlanarArcRetopologyConfig()),
            )

    def test_visible_boundary_moves_and_displacement_fades_into_surface_band(self) -> None:
        samples = 64
        angle = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        ripple = 0.16 * np.sin(11.0 * angle)
        outer = np.column_stack(
            ((5.0 + ripple) * np.cos(angle), 3.0 * np.sin(angle), np.zeros(samples))
        )
        inner = np.column_stack(
            (4.4 * np.cos(angle), 2.4 * np.sin(angle), np.zeros(samples))
        )
        vertices = np.vstack((outer, inner, np.asarray([[0.0, 0.0, 0.0]])))
        center = 2 * samples
        faces: list[list[int]] = []
        for index in range(samples):
            following = (index + 1) % samples
            faces.extend(
                (
                    [index, following, samples + index],
                    [following, samples + following, samples + index],
                    [samples + index, samples + following, center],
                )
            )
        context = PlanarArcRetopologyContext(
            config=PlanarArcRetopologyConfig(retopology_band_mm=1.30),
        )

        result, records = InterfaceRetopologyService.retopologize_local_loops(
            vertices,
            [list(range(samples))],
            np.arange(len(vertices), dtype=np.int64),
            context,
            local_faces=np.asarray(faces, dtype=np.int64),
        )

        self.assertGreater(float(np.linalg.norm(result[:samples] - outer, axis=1).max()), 0.01)
        self.assertGreater(
            float(np.linalg.norm(result[samples : 2 * samples] - inner, axis=1).max()),
            0.0,
        )
        np.testing.assert_allclose(result[center], vertices[center], atol=1e-12)
        self.assertFalse(records[0]["visible_top_source_preserved"])
        self.assertTrue(records[0]["visible_boundary_retopologized"])
        self.assertTrue(records[0]["visible_surface_band_quality"]["valid"])

    def test_invalid_surface_band_calls_failure_sink_without_weakening_audit(self) -> None:
        source = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        captured: list[dict] = []

        _result, quality = _surface_band_deformation(
            source,
            faces,
            np.arange(4, dtype=np.int64),
            source * 20.0,
            25.0,
            captured.append,
        )

        self.assertFalse(quality["valid"])
        self.assertGreater(quality["maximum_edge_stretch_ratio"], 8.0)
        self.assertEqual(len(captured), 1)
        self.assertFalse(captured[0]["quality"]["valid"])

        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = export_retopology_failure_diagnostics(
                captured[0],
                Path(temporary_directory),
            )
            self.assertTrue(all(Path(path).is_file() for path in artifacts.values()))

    def test_user_reviewed_surface_band_relaxes_visual_metrics_only(self) -> None:
        source = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

        _result, quality = _surface_band_deformation(
            source,
            faces,
            np.arange(4, dtype=np.int64),
            source * 100.0,
            175.0,
            validation_mode="advisory",
        )

        self.assertTrue(quality["valid"])
        self.assertEqual(quality["degenerate_face_count"], 0)
        self.assertEqual(quality["introduced_over_shared_edge_count"], 0)
        self.assertEqual(quality["introduced_inconsistent_shared_edge_count"], 0)
        self.assertGreater(quality["maximum_edge_stretch_ratio"], 64.0)
        self.assertLessEqual(quality["maximum_edge_stretch_ratio"], 128.0)
        self.assertTrue(quality["edge_stretch_advisory_accepted"])


if __name__ == "__main__":
    unittest.main()
