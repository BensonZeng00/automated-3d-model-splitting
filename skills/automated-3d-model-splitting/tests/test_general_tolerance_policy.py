from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf import common
common.load_core_dependencies()
from split3mf.tolerance_policy import TolerancePolicy, maximum_span, slope_warning
from split3mf.micro_regions import merge_micro_regions
from split3mf.micro_openings import seal_micro_openings
from split3mf.recognition import summarize_components
from split3mf.mesh import boundary_loops
from split3mf.part_geometry import (
    boundary_loop_interior_conormals,
    mesh_vertex_conormal_evidence,
    normalized_circular_convolution,
)
from split3mf.package_io import prepare_debug_directory
from split3mf.boundary_review import BoundaryReviewService
from split3mf.cli import build_parser
from split3mf.domain import PlanarArcRetopologyConfig


class GeneralToleranceTests(unittest.TestCase):
    def components(self, vertices, faces, groups):
        return summarize_components(vertices, faces, ['8'] * len(faces), groups, 1)[0]

    def test_strict_warning_boundary_and_defaults(self):
        self.assertFalse(TolerancePolicy().warning(1, 100))
        self.assertTrue(TolerancePolicy().warning(2, 100))
        self.assertFalse(slope_warning(0, 0)['warning'])
        self.assertFalse(slope_warning(1, 100)['warning'])
        self.assertTrue(slope_warning(1, 99)['warning'])
        args = build_parser().parse_args(['--input', 'unused.3mf'])
        self.assertEqual(args.boundary_retopology_band_mm, 3.0)
        self.assertEqual(args.connector_slope_validation, 'advisory')
        self.assertAlmostEqual(PlanarArcRetopologyConfig().maximum_safe_target_offset_mm, 1.35)

    def test_diameter_uses_euclidean_span_not_axis_or_radius(self):
        self.assertEqual(maximum_span([[0, 0, 0], [2, 0, 0]]), 2)
        self.assertGreater(maximum_span([[0, 0, 0], [1.5, 1.5, 0]]), 2)
        self.assertGreater(maximum_span([[0, 0, 0], [2.001, 0, 0]]), 2)

    def test_merge_preserves_faces_and_uses_only_shared_edge_neighbors(self):
        vertices = np.array([[0, 0, 0], [2, 0, 0], [1, .5, 0], [1, -5, 0]], float)
        faces = np.array([[0, 1, 2], [1, 0, 3]])
        original = faces.copy()
        components = self.components(vertices, faces, [np.array([0]), np.array([1])])
        merged, records = merge_micro_regions(vertices, faces, components)
        self.assertEqual(len(merged), 1)
        self.assertEqual(records[0]['action'], 'merged')
        self.assertEqual(records[0]['target_part'], 1)
        np.testing.assert_array_equal(np.sort(merged[0].global_faces), [0, 1])
        np.testing.assert_array_equal(faces, original)
        disconnected = np.vstack([vertices, vertices[:3] + [0, 0, .01]])
        separate_faces = np.vstack([faces, [4, 5, 6]])
        separate_components = self.components(disconnected, separate_faces,
                                               [np.array([0, 1]), np.array([2])])
        _, records = merge_micro_regions(disconnected, separate_faces, separate_components)
        self.assertEqual(records[0]['action'], 'kept')

    def test_small_hole_cap_preserves_original_and_closes_manifold(self):
        mesh = trimesh.creation.box(extents=(1, 1, 1))
        faces = mesh.faces[:-1].copy()
        colors = ['8'] * len(faces)
        components = self.components(mesh.vertices, faces, [np.arange(len(faces))])
        result, paint, parts, records = seal_micro_openings(mesh.vertices, faces, colors, components)
        self.assertEqual(records[0]['action'], 'merged-into-adjacent-surface')
        np.testing.assert_array_equal(result[:len(faces)], faces)
        self.assertEqual(paint[:len(faces)], colors)
        self.assertEqual(parts[0].face_count, 12)
        repaired = trimesh.Trimesh(mesh.vertices, result, process=False)
        self.assertTrue(repaired.is_watertight)
        self.assertTrue(repaired.is_winding_consistent)
        unchanged, _, _, records = seal_micro_openings(mesh.vertices * 3, faces, colors, components)
        np.testing.assert_array_equal(unchanged, faces)
        self.assertEqual(records, [])

    def test_many_touching_cycles_do_not_recurse(self):
        faces = np.array([[0, 2*i+1, 2*i+2] for i in range(20_000)])
        loops = boundary_loops(faces)
        self.assertEqual(len(loops), 20_000)
        self.assertTrue(all(len(loop) == 3 for loop in loops))
        self.assertEqual(sum(len(loop) for loop in loops), 60_000)
        expected_edges = {
            tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
            for face in faces
            for index in range(3)
        }
        actual_edges = {
            tuple(sorted((int(loop[index]), int(loop[(index + 1) % len(loop)]))))
            for loop in loops
            for index in range(len(loop))
        }
        self.assertEqual(actual_edges, expected_edges)

    def test_precomputed_conormal_evidence_matches_direct_loop_evaluation(self):
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [.5, .5, 1.]],
            dtype=np.float64,
        )
        faces = np.asarray(
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
            dtype=np.int64,
        )
        loop = [0, 1, 2, 3]
        direct = boundary_loop_interior_conormals(vertices, faces, loop)
        prepared = boundary_loop_interior_conormals(
            vertices,
            faces,
            loop,
            vertex_evidence=mesh_vertex_conormal_evidence(vertices, faces),
        )
        np.testing.assert_allclose(prepared, direct, atol=1e-12)

    def test_compiled_circular_convolution_matches_roll_definition(self):
        angles = np.linspace(0.0, 2.0 * np.pi, 257, endpoint=False)
        values = np.column_stack(
            (np.cos(angles), np.sin(angles), 0.2 * np.sin(7.0 * angles))
        )
        offsets = np.arange(-31, 32, dtype=np.int64)
        weights = np.exp(-0.5 * (offsets / 12.0) ** 2)
        weights /= weights.sum()
        expected = np.zeros_like(values)
        for offset, weight in zip(offsets, weights):
            expected += np.roll(values, int(offset), axis=0) * float(weight)
        expected /= np.linalg.norm(expected, axis=1)[:, None]

        actual = normalized_circular_convolution(values, weights)

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_debug_directories_are_atomic_and_never_erase_previous_run(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'debug'
            first = prepare_debug_directory(base)
            marker = first / 'keep.txt'
            marker.write_text('last valid mesh', encoding='utf-8')
            with ThreadPoolExecutor(max_workers=4) as pool:
                allocated = list(pool.map(lambda _: prepare_debug_directory(base, True), range(8)))
            self.assertEqual(len(set(allocated)), 8)
            self.assertNotIn(first, allocated)
            self.assertEqual(marker.read_text(), 'last valid mesh')

    def test_unresolved_endpoints_warn_without_false_repair_or_review(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        with tempfile.TemporaryDirectory() as directory:
            with patch('split3mf.boundary_preview.render_boundary_review', side_effect=AssertionError):
                owners, report = BoundaryReviewService(Path(directory)).prepare(vertices, faces, ['A', 'B'])
            np.testing.assert_array_equal(owners, ['A', 'B'])
            self.assertEqual(report['nearest_merge']['endpoints_after'], 2)
            self.assertEqual(report['nearest_merge']['changed_face_count'], 0)
            self.assertTrue(report['nearest_merge']['warning'])
            self.assertEqual(report['nearest_merge']['pairs'][0]['boundary_vertices'], 2)

    def test_pipeline_preserves_micro_source_region_and_paint(self):
        from split3mf.domain import SplitConfig
        from split3mf.pipeline import SplitPipeline
        import contextlib
        import io
        mesh = trimesh.creation.box(extents=(8, 8, 8)).subdivide().subdivide().subdivide()
        paint = ['8'] * len(mesh.faces)
        paint[0] = '0C'
        saved_info, saved_order = dict(common.COLOR_INFO), dict(common.COLOR_ORDER)
        self.addCleanup(common.COLOR_CATALOG.replace, saved_info, saved_order)
        with tempfile.TemporaryDirectory() as directory:
            parser = build_parser()
            args = parser.parse_args(['--input', str(Path(directory) / 'source.3mf'),
                '--recognize-only', '--recognition-surface-profile', 'all-faces',
                '--body-strategy', 'largest', '--noise-review-max-faces', '0',
                '--small-region-review-max-faces', '0'])
            pipeline = SplitPipeline(SplitConfig(args, Path(args.input), {}), parser)
            with patch.object(pipeline.reader, 'read', return_value=(mesh.vertices, mesh.faces, paint, {})):
                with patch('split3mf.pipeline.build_source_region_review', side_effect=AssertionError('unnecessary review')):
                    with patch('split3mf.pipeline.print_recognition') as inventory:
                        with contextlib.redirect_stdout(io.StringIO()):
                            pipeline.run()
            self.assertEqual(len(inventory.call_args.args[0]), 2)
            self.assertEqual(paint[0], '0C')
            self.assertEqual(len(paint), len(mesh.faces))


if __name__ == '__main__':
    unittest.main()
