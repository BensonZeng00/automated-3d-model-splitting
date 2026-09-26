from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.cli import build_parser
from split3mf.recursive_preflight import (
    FullTreePreflightError,
    FullTreePreflightService,
)
from split3mf.stage_cache import RecursiveStageCache, normalized_run_arguments
from split3mf.common import Component
from split3mf.explicit_merge import merge_body_components, parse_part_group
from split3mf.cap_template import fit_affine_cap_inside_parent
from split3mf.part_geometry import ParentThicknessProbe


class _Decision:
    def __init__(self, safe_maximum: float = 2.0) -> None:
        self.mode = "flat"
        self.record = {
            "parent_thickness_min_mm": safe_maximum + 0.05,
            "safe_maximum_inward_depth_mm": safe_maximum,
            "effective_minimum_inward_depth_mm": min(1.0, safe_maximum),
            "parent_thickness_is_lower_bound": False,
        }


class PipelineOptimizationTests(unittest.TestCase):
    def test_affine_cap_backoff_can_converge_after_four_measurements(self) -> None:
        template = np.array(
            [
                [0.0, 0.0, 4.0],
                [1.0, 0.0, 4.0],
                [0.0, 1.0, 4.0],
            ],
            dtype=np.float64,
        )
        fit_points = template.copy()
        fit_points[:, 2] = 0.0
        calls = 0

        def staged_safety(_points, _directions, _maximum):
            nonlocal calls
            calls += 1
            safe = 3.9 - 0.11 * (calls - 1) if calls < 6 else 10.0
            return safe, {"calls": calls}

        result = fit_affine_cap_inside_parent(
            template_points=template,
            boundary_indices=np.array([0, 1, 2], dtype=np.int64),
            fit_points=fit_points,
            inward_axis=np.array([0.0, 0.0, 1.0]),
            safety_limit=staged_safety,
            maximum_depth_mm=10.0,
        )
        self.assertEqual(result.attempts, 6)
        self.assertGreater(result.distances.min(), 0.08)
        self.assertEqual(result.thickness_record["calls"], 6)
        self.assertEqual(len(result.trace), 6)

    def test_affine_cap_reuses_only_broad_phase_and_remeasures_safety(self) -> None:
        class Probe:
            def __init__(self) -> None:
                self.preparations = 0
                self.measurements = 0
                self.cache = object()

            def prepare_safety_limit_candidates(self, points, search_limit):
                self.preparations += 1
                self.asserted_points = np.asarray(points).copy()
                self.asserted_limit = search_limit
                return self.cache

            def safety_limit(
                self, points, directions, maximum, *, broad_phase_cache=None
            ):
                self.measurements += 1
                if broad_phase_cache is not self.cache:
                    raise AssertionError("affine retry did not reuse candidate cache")
                safe = 3.9 if self.measurements == 1 else 10.0
                return safe, {"measurement": self.measurements}

        template = np.array(
            [[0.0, 0.0, 4.0], [1.0, 0.0, 4.0], [0.0, 1.0, 4.0]],
            dtype=np.float64,
        )
        fit_points = template.copy()
        fit_points[:, 2] = 0.0
        probe = Probe()
        result = fit_affine_cap_inside_parent(
            template_points=template,
            boundary_indices=np.arange(3, dtype=np.int64),
            fit_points=fit_points,
            inward_axis=np.array([0.0, 0.0, 1.0]),
            safety_limit=probe.safety_limit,
            maximum_depth_mm=10.0,
        )

        self.assertEqual(probe.preparations, 1)
        self.assertEqual(probe.measurements, 2)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.thickness_record["measurement"], 2)
        np.testing.assert_array_equal(probe.asserted_points, fit_points)
        self.assertAlmostEqual(probe.asserted_limit, 10.0)

    def test_reusable_candidates_preserve_exact_directional_hits(self) -> None:
        vertices = np.array(
            [
                [-2.0, -2.0, 2.0], [2.0, -2.0, 2.0],
                [2.0, 2.0, 2.0], [-2.0, 2.0, 2.0],
                [3.0, -2.0, 3.0], [3.0, 2.0, 3.0],
                [3.0, 2.0, 7.0], [3.0, -2.0, 7.0],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]],
            dtype=np.int64,
        )
        probe = ParentThicknessProbe(vertices, faces)
        points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        cache = probe.prepare_safety_limit_candidates(points, 10.0)
        for directions in (
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[3.0, 0.0, 5.0]]),
        ):
            expected = probe.first_hit_distances(points, directions, 10.05)
            actual = probe.first_hit_distances(
                points, directions, 10.05, broad_phase_cache=cache
            )
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_explicit_body_merge_preserves_anchor_and_remaps_indices(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

        def component(index: int, color: str) -> Component:
            points = vertices[faces[[index]].reshape(-1)]
            return Component(
                color_code=color,
                global_faces=np.array([index], dtype=np.int64),
                face_count=1,
                area=0.5,
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                center=points.mean(axis=0),
            )

        result = merge_body_components(
            vertices,
            faces,
            [component(0, "yellow"), component(1, "apricot")],
            parse_part_group("P02+P01"),
        )
        self.assertEqual(len(result.components), 1)
        self.assertEqual(result.body_index, 1)
        self.assertEqual(result.components[0].color_code, "apricot")
        self.assertEqual(result.components[0].global_faces.tolist(), [0, 1])
        self.assertEqual(result.original_to_effective, {1: 1, 2: 1})
        self.assertTrue(result.record["per_face_materials_preserved"])

    def test_cli_removes_recursive_pipeline_switches(self) -> None:
        parser = build_parser()
        for option in ("--full-tree-preflight", "--resume", "--debug-recursive-3mf"):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                parser.parse_args(["--input", "placeholder.3mf", option])

    def test_preflight_toggle_does_not_change_geometry_cache_arguments(self) -> None:
        enabled = normalized_run_arguments(
            SimpleNamespace(full_tree_preflight=True, cap_mode="adaptive")
        )
        disabled = normalized_run_arguments(
            SimpleNamespace(full_tree_preflight=False, cap_mode="adaptive")
        )
        self.assertEqual(enabled, disabled)

    def test_full_tree_preflight_visits_every_recursive_parent(self) -> None:
        visited: list[int] = []

        def loader(parent_index: int):
            visited.append(parent_index)
            return (
                [
                    {
                        "component_index": parent_index + 1,
                        "loop_index": 0,
                        "global_loop": [1, 2, 3],
                        "fit_clearance_mm": 0.4,
                        "requested_cap_mode": "adaptive",
                        "cap_decision": _Decision(),
                    }
                ],
                {},
                {},
            )

        report = FullTreePreflightService().run(
            [
                {"local_body_index": 1},
                {"local_body_index": 4},
            ],
            loader,
        )
        self.assertEqual(visited, [1, 4])
        self.assertEqual(report.status, "PASS")
        self.assertEqual(len(report.findings), 2)

    def test_full_tree_preflight_blocks_before_later_parents(self) -> None:
        visited: list[int] = []

        def loader(parent_index: int):
            visited.append(parent_index)
            if parent_index == 4:
                raise ValueError("P06 has no printable inward depth")
            return ([], {}, {})

        with self.assertRaises(FullTreePreflightError) as raised:
            FullTreePreflightService().run(
                [
                    {"local_body_index": 1},
                    {"local_body_index": 4},
                    {"local_body_index": 7},
                ],
                loader,
            )
        self.assertEqual(visited, [1, 4])
        self.assertEqual(raised.exception.report.status, "BLOCK")
        self.assertEqual(raised.exception.report.blocked_parent_index, 4)

    def test_stage_cache_rejects_tampered_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "part.3mf"
            source.write_bytes(b"validated recursive artifact")
            cache = RecursiveStageCache(root / "cache", mode="auto")
            key = cache.stage_key(
                run_fingerprint="run",
                implementation="implementation",
                step={"step_order": 0, "local_body_index": 1},
                input_artifact_sha256="source",
            )
            cache.commit(
                key=key,
                run_fingerprint="run",
                implementation="implementation",
                step={"step_order": 0, "local_body_index": 1},
                input_artifact_sha256="source",
                changed_entries=[
                    {
                        "part_id": "P01",
                        "source_3mf_path": str(source),
                        "source_part_index": 1,
                        "contains": [1],
                        "state_role": "root_body",
                        "origin_step": 0,
                        "stats": {},
                    }
                ],
                stage_records=[{"part_id": "P01"}],
            )
            hit = cache.lookup(key)
            self.assertIsNotNone(hit)
            artifact = next((hit["stage_dir"]).glob("*.3mf"))
            artifact.write_bytes(b"tampered")
            self.assertIsNone(cache.lookup(key))
            with self.assertRaises(ValueError):
                RecursiveStageCache(root / "cache", mode="strict").lookup(key)


if __name__ == "__main__":
    unittest.main()
