"""Opt-in, full-pipeline regression against the user-approved cathead split."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from golden_cli import artifact_directory, golden_directory

import numpy as np
from scipy.spatial import cKDTree


SKILL = Path(__file__).resolve().parents[1]
REPO = SKILL.parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SOURCE = REPO / "example/cathead.3mf"
APPROVED = REPO / "example/cathead_split_parts.3mf"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _run(command: list[str], log: Path, *, env: dict[str, str]) -> str:
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=output,
                                stderr=subprocess.STDOUT, text=True, timeout=1200)
    contents = log.read_text(encoding="utf-8", errors="replace")
    if result.returncode:
        raise AssertionError(f"{command[1]} exited {result.returncode}:\n{contents[-5000:]}")
    return contents


def _nearest_p99(reference: np.ndarray, candidate: np.ndarray) -> float:
    # Fixed stride bounds runtime without requiring vertex or triangle identity.
    stride = max(1, len(reference) // 5000)
    distances = cKDTree(candidate).query(reference[::stride], workers=1)[0]
    return float(np.quantile(distances, .99))


@unittest.skipUnless(os.environ.get("RUN_CATHEAD_GOLDEN") == "1",
                     "set RUN_CATHEAD_GOLDEN=1 for the full 3MF regression")
class CatheadGoldenRegression(unittest.TestCase):
    def test_fresh_split_matches_approved_parts(self) -> None:
        sys.path.insert(0, str(SKILL / "scripts"))
        from split3mf.common import load_core_dependencies
        from split3mf.package_io import load_colored_mesh_objects_3mf

        load_core_dependencies()
        fixture = _read(FIXTURES / "cathead_golden.json")
        semantics_path = FIXTURES / "cathead_visual_semantics.json"
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                         fixture["source_sha256"])
        self.assertTrue(APPROVED.is_file())
        self.assertEqual(hashlib.sha256(APPROVED.read_bytes()).hexdigest(),
                         fixture["approved_sha256"])

        with golden_directory(prefix="cathead-golden-") as directory:
            root = Path(directory)
            env = {**os.environ, "MPLCONFIGDIR": str(root / "matplotlib")}
            first = _run([
                sys.executable, "-X", "utf8", "-B",
                str(SKILL / "scripts/split_painted_3mf.py"),
                "--input", str(SOURCE), "--output", str(root / "result.3mf"),
                "--stage-artifacts-dir", str(root / "stages"),
                "--recovery-dir", str(root / "recovery"),
                "--visual-semantics-json", str(semantics_path),
                "--stop-after-stage", "recognize",
            ], root / "recognize.log", env=env)
            run_dir = artifact_directory(first, "stage_artifacts")

            recognition = _read(run_dir / "03_recognition_review.json")
            self.assertEqual(recognition["source_face_count"], fixture["source_face_count"])
            expected_regions = [
                {key: item[key] for key in
                 ("part", "source_min_face_index", "face_count", "color_code")}
                for item in fixture["regions"]
            ]
            self.assertEqual(recognition["revised_regions"], expected_regions)
            recognized = _read(run_dir / "03_recognition_summary.json")["result"]["regions"]
            self.assertEqual(
                {f"P{int(item['part_index']):02d}": item["visual_semantic_label"]
                 for item in recognized},
                {item["part"]: item["label"] for item in fixture["regions"]},
            )
            self.assertEqual(recognition["actions"], [])
            # Region signatures must match before this test writes a temporary
            # confirmation on behalf of the previously reviewed golden case.
            recognition["user_confirmed"] = True
            _write(run_dir / "03_recognition_review.json", recognition)
            status_path = run_dir / "03_recognition_review_status.json"
            status = _read(status_path)
            self.assertEqual(status["result"]["result_fingerprint"],
                             recognition["expected_result_fingerprint"])
            status["result"]["status"] = "user_confirmed"
            _write(status_path, status)

            _run([
                sys.executable, "-X", "utf8", "-B",
                str(SKILL / "scripts/run_stage04_from_artifacts.py"),
                "--source-run-dir", str(run_dir), "--input", str(SOURCE),
                "--recognition-review-json", str(run_dir / "03_recognition_review.json"),
                "--visual-semantics-json", str(semantics_path),
            ], root / "assembly.log", env=env)
            plan = _read(run_dir / "04_assembly_plan.json")["result"]
            self.assertEqual(plan["interface_count"], len(fixture["interfaces"]))
            for expected, actual in zip(fixture["interfaces"], plan["interfaces"]):
                with self.subTest(interface=expected["interface_id"]):
                    for key in ("interface_id", "tenon_part", "mortise_part"):
                        self.assertEqual(actual[key], expected[key])
                    a = np.asarray(actual["insertion_direction"], dtype=float)
                    b = np.asarray(expected["insertion_direction"], dtype=float)
                    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                    self.assertGreaterEqual(
                        cosine, math.cos(math.radians(fixture["max_axis_angle_degrees"]))
                    )
            review_path = run_dir / "04_assembly_review.json"
            review = _read(review_path)
            self.assertEqual(review["correction_requests"], [])
            review["user_confirmed"] = True
            _write(review_path, review)

            semantics = _read(semantics_path)["parts"]
            proposal = {
                "schema": "painted-3mf-semantic-proposals/v1",
                "source_run_dir": str(run_dir),
                "user_confirmed": True,
                "parts": {
                    part: {"label": item["label"], "confidence": item["confidence"],
                           "evidence": item["visual_evidence"]}
                    for part, item in semantics.items()
                },
                "fragments": {},
            }
            proposal_path = root / "semantic-review.json"
            _write(proposal_path, proposal)
            output = _run([
                sys.executable, "-X", "utf8", "-B",
                str(SKILL / "scripts/run_stage05_from_artifacts.py"),
                "--source-run-dir", str(run_dir), "--output-root", str(root / "stage05"),
                "--input", str(SOURCE), "--assembly-review-json", str(review_path),
                "--semantic-review-json", str(proposal_path),
            ], root / "interfaces.log", env=env)
            final_dir = artifact_directory(output, "stage05_artifacts")
            result = _read(final_dir / "05_interface_assembly_summary.json")["result"]
            self.assertEqual(result["stage_status"], "complete_parts_exported")
            self.assertFalse(result["quality_gates_blocking"])
            self.assertEqual(result["interface_surface_count"], len(fixture["interfaces"]))
            self.assertEqual(result["exported_part_count"], len(fixture["regions"]))
            self.assertTrue(result["package_validation"]["valid"])
            for interface in result["interfaces"]:
                with self.subTest(clearance=interface["interface_id"]):
                    self.assertAlmostEqual(interface["measured_clearance_mm"], .2, places=6)

            baseline = {item["part_id"]: item
                        for item in load_colored_mesh_objects_3mf(APPROVED)}
            actual = {item["part_id"]: item for item in load_colored_mesh_objects_3mf(
                final_dir / "05_complete_parts.3mf")}
            self.assertEqual(set(actual), set(fixture["part_colors"]))
            for part_id, expected_color in fixture["part_colors"].items():
                with self.subTest(part=part_id):
                    old, new = baseline[part_id], actual[part_id]
                    self.assertEqual(new["color_hex"], expected_color)
                    self.assertTrue(new["mesh"].is_watertight)
                    self.assertLessEqual(
                        float(np.max(np.abs(new["mesh"].bounds - old["mesh"].bounds))),
                        fixture["bbox_tolerance_mm"],
                    )
                    self.assertLessEqual(
                        _nearest_p99(old["mesh"].vertices, new["mesh"].vertices),
                        fixture["surface_vertex_p99_mm"],
                    )
                    self.assertLessEqual(
                        _nearest_p99(new["mesh"].vertices, old["mesh"].vertices),
                        fixture["surface_vertex_p99_mm"],
                    )
