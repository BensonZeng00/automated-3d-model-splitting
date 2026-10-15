from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from split3mf import common
common.load_core_dependencies()
from split3mf.region_review import select_region_review_groups
from split3mf.cli import build_parser
from split3mf.source_region_review import (
    build_source_region_review,
    load_confirmed_region_decisions,
)


class SourceRegionReviewTests(unittest.TestCase):
    @staticmethod
    def sample():
        vertices = np.asarray([[0,0,0],[1,0,0],[0,1,0],[2,0,0],[3,0,0],[2,1,0]], float)
        faces = np.asarray([[0,1,2],[3,4,5]], int)
        groups = [np.asarray([0]), np.asarray([1])]
        colors = ["base", "detail"]
        return vertices, faces, groups, colors

    def test_face_thresholds_select_review_without_changing_groups(self):
        vertices, faces, groups, _ = self.sample()
        ordinary, candidates = select_region_review_groups(vertices, faces, groups, 1, 999)
        self.assertEqual(ordinary, [])
        self.assertEqual([category for _, category in candidates], ["noise_candidate"] * 2)
        np.testing.assert_array_equal(candidates[0][0], groups[0])
        np.testing.assert_array_equal(candidates[1][0], groups[1])

    def test_100_1000_review_boundaries(self):
        vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        faces = np.tile(np.asarray([[0, 1, 2]], int), (1100, 1))
        groups = [np.arange(100), np.arange(100, 1100)]
        ordinary, candidates = select_region_review_groups(
            vertices, faces, groups, noise_max_faces=100,
            small_region_max_faces=999)
        self.assertEqual([category for _, category in candidates], ["noise_candidate"])
        self.assertEqual(len(ordinary), 1)
        self.assertEqual(len(ordinary[0]), 1000)

    def test_legacy_review_cli_is_not_supported(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--input", "model.3mf", "--min-faces", "1"])
        parsed = parser.parse_args([
            "--input", "model.3mf", "--noise-review-max-faces", "100",
            "--small-region-review-max-faces", "999"])
        self.assertEqual(parsed.noise_review_max_faces, 100)
        self.assertEqual(parsed.small_region_review_max_faces, 999)

    def test_review_classification_contract_preserves_all_candidates(self):
        vertices, faces, groups, colors = self.sample()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "sample.3mf"
            source.write_bytes(b"placeholder")
            review = build_source_region_review(
                input_path=source, review_dir=root / "review", vertices=vertices,
                faces=faces, connectivity_colors=colors, display_colors=colors,
                groups=groups, noise_max_faces=1, small_region_max_faces=999,
                visible_faces=np.ones(len(faces), dtype=bool), view_count=6,
                depth_map_resolution=64, image_resolution=128)
            self.assertEqual(len(review["items"]), 2)
            payload_path = Path(review["decision_path"])
            payload = json.loads(payload_path.read_text())
            payload["user_confirmed"] = True
            for index, item in enumerate(payload["items"]):
                item["semantic_label"] = "noise" if index == 0 else "detail"
                item["visual_confidence"] = "HIGH"
                item["classification"] = "noise" if index == 0 else "part"
            payload_path.write_text(json.dumps(payload))
            loaded = load_confirmed_region_decisions(
                payload_path, input_path=source, source_face_count=len(faces),
                noise_max_faces=1, small_region_max_faces=999,
                expected_records=review["items"])
            self.assertEqual(loaded[1]["classification"], "noise")
            self.assertEqual(loaded[2]["classification"], "part")
            self.assertNotIn("preserve", loaded[1])

    def test_invalid_classification_is_rejected(self):
        vertices, faces, groups, colors = self.sample()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir); source = root / "sample.3mf"; source.write_bytes(b"x")
            review = build_source_region_review(
                input_path=source, review_dir=root / "review", vertices=vertices,
                faces=faces, connectivity_colors=colors, display_colors=colors,
                groups=groups, noise_max_faces=1, small_region_max_faces=999,
                visible_faces=None, view_count=6, depth_map_resolution=64,
                image_resolution=128)
            path = Path(review["decision_path"]); payload = json.loads(path.read_text())
            payload["user_confirmed"] = True
            for item in payload["items"]:
                item.update(semantic_label="candidate", visual_confidence="MED", classification="merge")
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "noise, part, or uncertain"):
                load_confirmed_region_decisions(
                    path, input_path=source, source_face_count=len(faces),
                    noise_max_faces=1, small_region_max_faces=999,
                    expected_records=review["items"])


if __name__ == "__main__":
    unittest.main()
