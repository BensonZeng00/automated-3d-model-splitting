from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common

common.load_core_dependencies()

from split3mf.recognition import (
    merge_tiny_groups_with_semantic_preservation,
    merge_tiny_groups_with_user_review,
    tiny_group_pair_similarity,
)
from split3mf.small_component_review import (
    build_tiny_component_review,
    load_confirmed_tiny_component_decisions,
)


class TinyComponentSemanticTests(unittest.TestCase):
    @staticmethod
    def sample_groups():
        cube_vertices = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float64,
        )
        cube_faces = np.asarray(
            [
                [0, 2, 1], [0, 3, 2],
                [0, 1, 5], [0, 5, 4],
                [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6],
                [3, 0, 4], [3, 4, 7],
                [4, 5, 6], [4, 6, 7],
            ],
            dtype=np.int64,
        )
        noise_vertices = np.asarray(
            [[1.05, 0.45, 0.45], [1.10, 0.45, 0.45], [1.05, 0.50, 0.45], [1.05, 0.45, 0.50]],
            dtype=np.float64,
        )
        noise_faces = np.asarray(
            [[8, 10, 9], [8, 9, 11], [8, 11, 10], [9, 10, 11]],
            dtype=np.int64,
        )
        vertices = np.vstack((cube_vertices, noise_vertices))
        faces = np.vstack((cube_faces, noise_faces))
        base = np.arange(0, 10, dtype=np.int64)
        visual_patch = np.arange(10, 12, dtype=np.int64)
        noise = np.arange(12, 16, dtype=np.int64)
        colors = ["base"] * 10 + ["detail"] * 2 + ["noise"] * 4
        return vertices, faces, [base, visual_patch, noise], colors

    def test_projected_closed_patch_is_preserved_but_closed_island_noise_merges(self) -> None:
        cube_vertices = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float64,
        )
        cube_faces = np.asarray(
            [
                [0, 2, 1], [0, 3, 2],
                [0, 1, 5], [0, 5, 4],
                [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6],
                [3, 0, 4], [3, 4, 7],
                [4, 5, 6], [4, 6, 7],
            ],
            dtype=np.int64,
        )
        noise_vertices = np.asarray(
            [[1.05, 0.45, 0.45], [1.10, 0.45, 0.45], [1.05, 0.50, 0.45], [1.05, 0.45, 0.50]],
            dtype=np.float64,
        )
        noise_faces = np.asarray(
            [[8, 10, 9], [8, 9, 11], [8, 11, 10], [9, 10, 11]],
            dtype=np.int64,
        )
        vertices = np.vstack((cube_vertices, noise_vertices))
        faces = np.vstack((cube_faces, noise_faces))
        base = np.arange(0, 10, dtype=np.int64)
        visual_patch = np.arange(10, 12, dtype=np.int64)
        noise = np.arange(12, 16, dtype=np.int64)
        colors = ["base"] * 10 + ["detail"] * 2 + ["noise"] * 4

        components, ignored, merged, preserved = merge_tiny_groups_with_semantic_preservation(
            vertices=vertices,
            faces=faces,
            colors=colors,
            groups=[base, visual_patch, noise],
            min_faces=5,
            display_colors=colors,
            visible_faces=np.ones(len(faces), dtype=bool),
            view_count=12,
            depth_map_resolution=256,
        )

        self.assertEqual(len(components), 2)
        self.assertEqual(ignored, [])
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0]["color_code"], "detail")
        self.assertEqual(
            preserved[0]["semantic_decision"],
            "preserve_independent_small_component",
        )
        self.assertEqual(preserved[0]["boundary_loop_count"], 1)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["color_code"], "noise")
        self.assertEqual(merged[0]["semantic_decision"], "merge_probable_noise")
        self.assertIn("no_paint_boundary_loop", merged[0]["semantic_penalties"])

    def test_repeated_feature_similarity_requires_same_material(self) -> None:
        left = {
            "connectivity_color_code": "white",
            "faces": 260,
            "area_mm2": 3.5,
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [1.6, 1.1, 2.4],
        }
        matching = {
            "connectivity_color_code": "white",
            "faces": 264,
            "area_mm2": 3.45,
            "bbox_min": [4.0, 0.0, 0.0],
            "bbox_max": [5.7, 1.15, 2.42],
        }
        other_material = {**matching, "connectivity_color_code": "black"}
        self.assertGreater(tiny_group_pair_similarity(left, matching), 0.95)
        self.assertEqual(tiny_group_pair_similarity(left, other_material), 0.0)

    def test_user_selection_is_authoritative_and_unselected_fragment_merges(self) -> None:
        vertices, faces, groups, colors = self.sample_groups()
        decisions = {
            1: {
                "fragment_id": 1,
                "source_min_face_index": 10,
                "semantic_label": "painted facial detail",
                "visual_confidence": "HIGH",
                "preserve": True,
            },
            2: {
                "fragment_id": 2,
                "source_min_face_index": 12,
                "semantic_label": "isolated paint noise",
                "visual_confidence": "MED",
                "preserve": False,
            },
        }
        components, ignored, merged, preserved = merge_tiny_groups_with_user_review(
            vertices=vertices,
            faces=faces,
            colors=colors,
            groups=groups,
            min_faces=5,
            decisions=decisions,
            auto_noise_max_faces=1,
            display_colors=colors,
            visible_faces=np.ones(len(faces), dtype=bool),
            view_count=12,
            depth_map_resolution=256,
        )
        self.assertEqual(ignored, [])
        self.assertEqual(len(components), 2)
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0]["semantic_label"], "painted facial detail")
        self.assertEqual(preserved[0]["semantic_decision_source"], "user_confirmed_image_review")
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["semantic_label"], "isolated paint noise")
        self.assertEqual(merged[0]["semantic_decision"], "merge_user_unselected_component")

    def test_review_images_and_confirmed_decision_contract(self) -> None:
        vertices, faces, groups, colors = self.sample_groups()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "sample.3mf"
            source.write_bytes(b"placeholder")
            review = build_tiny_component_review(
                input_path=source,
                review_dir=root / "review",
                vertices=vertices,
                faces=faces,
                connectivity_colors=colors,
                display_colors=colors,
                groups=groups,
                min_faces=5,
                auto_noise_max_faces=1,
                visible_faces=np.ones(len(faces), dtype=bool),
                view_count=12,
                depth_map_resolution=256,
                image_resolution=128,
            )
            self.assertEqual(len(review["items"]), 2)
            for item in review["items"]:
                image_path = Path(item["review_image"])
                self.assertTrue(image_path.exists())
                self.assertEqual(image_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

            decision_path = Path(review["decision_path"])
            payload = json.loads(decision_path.read_text(encoding="utf-8"))
            payload["user_confirmed"] = True
            for index, item in enumerate(payload["items"], start=1):
                item["semantic_label"] = f"candidate {index}"
                item["visual_confidence"] = "MED"
                item["preserve"] = index == 1
            decision_path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_confirmed_tiny_component_decisions(
                decision_path,
                input_path=source,
                source_face_count=len(faces),
                min_faces=5,
                auto_noise_max_faces=1,
                expected_records=review["items"],
            )
            self.assertTrue(loaded[1]["preserve"])
            self.assertFalse(loaded[2]["preserve"])

    def test_face_count_at_or_below_auto_noise_threshold_skips_review(self) -> None:
        vertices, faces, groups, colors = self.sample_groups()
        components, ignored, merged, preserved = merge_tiny_groups_with_user_review(
            vertices=vertices,
            faces=faces,
            colors=colors,
            groups=groups,
            min_faces=5,
            decisions={},
            auto_noise_max_faces=4,
            display_colors=colors,
            visible_faces=np.ones(len(faces), dtype=bool),
            view_count=12,
            depth_map_resolution=256,
        )
        self.assertEqual(ignored, [])
        self.assertEqual(len(components), 1)
        self.assertEqual(preserved, [])
        self.assertEqual(len(merged), 2)
        self.assertTrue(
            all(item["semantic_decision"] == "merge_auto_noise_face_threshold" for item in merged)
        )
        self.assertTrue(all(item["review_image_generated"] is False for item in merged))
        self.assertTrue(all(item["user_confirmed"] is False for item in merged))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "sample.3mf"
            source.write_bytes(b"placeholder")
            review = build_tiny_component_review(
                input_path=source,
                review_dir=root / "review",
                vertices=vertices,
                faces=faces,
                connectivity_colors=colors,
                display_colors=colors,
                groups=groups,
                min_faces=5,
                auto_noise_max_faces=4,
                visible_faces=np.ones(len(faces), dtype=bool),
                view_count=12,
                depth_map_resolution=256,
                image_resolution=128,
            )
            self.assertFalse(review["review_required"])
            self.assertEqual(review["auto_noise_count"], 2)
            self.assertEqual(review["items"], [])
            self.assertEqual(list((root / "review").glob("*.png")), [])


if __name__ == "__main__":
    unittest.main()
