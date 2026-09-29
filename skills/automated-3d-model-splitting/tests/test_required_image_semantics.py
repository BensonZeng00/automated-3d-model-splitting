from pathlib import Path
import sys
import unittest
import json
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split3mf.recognition_metadata import (
    meaningful_semantic_label, require_recognition_semantics,
)
from split3mf.semantic_review_artifacts import (
    ensure_semantic_review_confirmed, render_semantic_review_from_artifacts,
)


class RequiredImageSemanticsTests(unittest.TestCase):
    def test_identifiers_are_not_visual_semantics(self):
        self.assertFalse(meaningful_semantic_label("F004"))
        self.assertFalse(meaningful_semantic_label("待确认区域 P03"))
        self.assertTrue(meaningful_semantic_label("右侧眼睛白色区域"))

    def test_each_part_needs_label_confidence_and_evidence(self):
        item = {"part_index": 7, "visual_semantic_label": "右侧眼睛白色区域",
                "visual_semantic_confidence": "HIGH",
                "visual_semantic_evidence": "位于右眼"}
        require_recognition_semantics([item])
        with self.assertRaisesRegex(ValueError, "P07"):
            require_recognition_semantics([{**item, "visual_semantic_evidence": ""}])

    def test_saved_artifacts_render_without_source_and_confirmation_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, images, output = root / "run", root / "images", root / "review"
            run.mkdir()
            images.mkdir()
            (images / "F001_context_and_zoom.png").write_bytes(b"saved preview")
            (images / "manifest.json").write_text(json.dumps({
                "schema_version": 2, "source": "example.3mf", "source_face_count": 8,
                "noise_max_faces": 100, "small_region_max_faces": 999,
                "items": [{"fragment_id": 1, "source_min_face_index": 0}],
            }), encoding="utf-8")
            summary = {
                "regions": [{"part_index": 1, "color_code": "5C", "faces": 8,
                             "area_mm2": 1.2}],
                "source_region_classifications": [
                    {"fragment_id": 1, "classification": "part"}
                ],
            }
            (run / "03_recognition_summary.json").write_text(json.dumps({
                "stage": "03_recognition_summary", "status": "completed", "result": summary,
            }), encoding="utf-8")
            proposal = {
                "schema": "painted-3mf-semantic-proposals/v1",
                "source_run_dir": str(run), "region_review_dir": str(images),
                "user_confirmed": False,
                "parts": {"P01": {"label": "右眼", "confidence": "HIGH",
                                    "evidence": "面部右侧"}},
                "fragments": {"F001": {"label": "右眼白色片", "confidence": "HIGH",
                                       "evidence": "面部右侧",
                                       "suggested_classification": "part"}},
            }
            proposal_path = root / "proposals.json"
            proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
            paths = render_semantic_review_from_artifacts(run, proposal_path, output)
            self.assertIn("右眼白色片", paths["fragments"].read_text(encoding="utf-8"))
            draft = json.loads(paths["region_decisions"].read_text(encoding="utf-8"))
            self.assertFalse(draft["user_confirmed"])
            self.assertEqual(draft["items"][0]["semantic_label"], "右眼白色片")
            with self.assertRaisesRegex(ValueError, "user confirmation"):
                ensure_semantic_review_confirmed(run, proposal_path, summary)
            proposal["user_confirmed"] = True
            proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
            ensure_semantic_review_confirmed(run, proposal_path, summary)


if __name__ == "__main__":
    unittest.main()
