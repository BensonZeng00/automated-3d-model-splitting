from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split3mf.stage_table_report import (
    interface_table, markdown_table, publish_stage_tables, recognition_table,
)


def test_stage_tables_are_readable_and_keep_source_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    recognition = {
        "regions": [{"part_index": 1, "visual_semantic_label": "左眉", "color_code": "0C",
                     "color_hex": "#312121", "faces": 42, "area_mm2": 1.25}],
        "recognized_boundaries": {"loop_vertex_counts": [[12]]},
    }
    plan = {
        "parts": [{"part": "P01", "visual_semantic_label": "左眉"},
                  {"part": "P02", "visual_semantic_label": "头部"}],
        "interfaces": [{"interface_id": "I001", "tenon_part": "P01", "mortise_part": "P02",
                        "contact": {"shared_boundary_loops": [{}], "shared_boundary_sample_count": 12},
                        "direction_evidence": {"method": "inward_alignment"}}],
    }
    for stage, value in (("03_recognition_summary", recognition), ("04_assembly_plan", plan)):
        (run_dir / f"{stage}.json").write_text(
            json.dumps({"stage": stage, "status": "completed", "result": value}), encoding="utf-8"
        )

    paths = publish_stage_tables(run_dir)

    assert "| P01 | 左眉 | 0C | #312121 | 42 | 1.25 | 1 |" in paths["03_recognition_table"].read_text(encoding="utf-8")
    assert "| I001 | P01 左眉 | P02 头部 | 1 | 12 | 内向法线 |" in paths["04_interface_table"].read_text(encoding="utf-8")
    assert json.loads((run_dir / "03_recognition_table.json").read_text(encoding="utf-8"))["result"] == recognition_table(recognition)
    assert json.loads((run_dir / "04_interface_table.json").read_text(encoding="utf-8"))["result"] == interface_table(plan)
    assert "region_table" not in json.loads((run_dir / "03_recognition_summary.json").read_text(encoding="utf-8"))["result"]
    assert "\\|" in markdown_table("x", {"columns": ["A"], "rows": [["a|b"]]})
