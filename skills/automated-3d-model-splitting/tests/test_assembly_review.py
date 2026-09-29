import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split3mf.assembly_review import (
    AssemblyReviewRequired, assembly_plan_fingerprint, ensure_assembly_review,
)
from split3mf.resume_interface_stage import resume_interface_stage


def _plan() -> dict:
    return {
        "schema": "contact-interface-plan/v1",
        "recognized_boundary_fingerprint": "boundary-1",
        "interface_count": 1,
        "parts": [{"part": "P01"}, {"part": "P02"}],
        "interfaces": [{
            "interface_id": "I001", "parts": ["P01", "P02"],
            "tenon_part": "P01", "mortise_part": "P02",
            "contact": {"shared_boundary_loops": [{}], "shared_boundary_sample_count": 10},
        }],
    }


def test_stage04_requires_explicit_fingerprint_bound_approval(tmp_path: Path) -> None:
    path = tmp_path / "04_assembly_review.json"
    table_path = tmp_path / "04_interface_table.md"
    plan = _plan()
    with pytest.raises(AssemblyReviewRequired):
        ensure_assembly_review(plan, source_sha256="source-1", template_path=path, table_path=table_path)
    review = json.loads(path.read_text(encoding="utf-8"))
    assert review["plan_fingerprint"] == assembly_plan_fingerprint(plan)
    assert review["user_confirmed"] is False

    review["user_confirmed"] = True
    path.write_text(json.dumps(review), encoding="utf-8")
    assert ensure_assembly_review(
        plan, source_sha256="source-1", template_path=path, table_path=table_path
    )["user_confirmed"] is True

    changed = _plan()
    changed["interfaces"][0]["tenon_part"] = "P02"
    with pytest.raises(ValueError, match="plan_fingerprint"):
        ensure_assembly_review(changed, source_sha256="source-1", template_path=path, table_path=table_path)


def test_correction_requests_block_stage05_even_if_confirmed(tmp_path: Path) -> None:
    path = tmp_path / "04_assembly_review.json"
    plan = _plan()
    with pytest.raises(AssemblyReviewRequired):
        ensure_assembly_review(plan, source_sha256="source-1", template_path=path, table_path=tmp_path / "table.md")
    review = json.loads(path.read_text(encoding="utf-8"))
    review["user_confirmed"] = True
    review["correction_requests"] = [{
        "interface_id": "I001", "issue": "榫卯反了", "requested_change": "交换 P01/P02 榫卯",
    }]
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(AssemblyReviewRequired):
        ensure_assembly_review(plan, source_sha256="source-1", template_path=path, table_path=tmp_path / "table.md")


def test_artifact_resume_cannot_bypass_stage04_review(tmp_path: Path) -> None:
    run_dir = tmp_path / "upstream"
    run_dir.mkdir()
    for name, result in (
        ("02_loaded_project_summary", {"source_sha256": "source-1"}),
        ("03_recognition_summary", {"regions": []}),
        ("04_assembly_plan", _plan()),
    ):
        (run_dir / f"{name}.json").write_text(json.dumps({
            "stage": name, "status": "completed", "result": result,
        }), encoding="utf-8")
    with pytest.raises(AssemblyReviewRequired):
        resume_interface_stage(run_dir, tmp_path / "output")
    assert (run_dir / "04_assembly_review.json").is_file()
