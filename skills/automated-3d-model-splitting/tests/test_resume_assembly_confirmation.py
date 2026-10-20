import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split3mf.application.recognition_review import result_fingerprint
from split3mf.application.stage_artifact_reader import read_stage_json
from split3mf.common import Component
from split3mf.resume_assembly_stage import _confirm_saved_recognition


def _case(tmp_path: Path):
    run_dir = tmp_path / "saved_run"
    run_dir.mkdir()
    source = tmp_path / "source.3mf"
    source.write_bytes(b"source")
    component = Component(
        color_code="A", global_faces=np.array([0]), face_count=1, area=1.0,
        bbox_min=np.zeros(3), bbox_max=np.ones(3), center=np.zeros(3),
    )
    boundaries = SimpleNamespace(fingerprint="frozen-boundary")
    fingerprint = result_fingerprint([component], boundaries)
    status = {
        "status": "needs_user_confirmation",
        "recognition_fingerprint": "original-regions",
        "result_fingerprint": fingerprint,
        "actions": [],
    }
    review = {
        "schema_version": 1,
        "source": str(source),
        "source_face_count": 1,
        "recognition_fingerprint": "original-regions",
        "expected_result_fingerprint": fingerprint,
        "actions": [],
        "user_confirmed": True,
    }
    review_path = run_dir / "03_recognition_review.json"
    return run_dir, source, component, boundaries, status, review, review_path


def test_pending_recognition_confirmation_is_recorded_from_saved_result(tmp_path: Path) -> None:
    run_dir, source, component, boundaries, status, review, review_path = _case(tmp_path)
    review_path.write_text(json.dumps(review), encoding="utf-8")

    _confirm_saved_recognition(run_dir, source, 1, status, review_path, [component], boundaries)

    saved = read_stage_json(run_dir, "03_recognition_review_status")
    assert saved["status"] == "user_confirmed"
    assert saved["result_fingerprint"] == review["expected_result_fingerprint"]
    assert saved["decision_file"] == str(review_path.resolve())


@pytest.mark.parametrize("changed", ["expected_result_fingerprint", "actions", "source"])
def test_stale_review_does_not_record_confirmation(tmp_path: Path, changed: str) -> None:
    run_dir, source, component, boundaries, status, review, review_path = _case(tmp_path)
    review[changed] = ["changed"] if changed == "actions" else "changed"
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError):
        _confirm_saved_recognition(run_dir, source, 1, status, review_path, [component], boundaries)

    assert not (run_dir / "03_recognition_review_status.json").exists()
    assert status["status"] == "needs_user_confirmation"


def test_saved_result_mismatch_blocks_even_previously_confirmed_status(tmp_path: Path) -> None:
    run_dir, source, component, boundaries, status, _, _ = _case(tmp_path)
    status["status"] = "user_confirmed"
    status["result_fingerprint"] = "stale"

    with pytest.raises(ValueError, match="does not match its artifacts"):
        _confirm_saved_recognition(run_dir, source, 1, status, None, [component], boundaries)
