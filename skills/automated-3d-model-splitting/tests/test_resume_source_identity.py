import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split3mf.common import load_core_dependencies

load_core_dependencies()
from split3mf.resume_interface_stage import resume_interface_stage
from split3mf.stage_reuse_provenance import provenance_record, validate_provenance


def _write_stage02_summary(run_dir: Path, source: Path, digest: str) -> None:
    (run_dir / "02_loaded_project_summary.json").write_text(
        json.dumps({
            "stage": "02_loaded_project_summary",
            "status": "completed",
            "result": {"source": str(source), "source_sha256": digest},
        }),
        encoding="utf-8",
    )


def test_resume_rejects_changed_source_before_loading_arrays(tmp_path: Path) -> None:
    source = tmp_path / "source.3mf"
    source.write_bytes(b"original")
    run_dir = tmp_path / "stages"
    run_dir.mkdir()
    _write_stage02_summary(run_dir, source, hashlib.sha256(b"original").hexdigest())
    source.write_bytes(b"changed")

    with pytest.raises(ValueError, match="source 3MF changed"):
        resume_interface_stage(run_dir, tmp_path / "output", expected_source=source)


def test_resume_rejects_artifacts_from_another_source(tmp_path: Path) -> None:
    source = tmp_path / "source.3mf"
    other = tmp_path / "other.3mf"
    source.write_bytes(b"same")
    other.write_bytes(b"same")
    run_dir = tmp_path / "stages"
    run_dir.mkdir()
    _write_stage02_summary(run_dir, source, hashlib.sha256(b"same").hexdigest())

    with pytest.raises(ValueError, match="stage artifacts belong to"):
        resume_interface_stage(run_dir, tmp_path / "output", expected_source=other)


def test_provenance_detects_changed_parameters_and_source() -> None:
    record = provenance_record(object(), "source-digest")
    validate_provenance(record, "source-digest")
    changed = dict(record)
    changed["upstream_options"] = dict(record["upstream_options"], model_entry="changed")
    with pytest.raises(ValueError, match="options hash mismatch"):
        validate_provenance(changed, "source-digest")
    with pytest.raises(ValueError, match="source hash mismatch"):
        validate_provenance(record, "another-digest")
