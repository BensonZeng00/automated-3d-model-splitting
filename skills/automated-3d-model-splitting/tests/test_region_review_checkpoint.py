"""A confirmed F review resumes recognition from the saved pre-review work."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[3]
ENTRY = REPO / "skills/automated-3d-model-splitting/scripts/split_painted_3mf.py"
sys.path.insert(0, str(REPO / "tools"))
from synthetic_fixture import write_fixture


def _run(command: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "TEMP": str(root), "TMP": str(root), "MPLCONFIGDIR": str(root / "mpl")}
    return subprocess.run(command, capture_output=True, text=True, env=env, cwd=REPO)


def _run_dir(result: subprocess.CompletedProcess[str]) -> Path:
    return Path(next(line.split("=", 1)[1] for line in result.stdout.splitlines()
                     if line.startswith("stage_artifacts=")))


def _result(run_dir: Path, name: str) -> dict:
    return json.loads((run_dir / f"{name}.json").read_text(encoding="utf-8"))["result"]


def test_region_review_resume_matches_fresh_recognition(tmp_path: Path) -> None:
    source = write_fixture(tmp_path / "painted.3mf")
    base = [
        sys.executable, str(ENTRY), "--input", str(source),
        "--output", str(tmp_path / "out.3mf"),
        "--stage-artifacts-dir", str(tmp_path / "stages"),
        "--exterior-view-count", "8", "--exterior-depth-map-resolution", "64",
        "--stop-after-stage", "recognize",
    ]
    first = _run(base, tmp_path)
    assert first.returncode == 4, first.stdout[-1000:] + first.stderr[-1000:]
    initial_run = _run_dir(first)
    assert (initial_run / "03_region_review_checkpoint.npz").is_file()
    review_path = tmp_path / "painted_region_review/user_decisions.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    for item in review["items"]:
        item.update(classification="part", semantic_label="painted cube surface",
                    visual_confidence="HIGH")
    review["user_confirmed"] = True
    review_path.write_text(json.dumps(review), encoding="utf-8")

    resumed = _run(base + ["--region-review-json", str(review_path),
                           "--resume-region-review-from", str(initial_run)], tmp_path)
    assert resumed.returncode == 0, resumed.stdout[-1000:] + resumed.stderr[-1000:]
    assert "source_stage02_reused" in resumed.stdout

    fresh = _run(base + ["--region-review-json", str(review_path)], tmp_path)
    assert fresh.returncode == 0, fresh.stdout[-1000:] + fresh.stderr[-1000:]
    resumed_summary = _result(_run_dir(resumed), "03_recognition_summary")
    fresh_summary = _result(_run_dir(fresh), "03_recognition_summary")
    assert resumed_summary["regions"] == fresh_summary["regions"]
    assert (resumed_summary["recognized_boundaries"]["fingerprint"]
            == fresh_summary["recognized_boundaries"]["fingerprint"])

    legacy_run = tmp_path / "legacy" / initial_run.name
    shutil.copytree(initial_run, legacy_run)
    for path in legacy_run.glob("03_region_review_checkpoint*"):
        path.unlink()
    legacy = _run(base + ["--region-review-json", str(review_path),
                          "--resume-region-review-from", str(legacy_run)], tmp_path)
    assert legacy.returncode == 0, legacy.stdout[-1000:] + legacy.stderr[-1000:]
    assert "source_stage02_reused_candidates_recomputed" in legacy.stdout
    legacy_summary = _result(_run_dir(legacy), "03_recognition_summary")
    assert legacy_summary["regions"] == fresh_summary["regions"]
    assert (legacy_summary["recognized_boundaries"]["fingerprint"]
            == fresh_summary["recognized_boundaries"]["fingerprint"])

    changed_options = _run(base + ["--region-review-json", str(review_path),
                                   "--resume-region-review-from", str(initial_run),
                                   "--max-region-review-candidates", "5"], tmp_path)
    assert changed_options.returncode == 2
    assert "checkpoint options changed" in changed_options.stderr

    with (initial_run / "03_region_review_checkpoint.npz").open("ab") as checkpoint_file:
        checkpoint_file.write(b"tampered")
    changed_artifact = _run(base + ["--region-review-json", str(review_path),
                                    "--resume-region-review-from", str(initial_run)], tmp_path)
    assert changed_artifact.returncode == 2
    assert "checksum mismatch" in changed_artifact.stderr
