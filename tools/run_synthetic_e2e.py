"""Run the public synthetic painted-3MF end-to-end release test."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from synthetic_fixture import write_fixture


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = REPOSITORY_ROOT / "skills" / "automated-3d-model-splitting"
ENTRY_SCRIPT = SKILL_ROOT / "scripts" / "split_painted_3mf.py"
def main(*, entry_script: Path = ENTRY_SCRIPT, working_directory: Path | None = None) -> int:
    with tempfile.TemporaryDirectory(prefix="split3mf-release-") as temporary:
        directory = Path(temporary)
        source = write_fixture(directory / "synthetic-painted-cube.3mf")
        output = directory / "synthetic-painted-cube-assembly.3mf"
        command = [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            str(entry_script),
            "--input",
            str(source),
            "--output",
            str(output),
            "--noise-review-max-faces",
            "0",
            "--small-region-review-max-faces",
            "0",
            "--exterior-view-count",
            "8",
            "--exterior-depth-map-resolution",
            "64",
            "--overwrite",
        ]
        completed = subprocess.run(
            command,
            cwd=working_directory,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 4:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            raise SystemExit(f"expected recognition review exit code 4, got {completed.returncode}")
        review = json.loads(completed.stdout.splitlines()[-1])
        if review.get("status") != "review_required" or not review.get("review_json"):
            raise AssertionError("synthetic run did not request recognition review")
        review_path = Path(review["review_json"])
        summary_path = review_path.with_name("03_recognition_summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        regions = summary["result"]["regions"]
        if len(regions) != 2 or sum(int(item["faces"]) for item in regions) != 768:
            raise AssertionError("synthetic recognition did not preserve two regions and 768 faces")
        if not review_path.is_file() or output.exists():
            raise AssertionError("recognition review must exist before final output")
        result = {"status": "passed", "stage": "recognition_review_required",
                  "source_faces": 768, "recognized_parts": len(regions)}
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
