"""Validate repository structure, release identity, and public-file hygiene."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from synthetic_fixture import write_fixture
from release_files import audit_public_files, release_candidate_paths, skill_fingerprint


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = REPOSITORY_ROOT / "skills" / "automated-3d-model-splitting"
TECHNICAL_ID = "automated-3d-model-splitting"
DISPLAY_NAME = "自动化3d模型拆件"
VERSION = "2.1.0"


def read(relative: str) -> str:
    return (SKILL_ROOT / relative).read_text(encoding="utf-8")


def main() -> int:
    candidates = release_candidate_paths(REPOSITORY_ROOT)
    errors = audit_public_files(REPOSITORY_ROOT, candidates)

    required_root_files = [
        "README.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "NOTICE",
        "COMMERCIAL_LICENSE.md",
        "SECURITY.md",
        "TEST_RESULTS.md",
        ".gitignore",
        ".github/workflows/windows-ci.yml",
    ]
    for relative in required_root_files:
        if not (REPOSITORY_ROOT / relative).is_file():
            errors.append(f"missing repository file: {relative}")

    if not SKILL_ROOT.is_dir():
        errors.append(f"missing Skill directory: {SKILL_ROOT}")
    else:
        for name in ("LICENSE", "NOTICE", "COMMERCIAL_LICENSE.md"):
            root_file = REPOSITORY_ROOT / name
            skill_file = SKILL_ROOT / name
            if not skill_file.is_file():
                errors.append(f"standalone Skill is missing licensing file: {name}")
            elif root_file.is_file() and root_file.read_bytes() != skill_file.read_bytes():
                errors.append(f"standalone Skill licensing file differs from repository: {name}")
        skill_text = read("SKILL.md")
        if not re.search(rf"(?m)^name:\s*{re.escape(TECHNICAL_ID)}\s*$", skill_text):
            errors.append("SKILL.md frontmatter technical ID does not match")
        if f"Current release: `{VERSION}`" not in skill_text:
            errors.append("SKILL.md release version does not match")

        agent_text = read("agents/openai.yaml")
        if f'display_name: "{DISPLAY_NAME}"' not in agent_text:
            errors.append("agents/openai.yaml display name does not match")
        if f"${TECHNICAL_ID}" not in agent_text:
            errors.append("agents/openai.yaml default prompt does not invoke the technical ID")

        common_text = read("scripts/split3mf/common.py")
        if not re.search(rf'(?m)^VERSION\s*=\s*["\']{re.escape(VERSION)}["\']\s*$', common_text):
            errors.append("scripts/split3mf/common.py VERSION does not match")
        if f"Current release: `{VERSION}`" not in read("references/cli-reference.md"):
            errors.append("references/cli-reference.md version does not match")

        for schema_name in ("color-map.schema.json", "visual-semantics.schema.json", "tiny-component-review.schema.json"):
            schema = json.loads(read(f"references/{schema_name}"))
            if TECHNICAL_ID not in str(schema.get("$id", "")):
                errors.append(f"{schema_name} has an unexpected $id")

        requirement_lines = [
            line.strip()
            for line in read("requirements.txt").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        for requirement in requirement_lines:
            if ">=" not in requirement or ",<" not in requirement:
                errors.append(f"dependency is not bounded on both sides: {requirement}")

        entry_script = SKILL_ROOT / "scripts" / "split_painted_3mf.py"
        completed = subprocess.run(
            [sys.executable, "-B", str(entry_script), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if completed.returncode != 0 or VERSION not in completed.stdout:
            errors.append(
                "entry script --version failed: "
                + (completed.stderr.strip() or completed.stdout.strip() or str(completed.returncode))
            )

        with tempfile.TemporaryDirectory(prefix="split3mf-preflight-") as temporary:
            source = write_fixture(Path(temporary) / "synthetic.3mf")
            preflight = subprocess.run(
                [sys.executable, "-X", "utf8", "-B", str(entry_script), "--input", str(source), "--preflight-only"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        if preflight.returncode != 0 or '"dependency_ready": true' not in preflight.stdout:
            errors.append(
                "entry script --preflight-only failed: "
                + (preflight.stderr.strip() or preflight.stdout.strip() or str(preflight.returncode))
            )

    summary = {
        "display_name": DISPLAY_NAME,
        "errors": errors,
        "skill": TECHNICAL_ID,
        "status": "passed" if not errors else "failed",
        "version": VERSION,
        "candidate_file_count": len(candidates),
        "skill_sha256": skill_fingerprint(SKILL_ROOT, candidates),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
