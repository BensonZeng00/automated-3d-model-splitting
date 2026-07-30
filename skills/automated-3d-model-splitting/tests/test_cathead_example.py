from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


SKILL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SKILL_ROOT.parents[1]
RUNNER_PATH = SKILL_ROOT / "scripts" / "run_cathead_example.py"
EXAMPLE_PATH = REPOSITORY_ROOT / "example" / "cathead.3mf"
EXPECTED_SHA256 = "16bd80f486afc7439be7ded2f17229d6c973dd78c27858cac3f714b8a8b48764"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_cathead_example", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load cathead example runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CatheadExampleTests(unittest.TestCase):
    def test_repository_example_matches_pinned_hash(self) -> None:
        self.assertTrue(EXAMPLE_PATH.is_file())
        self.assertEqual(hashlib.sha256(EXAMPLE_PATH.read_bytes()).hexdigest(), EXPECTED_SHA256)

    def test_runner_refuses_to_touch_example_without_consent_flag(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-X", "utf8", "-B", str(RUNNER_PATH)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("尚未获得用户同意", completed.stderr)

    def test_runner_uses_repository_example_when_available(self) -> None:
        runner = load_runner()
        resolved = runner.resolve_example(None, REPOSITORY_ROOT / "unused-output")
        self.assertEqual(resolved, EXAMPLE_PATH.resolve())

    def test_skill_documents_post_install_consent_flow(self) -> None:
        skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("python scripts/run_cathead_example.py --accept", skill)
        self.assertIn(EXPECTED_SHA256, skill)
        self.assertIn("Do not download, read, recognize, or split", skill)


if __name__ == "__main__":
    unittest.main()
