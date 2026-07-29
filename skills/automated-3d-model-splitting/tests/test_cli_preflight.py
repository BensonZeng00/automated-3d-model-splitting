from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


SKILL_ROOT = Path(__file__).resolve().parents[1]
ENTRY_SCRIPT = SKILL_ROOT / "scripts" / "split_painted_3mf.py"


class CliPreflightTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-X", "utf8", "-B", str(ENTRY_SCRIPT), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def test_preflight_only_does_not_require_an_input_model(self) -> None:
        completed = self.run_cli("--preflight-only")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        record = next(
            line.removeprefix("preflight=")
            for line in completed.stdout.splitlines()
            if line.startswith("preflight=")
        )
        checks = json.loads(record)
        self.assertFalse(checks["input_provided"])
        self.assertIsNone(checks["input_exists"])
        self.assertTrue(checks["dependency_ready"])

    def test_normal_execution_still_requires_an_input_model(self) -> None:
        completed = self.run_cli()
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "--input is required unless --preflight-only is used",
            completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
