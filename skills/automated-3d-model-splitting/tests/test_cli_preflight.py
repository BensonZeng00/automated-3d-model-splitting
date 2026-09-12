"""First-use behavior must work before third-party dependencies are installed."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))
from split3mf.common import CORE_DEPENDENCIES, dependency_install_command


class CliPreflightTests(unittest.TestCase):
    def run_without_dependencies(self, *arguments):
        return subprocess.run(
            [sys.executable, "-X", "utf8", "-B", "-S",
             str(SKILL_ROOT / "scripts/split_painted_3mf.py"), *arguments],
            capture_output=True, text=True, encoding="utf-8",
        )

    def test_help_and_version_work_without_site_packages(self):
        for option in ("--help", "--version"):
            with self.subTest(option=option):
                result = self.run_without_dependencies(option)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout)

    def test_missing_dependencies_report_actionable_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "synthetic.3mf"
            source.touch()  # Preflight checks presence/suffix; it must not parse geometry.
            result = self.run_without_dependencies("--input", str(source), "--preflight-only")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        record = json.loads(next(line.removeprefix("preflight=") for line in result.stdout.splitlines()
                                 if line.startswith("preflight=")))
        self.assertTrue(record["input_exists"])
        self.assertFalse(record["dependency_ready"])
        self.assertEqual(set(record["missing_dependencies"]), set(CORE_DEPENDENCIES))
        self.assertIn("matplotlib", record["missing_dependencies"])
        self.assertIn("PIL", record["missing_dependencies"])
        self.assertTrue(record["install_command"])

    def test_install_command_targets_real_requirements_without_user_site(self):
        with patch("sys.platform", "linux"):
            arguments = shlex.split(dependency_install_command())
        self.assertEqual(arguments[:4], [sys.executable, "-m", "pip", "install"])
        self.assertNotIn("--user", arguments)
        requirements = Path(arguments[arguments.index("-r") + 1])
        self.assertEqual(requirements, SKILL_ROOT / "requirements.txt")
        self.assertTrue(requirements.is_file())

    @unittest.skipUnless(sys.platform == "win32", "PowerShell command execution is Windows-specific")
    def test_powershell_install_command_preserves_literal_arguments(self):
        with tempfile.TemporaryDirectory(prefix="split3mf-'literal-") as directory:
            fake_python = Path(directory) / "fake python.ps1"
            fake_python.write_text("ConvertTo-Json -Compress -InputObject @($args)", encoding="utf-8")
            with patch("sys.executable", str(fake_python)):
                command = dependency_install_command()
            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
                capture_output=True, text=True, encoding="utf-8",
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout),
                         ["-m", "pip", "install", "-r", str(SKILL_ROOT / "requirements.txt")])


if __name__ == "__main__":
    unittest.main()
