"""Exercise the standalone release payload outside a project checkout."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_files import release_candidate_paths
from run_synthetic_e2e import main as run_synthetic

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills" / "automated-3d-model-splitting"


class InstalledSkillTests(unittest.TestCase):
    def test_standalone_payload_runs_from_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory(prefix="installed skill ") as temporary:
            base = Path(temporary)
            installed = base / "skills with spaces" / SKILL.name
            project = base / "unrelated project"
            project.mkdir()
            for source in release_candidate_paths(ROOT):
                if source.is_relative_to(SKILL):
                    target = installed / source.relative_to(SKILL)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            entry = installed / "scripts" / "split_painted_3mf.py"
            for option in ("--version", "--help"):
                result = subprocess.run(
                    [sys.executable, "-X", "utf8", "-B", "-S", str(entry), option],
                    cwd=project, capture_output=True, text=True, encoding="utf-8",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(run_synthetic(entry_script=entry, working_directory=project), 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "passed")
            self.assertEqual(list(project.iterdir()), [])
