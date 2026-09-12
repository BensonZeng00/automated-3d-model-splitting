"""Exercise release guards against real temporary Git indexes and source archives."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_files import audit_public_files, release_candidate_paths, skill_fingerprint


class ReleaseFilesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def write(self, name, content="example"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def git(self, *arguments):
        subprocess.run(["git", "-C", str(self.root), *arguments], check=True, capture_output=True)

    def test_inventory_covers_root_and_tools_and_omits_untracked_outputs(self):
        self.git("init", "--quiet")
        self.write(".gitignore", "outputs/\n")
        readme = self.write("README.md")
        tool = self.write("tools/helper.py")
        output = self.write("outputs/cache.json")
        candidates = release_candidate_paths(self.root)
        self.assertIn(readme, candidates)
        self.assertIn(tool, candidates)
        self.assertNotIn(output, candidates)

    def test_tracked_outputs_are_rejected_even_if_gitignored(self):
        self.git("init", "--quiet")
        self.write(".gitignore", "outputs/\n")
        output = self.write("outputs/cache.json")
        self.git("add", "--force", "outputs/cache.json")
        candidates = release_candidate_paths(self.root)
        self.assertIn(output, candidates)
        self.assertTrue(audit_public_files(self.root, candidates))

    def test_paths_outside_skill_are_checked_in_text_and_escaped_json(self):
        windows = "/".join(("C:", "Users", "tester", "model"))
        escaped = windows.replace("/", "\\" * 2)
        unix = "/".join(("", "home", "tester", "model"))
        for index, content in enumerate((windows, escaped, unix)):
            with self.subTest(content=content):
                path = self.write(f"notes{index}.md", content)
                self.assertTrue(audit_public_files(self.root, [path]))

    def test_only_named_public_models_and_fixtures_are_allowed(self):
        approved = self.write("example/cathead.3mf")
        fixture = self.write("skills/automated-3d-model-splitting/tests/fixtures/source_projection_fold.npz")
        self.assertEqual(audit_public_files(self.root, [approved, fixture]), [])
        for name in ("example/private.3mf", "tools/snapshot.npz", "notes/model.npy"):
            with self.subTest(name=name):
                self.assertTrue(audit_public_files(self.root, [self.write(name)]))

    def test_archive_without_git_still_checks_repository_root(self):
        path = self.write("outputs/snapshot.json")
        with patch("release_files.subprocess.run", side_effect=FileNotFoundError):
            candidates = release_candidate_paths(self.root)
        self.assertIn(path, candidates)
        self.assertTrue(audit_public_files(self.root, candidates))

    def test_fingerprint_is_portable_and_changes_with_content_or_name(self):
        skill = self.root / "skills/demo"
        first = self.write("skills/demo/SKILL.md", "instructions")
        other = self.write("README.md")
        initial = skill_fingerprint(skill, [first, other])
        self.assertEqual(initial, skill_fingerprint(skill, [other, first]))
        first.write_text("changed", encoding="utf-8")
        self.assertNotEqual(initial, skill_fingerprint(skill, [first, other]))
        first.write_text("instructions", encoding="utf-8")
        renamed = first.rename(first.with_name("different.md"))
        self.assertNotEqual(initial, skill_fingerprint(skill, [renamed, other]))


if __name__ == "__main__":
    unittest.main()
