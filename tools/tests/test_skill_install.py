"""Isolated installer tests: no real pip calls or user Skill changes."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from install_skill import main, request_consent
from skill_install import InstallConsent, InstallResult, USAGE_NOTICE, install_skill


class ConsentTests(unittest.TestCase):
    def test_explicit_acceptance_required_without_terminal(self):
        with self.assertRaises(ValueError):
            request_consent(False, interactive=False, write=lambda _: None)
        self.assertEqual(request_consent(True, interactive=False, write=lambda _: None),
                         InstallConsent(True))

    def test_interactive_notice_precedes_single_confirmation(self):
        events = []

        def read(prompt):
            events.append(prompt)
            return "同意"

        consent = request_consent(False, interactive=True, read=read, write=events.append)
        self.assertEqual(consent, InstallConsent(True))
        self.assertEqual(events, [USAGE_NOTICE, "[同意 / yes] > "])

    def test_success_repeats_exact_notice_after_installation(self):
        events = []

        def install(*args):
            events.append("INSTALL")
            return InstallResult(Path("installed-skill"))

        with patch("install_skill.sys.stdin.isatty", return_value=False), \
                patch("install_skill.install_skill", side_effect=install), \
                patch("builtins.print", side_effect=lambda value: events.append(value)):
            self.assertEqual(main(["--accept-usage"]), 0)
        self.assertEqual(events[0], USAGE_NOTICE)
        self.assertEqual(events[1], "INSTALL")
        self.assertEqual(events[-1], USAGE_NOTICE)
        self.assertEqual(events.count(USAGE_NOTICE), 2)

    def test_failure_does_not_report_completion_or_repeat_notice(self):
        events = []
        with patch("install_skill.sys.stdin.isatty", return_value=False), \
                patch("install_skill.install_skill", side_effect=OSError("failed")), \
                patch("builtins.print", side_effect=lambda value, **kwargs: events.append(value)):
            self.assertEqual(main(["--accept-usage"]), 1)
        self.assertEqual(events.count(USAGE_NOTICE), 1)
        self.assertFalse(any("安装完成" in event for event in events))

    def test_blank_or_vague_acceptance_cancels(self):
        for answer in ("", "ok", "no"):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                request_consent(False, interactive=True,
                                read=lambda _: answer, write=lambda _: None)

    def test_cli_does_not_call_installer_without_consent(self):
        with patch("install_skill.sys.stdin.isatty", return_value=False), \
                patch("install_skill.install_skill") as install, \
                patch("builtins.print"):
            self.assertEqual(main([]), 2)
            install.assert_not_called()


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="install test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.target = self.root / "user" / "skills" / "test-skill"
        for name in ("SKILL.md", "requirements.txt", "LICENSE", "NOTICE",
                     "COMMERCIAL_LICENSE.md", "scripts/split_painted_3mf.py",
                     "references/guide.md", "agents/config.yaml", "tools/helper.py"):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
        self.calls = []

    def runner(self, command, **kwargs):
        self.calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="[3, 12]")

    def install(self, **kwargs):
        return install_skill(self.source, self.target, "selected-python",
                             InstallConsent(True), runner=kwargs.get("runner", self.runner))

    def test_invalid_consent_has_no_side_effects(self):
        with self.assertRaises(ValueError):
            install_skill(self.source, self.target, "selected-python",
                          InstallConsent(False), runner=self.runner)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.target.parent.exists())

    def test_complete_install_uses_selected_python_and_skips_generated_cache(self):
        cache = self.source / "scripts" / "__pycache__"
        cache.mkdir()
        (cache / "old.pyc").write_bytes(b"cache")
        result = self.install()
        self.assertEqual(result.target, self.target)
        self.assertTrue((self.target / "tools/helper.py").exists())
        self.assertTrue((self.target / "COMMERCIAL_LICENSE.md").exists())
        self.assertFalse((self.target / "scripts/__pycache__").exists())
        self.assertTrue(all(command[0] == "selected-python" for command in self.calls))
        pip = next(command for command in self.calls if "install" in command)
        self.assertEqual(pip[1:5], ["-m", "pip", "install", "-r"])
        self.assertEqual(Path(pip[-1]).name, "requirements.txt")
        self.assertTrue(all("--input" not in command for command in self.calls))

    def existing(self):
        self.target.mkdir(parents=True)
        (self.target / "SKILL.md").write_text("old version", encoding="utf-8")

    def test_existing_install_is_deleted_without_backup_or_stale_files(self):
        self.existing()
        (self.target / "obsolete.py").write_text("incompatible old code")
        self.install()
        self.assertEqual((self.target / "SKILL.md").read_text(), "SKILL.md")
        self.assertFalse((self.target / "obsolete.py").exists())
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])
        self.assertEqual(list(self.target.parent.parent.iterdir()), [self.target.parent])

    def test_dependency_failure_preserves_existing_install_and_cleans_staging(self):
        self.existing()

        def fail(command, **kwargs):
            if "install" in command:
                raise subprocess.CalledProcessError(1, command)
            return self.runner(command, **kwargs)

        with self.assertRaises(subprocess.CalledProcessError):
            self.install(runner=fail)
        self.assertEqual((self.target / "SKILL.md").read_text(), "old version")
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])

    def test_publish_failure_reports_reinstall_and_does_not_restore_old_skill(self):
        self.existing()
        rename = Path.rename

        def fail_staging(path, destination):
            if path.parent.name.startswith(".skill-install-"):
                raise OSError("simulated promotion failure")
            return rename(path, destination)

        with patch.object(Path, "rename", fail_staging), self.assertRaisesRegex(OSError, "Reinstall"):
            self.install()
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.target.parent.iterdir()), [])

    def test_deletion_failure_stops_without_mixing_old_and_new_files(self):
        self.existing()
        import shutil
        remove = shutil.rmtree

        def fail_target(path, *args, **kwargs):
            if Path(path) == self.target:
                raise PermissionError("old Skill is in use")
            return remove(path, *args, **kwargs)

        with patch("skill_install.shutil.rmtree", side_effect=fail_target), \
                self.assertRaisesRegex(OSError, "removal incomplete"):
            self.install()
        self.assertEqual((self.target / "SKILL.md").read_text(), "old version")
        self.assertFalse((self.target / "scripts").exists())
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])

    def test_bad_source_and_overlapping_target_fail_before_pip(self):
        (self.source / "LICENSE").unlink()
        with self.assertRaises(ValueError):
            self.install()
        self.assertEqual(self.calls, [])
        with self.assertRaises(ValueError):
            install_skill(self.source, self.source / "nested", "selected-python",
                          InstallConsent(True), runner=self.runner)
        self.assertEqual(self.calls, [])

    def test_unsupported_python_does_not_copy_or_install_dependencies(self):
        def unsupported(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, stdout="[3, 13]")

        with self.assertRaises(ValueError):
            self.install(runner=unsupported)
        self.assertFalse(self.target.parent.exists())


if __name__ == "__main__":
    unittest.main()
