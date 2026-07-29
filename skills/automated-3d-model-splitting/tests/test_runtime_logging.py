from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf.reporting import machine_record, runtime_log


class RuntimeLoggingTests(unittest.TestCase):
    def test_nonblocking_machine_record_uses_stdout_only(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        payload = {"parts": ["P09"], "blocking": False}

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            machine_record("ratio_accepted_topology_findings", payload)

        self.assertEqual(stderr.getvalue(), "")
        line = stdout.getvalue().strip()
        self.assertTrue(line.startswith("ratio_accepted_topology_findings="))
        self.assertEqual(json.loads(line.split("=", 1)[1]), payload)

    def test_blocking_machine_record_uses_stderr_only(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        payload = {"parts": ["P09"]}

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            machine_record("validation_failures", payload, blocking=True)

        self.assertEqual(stdout.getvalue(), "")
        line = stderr.getvalue().strip()
        self.assertTrue(line.startswith("validation_failures="))
        self.assertEqual(json.loads(line.split("=", 1)[1]), payload)

    def test_runtime_log_emits_readable_and_machine_readable_lines(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runtime_log(
                "递归输入",
                "recursive_input_reload_done",
                "上阶段独立 3MF 已成为本阶段输入",
                step_order=2,
                input_3mf="parent-output.3mf",
                cumulative_3mf_is_input=False,
            )

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("[递归输入]"))
        self.assertIn("parent-output.3mf", lines[0])
        self.assertTrue(lines[1].startswith("runtime_step="))

        record = json.loads(lines[1].split("=", 1)[1])
        self.assertEqual(record["stage"], "递归输入")
        self.assertEqual(record["event"], "recursive_input_reload_done")
        self.assertEqual(record["step_order"], 2)
        self.assertEqual(record["input_3mf"], "parent-output.3mf")
        self.assertFalse(record["cumulative_3mf_is_input"])
        self.assertGreaterEqual(record["elapsed_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
