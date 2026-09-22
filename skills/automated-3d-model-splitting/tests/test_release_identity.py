from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


SKILL_ROOT = Path(__file__).resolve().parents[1]
TECHNICAL_ID = "automated-3d-model-splitting"
VERSION = "2.1.0"


class ReleaseIdentityTests(unittest.TestCase):
    def read(self, relative: str) -> str:
        return (SKILL_ROOT / relative).read_text(encoding="utf-8")

    def test_skill_identity_and_version_are_consistent(self) -> None:
        skill = self.read("SKILL.md")
        self.assertRegex(skill, rf"(?m)^name:\s*{re.escape(TECHNICAL_ID)}\s*$")
        self.assertIn(f"Current release: `{VERSION}`", skill)
        common = self.read("scripts/split3mf/common.py")
        self.assertRegex(common, rf'(?m)^VERSION\s*=\s*["\']{re.escape(VERSION)}["\']\s*$')
        self.assertIn(f"Current release: `{VERSION}`", self.read("references/cli-reference.md"))

    def test_agent_interface_uses_public_identity(self) -> None:
        agent = self.read("agents/openai.yaml")
        self.assertIn('display_name: "自动化3d模型拆件"', agent)
        self.assertIn(f"${TECHNICAL_ID}", agent)

    def test_schema_ids_use_public_identity(self) -> None:
        for filename in (
            "color-map.schema.json",
            "visual-semantics.schema.json",
            "source-region-review.schema.json",
        ):
            schema = json.loads(self.read(f"references/{filename}"))
            self.assertIn(TECHNICAL_ID, schema["$id"])

    def test_requirements_have_compatible_upper_bounds(self) -> None:
        requirements = [
            line.strip()
            for line in self.read("requirements.txt").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertTrue(requirements)
        for requirement in requirements:
            with self.subTest(requirement=requirement):
                self.assertIn(">=", requirement)
                self.assertIn(",<", requirement)


if __name__ == "__main__":
    unittest.main()
