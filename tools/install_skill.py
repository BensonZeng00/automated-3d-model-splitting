"""Install the checkout's complete Skill after informed user consent."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from skill_install import InstallConsent, SKILL_NAME, USAGE_NOTICE, install_skill


def request_consent(plan: str | None, accepted: bool, *, interactive: bool,
                    read=input, write=print) -> InstallConsent:
    write(USAGE_NOTICE)
    if plan is None and interactive:
        answer = read("是否有 Coding Plan（编程助手订阅或额度套餐）？"
                      "有/没有/不确定 [yes/no/unsure]: ").strip().lower()
        plan = {"有": "yes", "没有": "no", "不确定": "unsure"}.get(answer, answer)
    if plan not in {"yes", "no", "unsure"}:
        raise ValueError("请先回答套餐情况 / Answer yes, no or unsure before installation.")
    if not accepted and interactive:
        answer = read("已了解上述消耗，是否继续安装？输入“同意”或 yes 确认，"
                      "其他输入取消 / Accept usage and install? [yes/NO]: ").strip().lower()
        accepted = answer in {"同意", "yes"}
    consent = InstallConsent(plan, accepted)
    consent.validate()
    return consent


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Install the complete 3MF splitting Skill with a mandatory usage-consent gate.",
        epilog="Exit codes: 0 success; 1 installation failure; 2 input/consent missing; 130 cancelled. "
               "No sample model is run. Dependency changes cannot be automatically rolled back.",
    )
    result.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "skills" / SKILL_NAME,
                        help="Complete Skill source directory (default: this checkout's skills directory).")
    codex_dir = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    result.add_argument("--target", type=Path, default=codex_dir / "skills" / SKILL_NAME,
                        help="Destination Skill directory; after validation the old Skill is deleted, "
                             "then the new version is installed without backup or rollback.")
    result.add_argument("--python", default=sys.executable,
                        help="Python 3.10–3.12 executable (default: current interpreter).")
    result.add_argument("--plan", choices=("yes", "no", "unsure"),
                        help="User's actual Coding Plan answer from this installation session.")
    result.add_argument("--accept-usage", action="store_true",
                        help="Use ONLY after the user explicitly accepts the notice in this session; "
                             "does not replace --plan. Otherwise prompt interactively.")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        consent = request_consent(args.plan, args.accept_usage, interactive=sys.stdin.isatty())
    except (ValueError, EOFError) as error:
        print(f"未安装 / Not installed: {error}", file=sys.stderr)
        return 2
    try:
        result = install_skill(args.source, args.target, args.python, consent)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"安装失败 / Installation failed: {error}\n"
              "如 pip 已执行，所选 Python 环境可能已更改；请检查依赖。"
              " / If pip ran, dependency changes may remain in the selected environment.", file=sys.stderr)
        return 1
    print(f"已安装 / Installed: {result.target}")
    print("安装完成。可另行选择是否试用 cathead.3mf。 / You can separately choose to try cathead.3mf.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("已取消 / Cancelled.", file=sys.stderr)
        raise SystemExit(130)
