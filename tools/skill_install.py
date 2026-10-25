"""Reusable installation operations; no model processing or implicit consent."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile

from release_files import GENERATED_DIRECTORIES


SKILL_NAME = "automated-3d-model-splitting"
USAGE_NOTICE = (
    "拆件可能消耗较多 token，复杂模型有可能失败；如果没有 Coding Plan，请谨慎使用，"
    "并对 token 消耗做好预期。明确确认后才安装"
)


@dataclass(frozen=True)
class InstallConsent:
    accepted: bool

    def validate(self) -> None:
        if self.accepted is not True:
            raise ValueError("需要明确确认后才安装 / Explicit consent required.")


@dataclass(frozen=True)
class InstallResult:
    target: Path


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, text=True, encoding="utf-8",
                          capture_output=capture)


def _is_link(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _source_files(source: Path) -> list[Path]:
    required = ("SKILL.md", "requirements.txt", "LICENSE", "NOTICE", "COMMERCIAL_LICENSE.md",
                "scripts/split_painted_3mf.py")
    for name in required:
        if not (source / name).is_file():
            raise ValueError(f"技能文件缺失 / Missing skill file: {source / name}")
    for name in ("scripts", "references", "agents"):
        if not (source / name).is_dir():
            raise ValueError(f"技能目录缺失 / Missing skill directory: {source / name}")
    files = []
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if any(part in GENERATED_DIRECTORIES or part == ".git" for part in relative.parts):
            continue
        if _is_link(path):
            raise ValueError(f"不支持链接文件 / Linked payload is unsupported: {path}")
        if path.is_file() and path.suffix.lower() not in {".pyc", ".pyo", ".log"}:
            files.append(path)
    return files


def install_skill(source: Path, target: Path, python: str,
                  consent: InstallConsent, *, runner=_run) -> InstallResult:
    """Stage and verify, then delete the old Skill and install a clean payload.

    Dependency changes in the selected Python environment cannot be rolled back.
    Consent must be obtained by the caller in the current installation session.
    """
    consent.validate()  # Must precede subprocesses, directory creation and copying.
    if _is_link(source) or _is_link(target):
        raise ValueError("源与目标不能是链接 / Source and destination must not be links.")
    source = source.resolve()
    target = target.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("源目录与目标目录不能重叠 / Source and destination must not overlap.")
    if target.exists() and not target.is_dir():
        raise ValueError(f"安装目标不是目录 / Destination is not a directory: {target}")
    if target.exists() and not (target / "SKILL.md").is_file():
        raise ValueError(f"拒绝替换非技能目录 / Refusing to replace a non-skill directory: {target}")
    files = _source_files(source)
    version = runner([python, "-c", "import json,sys; print(json.dumps(list(sys.version_info[:2])))"],
                     capture=True)
    major, minor = json.loads(version.stdout)
    if major != 3 or not 10 <= minor <= 12:
        raise ValueError("需要 Python 3.10–3.12 / Python 3.10–3.12 required.")
    runner([python, "-m", "pip", "--version"], capture=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".skill-install-", dir=target.parent) as temporary:
        staged = Path(temporary) / SKILL_NAME
        staged.mkdir()
        for path in files:
            destination = staged / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        runner([python, "-m", "pip", "install", "-r", str(staged / "requirements.txt")])
        runner([python, "-c", "import numpy, scipy, trimesh, networkx, manifold3d, matplotlib, PIL"])
        runner([python, str(staged / "scripts" / "split_painted_3mf.py"), "--version"])
        # Recheck the exact resolved target after subprocesses and before deletion.
        if _is_link(target) or target.resolve() != target:
            raise ValueError("安装目标已变化 / Destination changed during installation.")
        if target.exists():
            if not target.is_dir() or not (target / "SKILL.md").is_file():
                raise ValueError(f"拒绝删除非技能目录 / Refusing to delete a non-skill directory: {target}")
            try:
                shutil.rmtree(target)
            except OSError as error:
                raise OSError(f"旧 Skill 删除未完成，请关闭占用后重试 / "
                              f"Old Skill removal incomplete; close users of the directory and retry: {target}") from error
        try:
            staged.rename(target)
        except OSError as error:
            raise OSError(f"新版 Skill 写入失败；未保留旧版，请重新安装 / "
                          f"New Skill placement failed; no old version retained. Reinstall: {target}") from error
    return InstallResult(target=target)
