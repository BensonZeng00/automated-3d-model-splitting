"""Inventory and audit the repository files that may enter a public release."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess


GENERATED_DIRECTORIES = {
    "outputs", "artifacts", "test-output", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".venv", "venv", "env", "build", "dist",
}
MODEL_SUFFIXES = {".3mf", ".stl", ".obj", ".glb", ".gltf", ".blend", ".blend1", ".npz", ".npy"}
# Existing public example and the two maintained geometry regression fixtures.
PUBLIC_MODEL_PATHS = {
    "example/cathead.3mf",
    "skills/automated-3d-model-splitting/tests/fixtures/concave_strip_fold.npz",
    "skills/automated-3d-model-splitting/tests/fixtures/source_projection_fold.npz",
}
LOCAL_PATH = re.compile(r"(?i)\b[a-z]:[\\/]+|/(?:Users|home)/[^/\s\"'<>]+")
FORBIDDEN_FRAGMENTS = (".skill_" "work", "codex-" "clipboard", "大熊" "猫", "split-painted" "-3mf")


def release_candidate_paths(root: Path) -> list[Path]:
    """Include tracked files even when ignored, plus non-ignored untracked files."""
    root = root.resolve()
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            capture_output=True,
        )
    except FileNotFoundError:
        completed = None
    if completed is not None and completed.returncode == 0:
        candidates = {root / raw.decode("utf-8") for raw in completed.stdout.split(b"\0") if raw}
        return sorted(path for path in candidates if path.is_file())
    # A source archive has no Git index: inspect its full payload, not just skills/.
    return sorted(path for path in root.rglob("*") if path.is_file() and ".git" not in path.relative_to(root).parts)


def audit_public_files(root: Path, candidates: list[Path]) -> list[str]:
    errors = []
    for path in candidates:
        relative = path.relative_to(root)
        name = relative.as_posix()
        if path.is_symlink():
            errors.append(f"symlink requires explicit packaging review: {name}")
            continue
        if any(part in GENERATED_DIRECTORIES for part in relative.parts) or path.suffix.lower() in {".pyc", ".pyo", ".log"}:
            errors.append(f"generated artifact is packaged: {name}")
            continue
        if path.suffix.lower() in MODEL_SUFFIXES and name not in PUBLIC_MODEL_PATHS:
            errors.append(f"unapproved model artifact is packaged: {name}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if LOCAL_PATH.search(content):
            errors.append(f"machine-specific absolute path in {name}")
        for fragment in FORBIDDEN_FRAGMENTS:
            if fragment in content:
                errors.append(f"forbidden fragment {fragment!r} in {name}")
    return errors


def skill_fingerprint(skill_root: Path, candidates: list[Path]) -> str:
    """SHA-256 of sorted relative paths, NUL, file SHA-256 hex, and newline."""
    digest = hashlib.sha256()
    files = sorted((path for path in candidates if path.is_relative_to(skill_root)),
                   key=lambda path: path.relative_to(skill_root).as_posix())
    for path in files:
        relative = path.relative_to(skill_root).as_posix()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{relative}\0{file_hash}\n".encode("utf-8"))
    return digest.hexdigest()
