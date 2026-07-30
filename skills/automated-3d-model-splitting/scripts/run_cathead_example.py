"""Run the consent-gated cathead installation smoke test."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen


CATHEAD_SHA256 = "16bd80f486afc7439be7ded2f17229d6c973dd78c27858cac3f714b8a8b48764"
CATHEAD_URL = (
    "https://raw.githubusercontent.com/BensonZeng00/"
    "automated-3d-model-splitting/main/example/cathead.3mf"
)
ENTRY_SCRIPT = Path(__file__).resolve().with_name("split_painted_3mf.py")
REPOSITORY_EXAMPLE = Path(__file__).resolve().parents[3] / "example" / "cathead.3mf"
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_expected_example(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 cathead 示例模型：{path}")
    actual = sha256_file(path)
    if actual != CATHEAD_SHA256:
        raise ValueError(
            "cathead 示例模型 SHA-256 校验失败："
            f"expected={CATHEAD_SHA256}, actual={actual}, path={path}"
        )
    return path.resolve()


def download_example(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.download")
    if temporary.exists():
        temporary.unlink()

    request = Request(CATHEAD_URL, headers={"User-Agent": "automated-3d-model-splitting"})
    received = 0
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > MAX_DOWNLOAD_BYTES:
                raise ValueError("cathead 示例下载大小超过 20 MiB 安全上限")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > MAX_DOWNLOAD_BYTES:
                    raise ValueError("cathead 示例下载大小超过 20 MiB 安全上限")
                output.write(chunk)
        require_expected_example(temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination.resolve()


def resolve_example(explicit_source: Path | None, output_directory: Path) -> Path:
    if explicit_source is not None:
        return require_expected_example(explicit_source)
    if REPOSITORY_EXAMPLE.is_file():
        return require_expected_example(REPOSITORY_EXAMPLE)

    cached = output_directory / "cathead.3mf"
    if cached.is_file():
        return require_expected_example(cached)
    print(f"本地未找到仓库示例，正在下载固定 cathead 文件：{CATHEAD_URL}", flush=True)
    return download_example(cached)


def run_cli(*arguments: str) -> int:
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-B",
        str(ENTRY_SCRIPT),
        *arguments,
    ]
    return subprocess.run(command, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在用户明确同意后运行 cathead 安装测试。"
    )
    parser.add_argument(
        "--accept",
        action="store_true",
        help="确认用户已经明确同意读取并拆分 cathead 示例。",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="可选的本地 cathead.3mf；仍会核对固定 SHA-256。",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path.cwd() / "cathead-example-output",
        help="示例模型缓存和最终装配版 3MF 的输出目录。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.accept:
        print(
            "尚未获得用户同意，未读取或下载 cathead。"
            "请先询问用户，同意后再加 --accept。",
            file=sys.stderr,
        )
        return 2

    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    print("正在复查自动化3d模型拆件依赖……", flush=True)
    preflight_exit = run_cli("--preflight-only")
    if preflight_exit != 0:
        return preflight_exit

    try:
        source = resolve_example(args.source, output_directory)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"cathead 示例准备失败：{exc}", file=sys.stderr, flush=True)
        return 2

    print(f"cathead 示例校验通过：{source}", flush=True)
    print("第一阶段：识别并分类 cathead 零件……", flush=True)
    recognition_exit = run_cli("--input", str(source), "--recognize-only")
    if recognition_exit != 0:
        return recognition_exit

    output = output_directory / "cathead_split_parts.3mf"
    print("第二阶段：拆件、装配并验证 cathead……", flush=True)
    export_exit = run_cli("--input", str(source), "--output", str(output))
    if export_exit != 0:
        return export_exit

    print(f"cathead 安装测试完成：{output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
