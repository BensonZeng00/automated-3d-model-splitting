#!/usr/bin/env python3
"""Render completed Stage 03/04 JSON as readable Markdown tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from split3mf.stage_table_report import publish_stage_tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Completed run directory containing 03 and 04 stage JSON.")
    args = parser.parse_args(argv)
    try:
        paths = publish_stage_tables(args.run_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(3, f"Stage table rendering failed: {exc}\n")
    for stage, path in paths.items():
        print(f"{stage}={path}")
        print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
