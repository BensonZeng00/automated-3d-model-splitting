#!/usr/bin/env python3
"""Render semantic review tables from saved Stage 03 JSON and images; no 3MF read."""

from __future__ import annotations

import argparse
from pathlib import Path

from split3mf.semantic_review_artifacts import render_semantic_review_from_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Completed Stage 03 artifact directory")
    parser.add_argument("--proposals-json", required=True, type=Path,
                        help="Image-grounded labels, confidence, and evidence for every F/P region")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for review tables; saved stage files are unchanged")
    args = parser.parse_args()
    try:
        paths = render_semantic_review_from_artifacts(
            args.run_dir, args.proposals_json, args.output_dir
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(3, f"Semantic review rendering failed: {exc}\n")
    for kind, path in paths.items():
        print(f"{kind}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
