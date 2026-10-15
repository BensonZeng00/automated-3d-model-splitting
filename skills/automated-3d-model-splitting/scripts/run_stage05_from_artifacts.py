#!/usr/bin/env python3
"""Resume Stage 05 directly from a completed 02/03/04 artifact run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from split3mf.common import load_core_dependencies
from split3mf.resume_interface_stage import resume_interface_stage
from split3mf.assembly_review import AssemblyReviewRequired


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build complete Stage 05 mortise-and-tenon parts from completed Stage 02/03/04 artifacts; "
            "the source 3MF is not reread and boundaries are not recomputed."
        )
    )
    parser.add_argument(
        "--source-run-dir", required=True, type=Path,
        help="Run directory containing completed 02, 03, and 04 artifacts.",
    )
    parser.add_argument(
        "--input", type=Path,
        help="Optional source 3MF to verify against the Stage 02 SHA-256 without parsing it.",
    )
    parser.add_argument(
        "--assembly-review-json", type=Path,
        help="Confirmed review JSON for the exact Stage 04 interface plan.",
    )
    parser.add_argument(
        "--semantic-review-json", type=Path,
        help="Confirmed image-semantic review for every saved F/P region.",
    )
    parser.add_argument(
        "--output-root", required=True, type=Path,
        help="Root directory for the new Stage 05 artifact run.",
    )
    parser.add_argument(
        "--interface-scale-ratio", type=float, default=0.50,
        help="Elliptical inner profile size relative to the contact boundary (default: 0.50).",
    )
    parser.add_argument(
        "--interface-clearance-mm", type=float, default=0.20,
        help="Additional mortise side and floor clearance in mm (default: 0.20).",
    )
    parser.add_argument(
        "--remove-detached-micro-shells-part", action="append", default=[],
        metavar="PXX", help="Remove negligible detached shells from this part after assembly; repeatable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 < args.interface_scale_ratio <= 1.0:
        parser.error("--interface-scale-ratio must be greater than 0 and at most 1")
    if args.interface_clearance_mm < 0.0:
        parser.error("--interface-clearance-mm must be non-negative")
    load_core_dependencies()
    try:
        run_dir = resume_interface_stage(
            args.source_run_dir,
            args.output_root,
            scale_ratio=args.interface_scale_ratio,
            clearance_mm=args.interface_clearance_mm,
            expected_source=args.input,
            assembly_review_path=args.assembly_review_json,
            semantic_review_path=args.semantic_review_json,
            remove_detached_micro_shells_part_ids=frozenset(args.remove_detached_micro_shells_part),
        )
    except AssemblyReviewRequired as exc:
        parser.exit(4, f"Stage 04 review required: {exc.review_path}\nTable: {exc.table_path}\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(3, f"Stage 05 resume failed: {exc}\n")
    print(f"stage05_artifacts={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
