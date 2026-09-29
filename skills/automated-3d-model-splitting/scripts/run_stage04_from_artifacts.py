#!/usr/bin/env python3
"""Build a Stage 04 interface plan from saved, confirmed Stage 02/03 artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from split3mf.common import load_core_dependencies
from split3mf.resume_assembly_stage import resume_assembly_stage
from split3mf.contact_interface_planner import DEFAULT_AREA_PRIORITY_RATIO


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plan Stage 04 interfaces from saved Stage 02/03 artifacts. "
            "The source 3MF is not parsed again and frozen boundaries are reused."
        )
    )
    parser.add_argument("--source-run-dir", required=True, type=Path,
                        help="Run directory with completed, confirmed Stage 02/03 artifacts.")
    parser.add_argument("--input", type=Path,
                        help="Optional source 3MF; only its path and SHA-256 are checked.")
    parser.add_argument("--visual-semantics-json", type=Path,
                        help="Image-grounded labels, confidence, and evidence for saved P regions.")
    parser.add_argument("--recognition-review-json", type=Path,
                        help="Confirmed recognition review matching the saved Stage 03 fingerprint.")
    parser.add_argument("--assembly-review-json", type=Path,
                        help="Prior assembly review used only for explicit direction-policy corrections.")
    parser.add_argument("--replace-unconfirmed-plan", action="store_true",
                        help="Refresh an existing pending Stage 04 plan; confirmed plans and recorded corrections are protected.")
    parser.add_argument("--area-priority-ratio", type=float, default=DEFAULT_AREA_PRIORITY_RATIO,
                        help="Prefer smaller-area tenon at or above this ratio (default: 3; must be >1).")
    args = parser.parse_args(argv)
    import math
    if not math.isfinite(args.area_priority_ratio) or args.area_priority_ratio <= 1:
        parser.error("--area-priority-ratio must be finite and greater than 1")
    load_core_dependencies()
    try:
        run_dir = resume_assembly_stage(
            args.source_run_dir,
            expected_source=args.input,
            visual_semantics_path=args.visual_semantics_json,
            recognition_review_path=args.recognition_review_json,
            prior_assembly_review_path=args.assembly_review_json,
            replace_unconfirmed_plan=args.replace_unconfirmed_plan,
            area_priority_ratio=args.area_priority_ratio,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(3, f"Stage 04 resume failed: {exc}\n")
    print(f"stage04_artifacts={run_dir}")
    status_path = run_dir / "04_assembly_review_status.json"
    if status_path.is_file():
        import json

        status = json.loads(status_path.read_text(encoding="utf-8")).get("result", {})
        if status.get("status") == "needs_user_confirmation":
            print(f"assembly_review={status.get('review_json')}")
            print(f"interface_table={status.get('table')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
