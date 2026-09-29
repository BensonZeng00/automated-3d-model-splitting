"""CLI for painted 3MF recognition and direct pairwise interface assembly."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path

from .common import (
    VERSION,
    load_core_dependencies,
    preflight,
    preflight_failures,
    progress,
    sanitize_name,
)
from .domain import SplitConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize painted 3MF parts and build pairwise tenon/mortise interfaces."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--input", required=True, help="Vendor-painted source .3mf.")
    parser.add_argument("--output", help="Final colored .3mf path.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output 3MF.")
    parser.add_argument("--model-entry", help="Internal .model entry to read.")
    parser.add_argument("--format-profile", choices=["auto", "vendor-paint"], default="auto")
    parser.add_argument("--color-map-json", help="Optional filament color mapping JSON.")
    parser.add_argument("--output-layout", choices=["assembly", "separate-items"], default="assembly")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--recognize-only", action="store_true")
    parser.add_argument("--stage-artifacts-dir", help="Directory for stage JSON/NPZ artifacts.")
    parser.add_argument(
        "--resume-stage05-from", type=Path,
        help="Reuse completed Stage 02/03/04 artifacts for Stage 05 after verifying the source 3MF.",
    )
    parser.add_argument(
        "--stop-after-stage",
        choices=["preflight", "load", "recognize", "assembly", "interface-assembly"],
        help="Stop after writing the selected stage artifacts.",
    )
    parser.add_argument("--interface-scale-ratio", type=float, default=0.50)
    from .common import DEFAULT_AREA_PRIORITY_RATIO
    parser.add_argument("--area-priority-ratio", type=float, default=DEFAULT_AREA_PRIORITY_RATIO,
                        help="Stage 04: prefer the smaller-area tenon at or above this area ratio (default: 3; must be >1).")
    parser.add_argument("--interface-clearance-mm", type=float, default=0.20)
    parser.add_argument("--remove-detached-micro-shells-part", action="append", default=[],
                        metavar="PXX", help="Remove negligible detached shells from this part after assembly; repeat for multiple parts.")
    parser.add_argument("--boundary-review-json", help="Confirmed source boundary ownership JSON.")
    parser.add_argument("--boundary-check-only", action="store_true")
    parser.add_argument("--region-review-json", help="Confirmed small-region classifications JSON.")
    parser.add_argument("--region-review-dir", help="Where source-region review images are written.")
    parser.add_argument("--region-review-resolution", type=int, default=320)
    parser.add_argument("--recognition-review-json", help="Confirmed recognition actions JSON.")
    parser.add_argument("--assembly-review-json", help="Confirmed Stage 04 interface-plan review JSON.")
    parser.add_argument("--semantic-review-json", help="Confirmed saved F/P image-semantic review for Stage 05 resume.")
    parser.add_argument("--noise-review-max-faces", type=int, default=100)
    parser.add_argument("--small-region-review-max-faces", type=int, default=999)
    parser.add_argument(
        "--max-region-review-candidates", type=int, default=10,
        help="Review only the largest source regions by area; classify remaining review candidates as noise (default: 10).",
    )
    parser.add_argument("--exterior-view-count", type=int, default=32)
    parser.add_argument("--exterior-depth-map-resolution", type=int, default=768)
    parser.add_argument("--exterior-depth-tolerance-mm", type=float, default=0.08)
    parser.add_argument("--visual-semantics-json", help="Image-grounded part labels, confidence, and evidence; required after the first recognition preview to proceed to assembly.")
    parser.add_argument("--visual-semantic-min-confidence", default="MED")
    parser.add_argument("--micro-defect-area-mm2", type=float, default=1.0)
    parser.add_argument("--print-surface-tolerance-mm", type=float, default=0.05)
    parser.add_argument("--recovery-dir", help="Persistent boundary-recovery directory.")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not math.isfinite(args.area_priority_ratio) or args.area_priority_ratio <= 1:
        parser.error("--area-priority-ratio must be finite and greater than 1")
    if not math.isfinite(args.interface_scale_ratio) or not 0 < args.interface_scale_ratio <= 1:
        parser.error("--interface-scale-ratio must be greater than 0 and at most 1")
    if not math.isfinite(args.interface_clearance_mm) or args.interface_clearance_mm < 0:
        parser.error("--interface-clearance-mm must be non-negative")
    if args.exterior_view_count < 6 or args.exterior_depth_map_resolution < 64:
        parser.error("exterior recognition needs at least 6 views and 64 pixels per side")
    if not 128 <= args.region_review_resolution <= 1024:
        parser.error("--region-review-resolution must be between 128 and 1024")
    if args.noise_review_max_faces < 0 or args.small_region_review_max_faces < args.noise_review_max_faces:
        parser.error("small-region review threshold must be at least the noise threshold")
    if args.max_region_review_candidates < 1:
        parser.error("--max-region-review-candidates must be at least 1")
    if args.exterior_depth_tolerance_mm < 0:
        parser.error("--exterior-depth-tolerance-mm must be non-negative")
    if args.micro_defect_area_mm2 < 0 or args.print_surface_tolerance_mm < 0:
        parser.error("print tolerances must be non-negative")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    input_path = Path(args.input).expanduser()
    progress("预检", "检查输入文件和 Python 依赖", input=str(input_path))
    checks = preflight(input_path)
    print("preflight=" + json.dumps(checks, ensure_ascii=False), flush=True)
    failures = preflight_failures(checks)
    if failures:
        print("preflight_failures=" + json.dumps(failures, ensure_ascii=False), file=sys.stderr)
        if checks.get("missing_dependencies"):
            print("依赖未就绪，请确认后运行：" + checks["install_command"], file=sys.stderr)
        raise SystemExit(2)
    from .application.stage_artifacts import StageArtifactStore

    output_path = (
        Path(args.output).expanduser() if args.output else
        input_path.resolve().with_name(f"{sanitize_name(input_path.stem) or 'model'}_split_parts.3mf")
    )
    artifact_root = (
        Path(args.stage_artifacts_dir).expanduser() if args.stage_artifacts_dir else
        output_path.with_name(output_path.stem + "_stages")
    )
    args.stage_artifacts_run_id = uuid.uuid4().hex
    artifact_store = StageArtifactStore(artifact_root, run_id=args.stage_artifacts_run_id)
    artifact_store.write_json("01_preflight", {"input": str(input_path.resolve()), "checks": checks})
    print("stage_artifacts=" + str(artifact_store.run_dir), flush=True)
    if args.preflight_only or args.stop_after_stage == "preflight":
        return

    load_core_dependencies()
    from .assembly_review import AssemblyReviewRequired
    if args.resume_stage05_from is not None:
        from .resume_interface_stage import resume_interface_stage

        if output_path.exists() and not args.overwrite:
            parser.error(f"output already exists; pass --overwrite: {output_path}")
        try:
            resumed_run = resume_interface_stage(
                args.resume_stage05_from,
                artifact_root,
                scale_ratio=args.interface_scale_ratio,
                clearance_mm=args.interface_clearance_mm,
                expected_source=input_path,
                assembly_review_path=args.assembly_review_json,
                semantic_review_path=(Path(args.semantic_review_json) if args.semantic_review_json else None),
            )
            completed_3mf = resumed_run / "05_complete_parts.3mf"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_output = output_path.with_name(output_path.name + ".tmp")
            try:
                shutil.copyfile(completed_3mf, temporary_output)
                os.replace(temporary_output, output_path)
            finally:
                temporary_output.unlink(missing_ok=True)
        except AssemblyReviewRequired as exc:
            print("assembly_review_required=" + json.dumps({
                "status": "needs_user_confirmation",
                "review_json": str(exc.review_path),
                "table": str(exc.table_path),
            }, ensure_ascii=False), flush=True)
            raise SystemExit(4) from None
        except (OSError, ValueError, KeyError) as exc:
            parser.exit(3, f"Stage 05 resume failed: {exc}\n")
        print("stage05_artifacts=" + str(resumed_run), flush=True)
        print("output_3mf=" + str(output_path), flush=True)
        return

    from .boundary_review import BoundaryDecisionError, BoundaryReviewRequired
    from .pipeline import SplitPipeline
    from .print_tolerance import PrintTolerance, tolerance_scope

    recovery = Path(args.recovery_dir) if args.recovery_dir else input_path.parent / (input_path.stem + "_split_recovery")
    try:
        with tolerance_scope(PrintTolerance(
            args.micro_defect_area_mm2, args.print_surface_tolerance_mm, recovery, True
        )):
            SplitPipeline(SplitConfig(namespace=args, input_path=input_path, preflight_checks=checks), parser).run()
    except BoundaryReviewRequired as exc:
        print("boundary_review=" + json.dumps({
            "status": "needs_user_confirmation", "directory": str(exc.directory), "report": exc.report
        }, ensure_ascii=False), flush=True)
        raise SystemExit(4) from None
    except BoundaryDecisionError as exc:
        parser.error(str(exc))
    except AssemblyReviewRequired as exc:
        print("assembly_review_required=" + json.dumps({
            "status": "needs_user_confirmation",
            "review_json": str(exc.review_path),
            "table": str(exc.table_path),
        }, ensure_ascii=False), flush=True)
        raise SystemExit(4) from None
    except ValueError as exc:
        print("split_failure=" + json.dumps({
            "error": str(exc), "detail": getattr(exc, "record", {})
        }, ensure_ascii=False), flush=True)
        raise SystemExit(3) from None
