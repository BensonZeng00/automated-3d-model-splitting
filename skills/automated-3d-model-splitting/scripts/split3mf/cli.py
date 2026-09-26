from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from pathlib import Path

from .common import *
from .domain import SplitConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recognize externally painted parts, plan pairwise tenon/mortise interfaces, "
            "and construct matching interfaces from the Stage 04 boundary relations."
        )
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument('--micro-defect-area-mm2', type=float, default=1.0)
    parser.add_argument('--print-surface-tolerance-mm', type=float, default=0.05)
    parser.add_argument('--boundary-shape', type=str.lower, choices=['smooth'], default='smooth', help='Smooth shared cut boundaries; source mode has been removed.')
    parser.add_argument('--hidden-surface-refinement', choices=['preserve', 'refine'],
                        default='preserve', help='Preserve audited hidden annuli or request strict density refinement.')
    parser.add_argument('--recovery-dir', help='Persistent inputs and candidates for local failure replay')
    parser.add_argument(
        "--stage-artifacts-dir",
        default=None,
        help="Write inspectable JSON/NPZ results after each completed application stage.",
    )
    parser.add_argument(
        "--stop-after-stage",
        choices=["preflight", "load", "recognize", "assembly", "interface-assembly"],
        default=None,
        help="Stop after a completed stage artifact is written, for stepwise inspection.",
    )
    parser.add_argument(
        "--interface-scale-ratio",
        type=float,
        default=0.50,
        help="Homothetic scale for the inner ring of each Stage 04 interface (default: 0.50).",
    )
    parser.add_argument(
        "--interface-clearance-mm",
        type=float,
        default=0.20,
        help="Extra side and bottom clearance added to each mortise relative to its tenon (default: 0.20 mm).",
    )
    parser.add_argument("--input", required=True, help="Source .3mf file. No other model files are read.")
    parser.add_argument("--output", default=None, help="Final colored .3mf path; defaults beside the source file.")
    parser.add_argument("--boundary-review-json", default=None,
                        help="Fingerprint-bound user-approved boundary ownership decisions.")
    parser.add_argument("--boundary-check-only", action="store_true",
                        help="Check boundary routing and export ambiguous candidates without splitting.")
    parser.add_argument(
        "--format-profile",
        choices=["auto", "vendor-paint"],
        default="auto",
        help="Detect or require per-triangle paint_color input. Standard 3MF material properties are rejected explicitly.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the final .3mf when it already exists.",
    )
    parser.add_argument(
        "--diagnostic-preview",
        action="store_true",
        help=(
            "Export an explicitly annotated preview even when strict mesh validation fails. "
            "The result is for visual diagnosis, not a validated printable deliverable."
        ),
    )
    parser.add_argument("--model-entry", default=None, help="3MF internal .model entry to read; defaults to the mesh entry with most triangles.")
    parser.add_argument("--color-map-json", default=None, help="Optional JSON mapping color codes to hex strings or {name, hex, rgba}.")
    parser.add_argument("--exterior-view-count", type=int, default=32, help="Outside depth-map directions used for recognition.")
    parser.add_argument("--exterior-depth-map-resolution", type=int, default=768, help="Square depth-map resolution per exterior view.")
    parser.add_argument("--exterior-depth-tolerance-mm", type=float, default=0.08, help="Depth tolerance for externally visible recognition faces.")
    parser.add_argument(
        "--noise-review-max-faces",
        type=int,
        default=100,
        help=(
            "Classify source regions with at most this many faces as noise candidates "
            "for mandatory review; never merge or repair them (default: 100)."
        ),
    )
    parser.add_argument(
        "--small-region-review-max-faces",
        type=int,
        default=999,
        help=(
            "Require semantic review for source regions through this face count; "
            "regions remain unchanged and enter normal splitting (default: 999)."
        ),
    )
    parser.add_argument(
        "--region-review-json",
        default=None,
        help=(
            "User-confirmed noise/part/uncertain classifications for every review candidate. "
            "Classifications never change source geometry."
        ),
    )
    parser.add_argument(
        "--recognition-review-json",
        default=None,
        help=(
            "User-reviewed delete/merge actions and final recognition confirmation. "
            "Without a matching confirmation, full runs stop before assembly."
        ),
    )
    parser.add_argument(
        "--region-review-dir",
        default=None,
        help="Directory for generated source-region review PNGs and manifest; defaults beside the source 3MF.",
    )
    parser.add_argument(
        "--region-review-resolution",
        type=int,
        default=320,
        help="Pixel size of each whole-model or zoom tile in a six-view source-region review sheet.",
    )
    parser.add_argument(
        "--max-extension-mm",
        type=float,
        default=3.0,
        help=(
            "Preferred minimum inward depth. The default is 3 mm; measured parent thickness "
            "may reduce it when the available safe depth is smaller."
        ),
    )
    parser.add_argument(
        "--max-planar-travel-mm",
        type=float,
        default=50.0,
        help=(
            "Absolute inward search ceiling; the measured active-parent bounds and the "
            "0.05 mm reserve set the effective safe depth. Defaults to 50 mm so hollow "
            "parts can use centimetre-scale paths."
        ),
    )
    parser.add_argument(
        "--cap-mode",
        choices=["adaptive", "flat", "tilted", "offset"],
        default="adaptive",
        help=(
            "adaptive uses one coplanar bottom whenever it fits inside the measured safety ceiling, "
            "then falls back to safe local-offset; flat, tilted, and offset remain explicit options."
        ),
    )
    parser.add_argument(
        "--planar-extra-limit-mm",
        type=float,
        default=None,
        help=(
            "Compatibility override for travel between the preferred minimum and safety ceiling. "
            "When omitted, derive it from --max-planar-travel-mm."
        ),
    )
    parser.add_argument(
        "--output-layout",
        choices=["assembly", "separate-items"],
        default="assembly",
        help="Write one top-level component assembly by default, or legacy parallel build items.",
    )
    parser.add_argument("--boundary-target-samples", type=int, default=384)
    parser.add_argument("--boundary-smooth-passes", type=int, default=28)
    parser.add_argument(
        "--maximum-boundary-displacement-mm",
        type=float,
        default=10.0,
        help=(
            "Maximum and P95 planar-arc target displacement in millimeters "
            "before the interface is rejected (default: 10)."
        ),
    )
    parser.add_argument(
        "--seam-smoothing-profile",
        choices=["source-conservative", "print-balanced", "print-smooth"],
        default="print-balanced",
        help=(
            "Printable seam quality budget. print-balanced allows each affected "
            "face or connected reversal cluster through 2%% and their aggregate "
            "through 15%%, while preserving hard topology gates."
        ),
    )
    parser.add_argument(
        "--boundary-retopology-band-mm",
        type=float,
        default=3.0,
        help="Width of the generated local interface band used to absorb boundary motion.",
    )
    parser.add_argument(
        "--connector-slope-validation",
        choices=["strict", "advisory"],
        default="advisory",
        help=(
            "Whether measured 30-75 degree backing-slope departures block the run. "
            "Advisory preserves the measurements in the report while topology and "
            "Boolean quality checks remain blocking."
        ),
    )
    parser.add_argument(
        "--surface-band-validation",
        choices=["strict", "advisory"],
        default="strict",
        help=(
            "Whether a user-reviewed surface band may report source-normal "
            "changes and bounded edge stretch up to 128x as visual advisories, "
            "replace area/vertex coverage limits with the configured maximum and "
            "P95 displacement envelope when both are satisfied, "
            "allow the target to use up to 60%% of the requested real surface "
            "band, and accept eligible sparse isolated inversions down to the "
            "profile's result-angle floor "
            "result angle. Degeneracy, topology, and Boolean checks remain "
            "blocking."
        ),
    )
    parser.add_argument(
        "--connector-surface-validation",
        choices=["strict", "advisory"],
        default="strict",
        help=(
            "Whether an already topology-audited hidden connector annulus may "
            "skip optional flat-shading resolution refinement. Advisory keeps "
            "the measured internal-edge length in the report; topology, "
            "degeneracy, thickness, and Boolean checks remain blocking."
        ),
    )
    parser.add_argument(
        "--visual-semantics-json",
        default=None,
        help=(
            "JSON mapping every recognized P## region to a visual semantic label; "
            "recognition review artifacts are not written when any label is missing."
        ),
    )
    parser.add_argument("--visual-semantic-min-confidence", default="MED", help="Minimum confidence for applying semantic parent hints: LOW, MED, HIGH, or 0-1.")
    parser.add_argument("--recognize-only", action="store_true", help="Only parse and print recognized parts; do not export the final 3MF.")
    parser.add_argument("--preflight-only", action="store_true", help="Only run preflight checks.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.boundary_target_samples < 16:
        parser.error("--boundary-target-samples must be at least 16")
    if args.boundary_smooth_passes < 0:
        parser.error("--boundary-smooth-passes must be non-negative")
    if args.boundary_retopology_band_mm <= 0:
        parser.error("--boundary-retopology-band-mm must be positive")
    if args.maximum_boundary_displacement_mm <= 0:
        parser.error("--maximum-boundary-displacement-mm must be positive")
    if not math.isfinite(args.interface_scale_ratio) or not 0.0 < args.interface_scale_ratio < 1.0:
        parser.error("--interface-scale-ratio must be greater than 0 and less than 1")
    if not math.isfinite(args.interface_clearance_mm) or args.interface_clearance_mm < 0:
        parser.error("--interface-clearance-mm must be non-negative")
    if args.exterior_view_count < 6:
        parser.error("--exterior-view-count must be at least 6")
    if args.exterior_depth_map_resolution < 64:
        parser.error("--exterior-depth-map-resolution must be at least 64")
    if not 128 <= args.region_review_resolution <= 1024:
        parser.error("--region-review-resolution must be between 128 and 1024")
    if args.noise_review_max_faces < 0:
        parser.error("--noise-review-max-faces must be non-negative")
    if args.small_region_review_max_faces < args.noise_review_max_faces:
        parser.error("--small-region-review-max-faces must be at least --noise-review-max-faces")
    if args.exterior_depth_tolerance_mm < 0:
        parser.error("--exterior-depth-tolerance-mm must be non-negative")
    if args.max_planar_travel_mm < max(float(args.max_extension_mm), 0.4):
        parser.error("--max-planar-travel-mm must be at least the effective target inward depth")
    if args.planar_extra_limit_mm is not None and args.planar_extra_limit_mm < 0:
        parser.error("--planar-extra-limit-mm must be non-negative")
    input_path = Path(args.input).expanduser()
    progress("预检", "检查输入文件和 Python 依赖", input=str(input_path))
    checks = preflight(input_path)
    print("preflight=" + json.dumps(checks, ensure_ascii=False), flush=True)
    from .application.stage_artifacts import StageArtifactStore
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else input_path.expanduser().resolve().with_name(
            f"{sanitize_name(input_path.stem) or 'model'}_split_parts.3mf"
        )
    )
    artifact_root = (
        Path(args.stage_artifacts_dir).expanduser()
        if args.stage_artifacts_dir
        else output_path.with_name(output_path.stem + "_stages")
    )
    args.stage_artifacts_run_id = uuid.uuid4().hex
    artifact_store = StageArtifactStore(artifact_root, run_id=args.stage_artifacts_run_id)
    artifact_store.write_json(
        "01_preflight", {"input": str(input_path.resolve()), "checks": checks}
    )
    print("stage_artifacts=" + str(artifact_store.run_dir), flush=True)
    failures = preflight_failures(checks)
    if failures:
        print("preflight_failures=" + json.dumps(failures, ensure_ascii=False), file=sys.stderr)
        if checks.get("missing_dependencies"):
            print("依赖未就绪，请确认后运行：" + checks["install_command"], file=sys.stderr)
        raise SystemExit(2)
    progress("预检", "依赖与输入检查通过", python=checks["python"])
    if args.preflight_only:
        return
    if args.stop_after_stage == "preflight":
        return
    load_core_dependencies()
    from .print_tolerance import PrintTolerance, tolerance_scope
    from .pipeline import SplitPipeline

    config = SplitConfig(namespace=args, input_path=input_path, preflight_checks=checks)
    from .boundary_review import BoundaryReviewRequired, BoundaryDecisionError
    try:
        if args.micro_defect_area_mm2 < 0 or args.print_surface_tolerance_mm < 0:
            parser.error('Print tolerance values must be non-negative')
        recovery = Path(args.recovery_dir) if args.recovery_dir else input_path.parent / (input_path.stem + '_split_recovery')
        with tolerance_scope(PrintTolerance(args.micro_defect_area_mm2,
                                            args.print_surface_tolerance_mm, recovery,
                                            args.hidden_surface_refinement == 'preserve')):
            SplitPipeline(config, parser).run()
    except BoundaryReviewRequired as exc:
        print("boundary_review=" + json.dumps(dict(status="needs_user_confirmation",
              directory=str(exc.directory), report=exc.report), ensure_ascii=False), flush=True)
        raise SystemExit(4) from None
    except BoundaryDecisionError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        print('split_failure=' + json.dumps(dict(error=str(exc), detail=getattr(exc, 'record', {})),
                                            ensure_ascii=False), flush=True)
        raise SystemExit(3) from None
