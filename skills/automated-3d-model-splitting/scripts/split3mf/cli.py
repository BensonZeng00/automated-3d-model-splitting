from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import *
from .domain import SplitConfig
from .uniform_fit import configure_uniform_fit
from .overlap_policy import DEFAULT_IGNORE_OVERLAP_RATIO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recognize parts from externally visible per-triangle paint, use structural evidence "
            "to choose the root body, then recursively inward-extrude every non-body part."
        )
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument('--micro-defect-area-mm2', type=float, default=1.0)
    parser.add_argument('--print-surface-tolerance-mm', type=float, default=0.05)
    parser.add_argument('--boundary-shape', type=str.lower, choices=['smooth'], default='smooth', help='Smooth shared cut boundaries; source mode has been removed.')
    parser.add_argument('--hidden-surface-refinement', choices=['preserve', 'refine'],
                        default='preserve', help='Preserve audited hidden annuli or request strict density refinement.')
    parser.add_argument('--recovery-dir', help='Persistent inputs and candidates for local failure replay')
    parser.add_argument("--input", required=True, help="Source .3mf file. No other model files are read.")
    parser.add_argument("--output", default=None, help="Final colored .3mf path; defaults beside the source file.")
    parser.add_argument("--post-split-uniform-scale", type=float, default=0.99,
                        help="Subtract exact full-size child solids with no added clearance, then scale complete inserts about their own bbox centers, e.g. 0.99.")
    parser.add_argument(
        "--allow-coupled-seating",
        action="store_true",
        help=(
            "Apply a user-confirmed bounded rigid seating correction to an inward "
            "subassembly and all of its descendants as one unit."
        ),
    )
    parser.add_argument(
        "--seating-overlap-tolerance-mm3",
        type=float,
        default=1e-8,
        help=(
            "Maximum measured parent/insert overlap accepted as numerical roundoff "
            "during final seating validation (default: 1e-8 mm^3)."
        ),
    )
    parser.add_argument('--assembly-ignore-overlap-ratio', type=float,
                        default=DEFAULT_IGNORE_OVERLAP_RATIO,
                        help='Silently accept pair overlap / original actual cutting volume strictly below this fraction (default: 0.01 = 1%%; 0 disables).')
    parser.add_argument('--seating-penetration-tolerance-mm', type=float, default=0.0,
                        help='Accepted local intersection slab thickness bound in millimeters.')
    parser.add_argument('--post-fit-parent-difference', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Subtract final ancestor solids from scaled inserts before seating (enabled by default); retain topology, thickness and visual gates.')
    parser.add_argument('--repair-thin-backing', action='store_true',
                        help='Rebuild failed hidden backing along source-local inward normals before exact parent subtraction.')
    parser.add_argument('--assembly-fit-validation', choices=['manual', 'strict'], default='manual',
                        help='Export valid parts with measured assembly issues for manual adjustment (default), or block on unresolved fit.')
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
        "--full-tree-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Plan and safety-check every recursive interface before the first "
            "expensive Boolean (enabled by default)."
        ),
    )
    parser.add_argument(
        "--resume",
        choices=["off", "auto", "strict"],
        default="off",
        help=(
            "Reuse validated content-addressed recursive stages. auto treats an "
            "invalid entry as a miss; strict stops on invalid cache data."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Persistent recursive-stage cache directory used by --resume.",
    )
    parser.add_argument(
        "--debug-recursive-3mf",
        dest="debug_recursive_3mf",
        action="store_true",
        help="Write every changed part as a colored 3MF, feed parent-emitted child 3MF files into later recursion, and retain cumulative audit 3MF files under <output-stem>_debug/recursive_layers/.",
    )
    parser.add_argument(
        "--debug-recursive-stl",
        dest="debug_recursive_3mf",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--debug-recursive-steps",
        type=int,
        default=None,
        help=(
            "Stop successfully after this many strict recursive debug steps. "
            "Requires --debug-recursive-3mf and intentionally produces no final deliverable."
        ),
    )
    parser.add_argument(
        "--diagnostic-preview",
        action="store_true",
        help=(
            "Export an explicitly annotated preview even when strict mesh validation fails. "
            "The result is for visual diagnosis, not a validated printable deliverable."
        ),
    )
    parser.add_argument(
        "--validation-profile",
        choices=["ratio", "strict"],
        default="ratio",
        help=(
            "ratio accepts localized topology defects whose combined edge ratio stays within the configured limit; "
            "strict requires zero open and over-shared edges."
        ),
    )
    parser.add_argument(
        "--max-topology-defect-ratio",
        type=float,
        default=0.001,
        help="Maximum per-part (open + over-shared edges) / unique edges ratio in ratio validation mode.",
    )
    parser.add_argument("--model-entry", default=None, help="3MF internal .model entry to read; defaults to the mesh entry with most triangles.")
    parser.add_argument("--color-map-json", default=None, help="Optional JSON mapping color codes to hex strings or {name, hex, rgba}.")
    parser.add_argument(
        "--recognition-surface-profile",
        choices=["exterior-visible", "all-faces"],
        default="exterior-visible",
        help=(
            "Use paint visible from outside for part recognition, or retain legacy all-face recognition. "
            "Geometry generation uses recursive inward extrusion for every non-body part."
        ),
    )
    parser.add_argument("--exterior-view-count", type=int, default=32, help="Outside depth-map directions used for recognition.")
    parser.add_argument("--exterior-depth-map-resolution", type=int, default=768, help="Square depth-map resolution per exterior view.")
    parser.add_argument("--exterior-depth-tolerance-mm", type=float, default=0.08, help="Depth tolerance for externally visible recognition faces.")
    parser.add_argument(
        "--part-processing-mode",
        choices=["auto", "inward"],
        default="auto",
        help="All non-body parts use inward geometry; inward is retained as an explicit compatibility value.",
    )
    parser.add_argument(
        "--interface-geometry",
        choices=["local-connector"],
        default="local-connector",
        help=(
            "Build male backing/peg solids, subtract them at full size, then scale the emitted inserts."
        ),
    )
    parser.add_argument(
        "--part-mode-overrides",
        default="",
        help="Legacy-compatible inward-only overrides such as P10=inward.",
    )
    parser.add_argument(
        "--accept-ambiguous-inward",
        action="store_true",
        help="Resolve uncertain automatic classifications as legacy inward inserts instead of stopping for confirmation.",
    )
    parser.add_argument("--min-faces", type=int, default=1000)
    parser.add_argument(
        "--tiny-component-policy",
        choices=["semantic", "merge", "ignore"],
        default="semantic",
        help=(
            "How to handle color-connected fragments below --min-faces. "
            "semantic auto-merges fragments at or below --tiny-component-auto-noise-max-faces, "
            "then renders the remaining candidates for image review and requires a user-confirmed decision file; "
            "merge assigns every fragment to an effective part; ignore preserves legacy filtering."
        ),
    )
    parser.add_argument(
        "--tiny-component-auto-noise-max-faces",
        type=int,
        default=100,
        help=(
            "Under semantic policy, automatically classify connected regions with at most this many faces "
            "as noise and merge them without image review (default: 100)."
        ),
    )
    parser.add_argument(
        "--tiny-component-review-json",
        default=None,
        help=(
            "User-confirmed image-review decisions for every rendered candidate above the auto-noise threshold. "
            "Selected items are preserved and every unselected item is merged."
        ),
    )
    parser.add_argument(
        "--tiny-component-review-dir",
        default=None,
        help="Directory for generated small-component review PNGs and manifest; defaults beside the source 3MF.",
    )
    parser.add_argument(
        "--tiny-component-review-resolution",
        type=int,
        default=320,
        help="Pixel size of each whole-model or zoom tile in a six-view small-component review sheet.",
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
        default=10.0,
        help=(
            "Global inward safety ceiling. The actual ceiling is min(this value, 10 mm, "
            "measured parent thickness - 0.05 mm)."
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
        "--nested-cap-mode",
        choices=["inherit", "adaptive", "flat", "tilted", "offset"],
        default="adaptive",
        help="Cap mode for inserts whose parent is another insert and for matching child sockets.",
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
        "--force-flat-parts",
        default="",
        help="Comma/space separated recognized part ids or indices, e.g. P06, to force flat caps for user-confirmed planar parts.",
    )
    parser.set_defaults(flat_clearance_mm=0.0, fit_clearance_mm=0.0,
                        clearance_profile="fixed", clearance_feature_ratio=0.08,
                        clearance_min_mm=0.0, bottom_clearance_mm=0.0,
                        sibling_clearance_mm=0.0, clearance_mode="insert-shrink")
    parser.add_argument(
        "--lead-in-mm",
        type=float,
        default=0.60,
        help="Maximum entry-taper depth; actual depth follows the local lateral shrink for an approximately 45-degree slope.",
    )
    parser.add_argument(
        "--assembly-mode",
        choices=["tree", "flat", "legacy-flat"],
        default="tree",
        help="tree and flat infer recursive subassemblies; legacy-flat makes every non-body part a direct body insert.",
    )
    parser.add_argument(
        "--output-layout",
        choices=["assembly", "separate-items"],
        default="assembly",
        help="Write one top-level component assembly by default, or legacy parallel build items.",
    )
    parser.add_argument(
        "--assembly-tree-strategy",
        choices=["recursive-minimal", "strongest-path"],
        default="recursive-minimal",
        help=(
            "recursive-minimal splits each local body into only its direct child subassemblies, then recurses; "
            "strongest-path preserves the older global body-rooted path heuristic."
        ),
    )
    parser.add_argument("--min-assembly-shared-edges", type=int, default=20, help="Minimum shared original mesh edges used to infer a parent-child assembly relation.")
    parser.add_argument(
        "--assembly-direction-override-dot",
        type=float,
        default=0.0,
        help=(
            "Only override a nested insert's inward direction toward its inferred parent when the original inward direction "
            "has dot(parent_direction) below this threshold. Lower values preserve more original geometry."
        ),
    )
    parser.add_argument("--boundary-target-samples", type=int, default=384)
    parser.add_argument("--boundary-smooth-passes", type=int, default=28)
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
            "allow the target to use up to 60%% of the requested real surface "
            "band, and accept eligible sparse isolated inversions down to a 1 degree "
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
        "--body-strategy",
        choices=["auto-score", "largest", "none"],
        default="auto-score",
        help="How to choose the body part. Automatic scoring never filters candidates by color.",
    )
    parser.add_argument("--body-color", default=None, help="Choose the largest effective component with this color code as body.")
    parser.add_argument("--body-index", type=int, default=None, help="Choose the 1-based recognized component index as body.")
    parser.add_argument(
        "--merge-body-parts",
        default=None,
        help=(
            "Merge two or more pre-merge recognized parts into one multi-material body, "
            "for example P14+P04. The first part supplies the body identity while all "
            "source per-face filament assignments are preserved."
        ),
    )
    parser.add_argument("--visual-semantics-json", default=None, help="Optional JSON with visual part labels and semantic parent-child relation hints.")
    parser.add_argument("--visual-semantic-min-confidence", default="MED", help="Minimum confidence for applying semantic parent hints: LOW, MED, HIGH, or 0-1.")
    parser.add_argument(
        "--visual-validation-profile",
        choices=["strict", "report", "off"],
        default="strict",
        help="Block, report, or skip deterministic multi-view surface/depth consistency validation.",
    )
    parser.add_argument("--visual-validation-view-count", type=int, default=32)
    parser.add_argument("--visual-validation-resolution", type=int, default=384)
    parser.add_argument("--visual-depth-tolerance-mm", type=float, default=0.12)
    parser.add_argument(
        "--visual-max-intrusion-ratio",
        type=float,
        default=0.04,
        help=(
            "Blocking generated-surface intrusion ratio. Ratios above 2%% but not above "
            "this default 4%% limit are retained as advisory findings."
        ),
    )
    parser.add_argument("--visual-max-material-mismatch-ratio", type=float, default=0.03)
    parser.add_argument(
        "--visual-max-local-material-mismatch-ratio",
        type=float,
        default=0.10,
        help="Maximum generated-material coverage of any one source part across all validation views.",
    )
    parser.add_argument(
        "--visual-max-local-material-mismatch-pixels",
        type=int,
        default=64,
        help="Maximum absolute mismatched pixels for one source/generated material pair across all validation views.",
    )
    parser.add_argument("--visual-min-coverage-ratio", type=float, default=0.65)
    parser.add_argument("--recognize-only", action="store_true", help="Only parse and print recognized parts; do not export the final 3MF.")
    parser.add_argument("--preflight-only", action="store_true", help="Only run preflight checks.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configure_uniform_fit(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.min_faces < 1:
        parser.error("--min-faces must be at least 1")
    if args.boundary_target_samples < 16:
        parser.error("--boundary-target-samples must be at least 16")
    if args.boundary_smooth_passes < 0:
        parser.error("--boundary-smooth-passes must be non-negative")
    if args.boundary_retopology_band_mm <= 0:
        parser.error("--boundary-retopology-band-mm must be positive")
    if args.fit_clearance_mm < 0 or args.lead_in_mm < 0 or args.sibling_clearance_mm < 0:
        parser.error("clearance and lead-in values must be non-negative")
    if args.clearance_feature_ratio <= 0 or args.clearance_min_mm < 0:
        parser.error("adaptive clearance ratio must be positive and minimum must be non-negative")
    if args.exterior_view_count < 6:
        parser.error("--exterior-view-count must be at least 6")
    if args.exterior_depth_map_resolution < 64:
        parser.error("--exterior-depth-map-resolution must be at least 64")
    if not 128 <= args.tiny_component_review_resolution <= 1024:
        parser.error("--tiny-component-review-resolution must be between 128 and 1024")
    if args.tiny_component_auto_noise_max_faces < 0:
        parser.error("--tiny-component-auto-noise-max-faces must be non-negative")
    if args.exterior_depth_tolerance_mm < 0:
        parser.error("--exterior-depth-tolerance-mm must be non-negative")
    if args.max_planar_travel_mm < max(float(args.max_extension_mm), 0.4):
        parser.error("--max-planar-travel-mm must be at least the effective target inward depth")
    if args.planar_extra_limit_mm is not None and args.planar_extra_limit_mm < 0:
        parser.error("--planar-extra-limit-mm must be non-negative")
    if not 0.0 <= args.max_topology_defect_ratio <= 0.001:
        parser.error("--max-topology-defect-ratio must be between 0 and 0.001 (0.10%)")
    if args.visual_validation_view_count < 6 or args.visual_validation_resolution < 64:
        parser.error("visual validation requires at least 6 views and 64-pixel depth maps")
    if args.visual_depth_tolerance_mm < 0:
        parser.error("--visual-depth-tolerance-mm must be non-negative")
    for value, option in (
        (args.visual_max_intrusion_ratio, "--visual-max-intrusion-ratio"),
        (args.visual_max_material_mismatch_ratio, "--visual-max-material-mismatch-ratio"),
        (
            args.visual_max_local_material_mismatch_ratio,
            "--visual-max-local-material-mismatch-ratio",
        ),
        (args.visual_min_coverage_ratio, "--visual-min-coverage-ratio"),
    ):
        if not 0 <= value <= 1:
            parser.error(f"{option} must be between 0 and 1")
    if args.visual_max_local_material_mismatch_pixels < 0:
        parser.error("--visual-max-local-material-mismatch-pixels must be non-negative")
    if args.debug_recursive_3mf and (
        args.assembly_mode not in {"tree", "flat"} or args.assembly_tree_strategy != "recursive-minimal"
    ):
        parser.error("--debug-recursive-3mf requires tree/flat assembly with recursive-minimal strategy")
    if args.resume != "off" and not args.cache_dir:
        parser.error("--resume requires --cache-dir")
    if args.debug_recursive_3mf and args.resume != "off":
        parser.error("--resume cannot be combined with --debug-recursive-3mf")
    if args.merge_body_parts and (args.body_index is not None or args.body_color):
        parser.error(
            "--merge-body-parts already selects the body and cannot be combined "
            "with --body-index or --body-color"
        )
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
    progress("预检", "依赖与输入检查通过", python=checks["python"])
    if args.preflight_only:
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
                                            args.hidden_surface_refinement == 'preserve',
                                            args.repair_thin_backing,
                                            args.post_split_uniform_scale)):
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
