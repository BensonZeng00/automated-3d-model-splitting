from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import *
from .domain import SplitConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recognize parts from externally visible per-triangle paint, use structural evidence "
            "to choose the root body, then recursively inward-extrude every non-body part."
        )
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument(
        "--input",
        help="Source .3mf file. Required unless --preflight-only is used. No other model files are read.",
    )
    parser.add_argument("--output", default=None, help="Final colored .3mf path; defaults beside the source file.")
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
        choices=["merge", "ignore"],
        default="merge",
        help=(
            "How to handle color-connected fragments below --min-faces. "
            "merge assigns them to effective parts before recognition/tree cutting; ignore preserves legacy filtering."
        ),
    )
    parser.add_argument(
        "--max-extension-mm",
        type=float,
        default=1.0,
        help=(
            "Preferred minimum inward depth. The effective minimum is normally 1 mm and is "
            "reduced only when measured parent thickness is below 1.05 mm."
        ),
    )
    parser.add_argument(
        "--max-planar-travel-mm",
        type=float,
        default=5.0,
        help=(
            "Global inward safety ceiling. The actual ceiling is min(this value, 5 mm, "
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
    parser.add_argument("--flat-clearance-mm", type=float, default=0.05, help="Legacy compatibility field; not added to fixed inward cap depth.")
    parser.add_argument("--fit-clearance-mm", type=float, default=0.30, help="Total side clearance for assembly fits.")
    parser.add_argument(
        "--clearance-profile",
        choices=["feature-adaptive", "fixed"],
        default="feature-adaptive",
        help="Clamp insert clearance to local feature scale, or use the requested fixed value unchanged.",
    )
    parser.add_argument(
        "--clearance-feature-ratio",
        type=float,
        default=0.04,
        help="Maximum adaptive clearance as a fraction of the part's smallest nonzero bounding-box extent.",
    )
    parser.add_argument(
        "--clearance-min-mm",
        type=float,
        default=0.05,
        help="Lower practical target for feature-adaptive clearance; never increases above --fit-clearance-mm.",
    )
    parser.add_argument("--bottom-clearance-mm", type=float, default=0.0, help="Legacy compatibility field in fixed-depth mode; not added to inward cap depth.")
    parser.add_argument("--lead-in-mm", type=float, default=0.60, help="Entry chamfer depth before reaching full side clearance.")
    parser.add_argument("--sibling-clearance-mm", type=float, default=0.0, help="Optional per-side clearance applied along same-parent sibling seams. Default is off.")
    parser.add_argument(
        "--clearance-mode",
        choices=["insert-shrink", "socket-overcut", "split"],
        default="insert-shrink",
        help="Allocate side clearance by shrinking inserts, overcutting body sockets, or splitting the difference.",
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
    parser.add_argument(
        "--boundary-fairing-mode",
        choices=["constrained", "taubin", "off"],
        default="constrained",
        help="Fair cut loops with constrained arc-length optimization, legacy Taubin, or no position smoothing.",
    )
    parser.add_argument("--boundary-fairing-radius-mm", type=float, default=1.20)
    parser.add_argument("--boundary-max-displacement-mm", type=float, default=0.075)
    parser.add_argument("--boundary-feature-angle-deg", type=float, default=35.0)
    parser.add_argument("--boundary-fidelity-weight", type=float, default=1.0)
    parser.add_argument("--smooth-iterations", type=int, default=16, help="Legacy Taubin mode only.")
    parser.add_argument("--lambda-factor", type=float, default=0.5, help="Legacy Taubin mode only.")
    parser.add_argument("--mu-factor", type=float, default=-0.53, help="Legacy Taubin mode only.")
    parser.add_argument(
        "--body-strategy",
        choices=["auto-score", "largest", "none"],
        default="auto-score",
        help="How to choose the body part. Automatic scoring never filters candidates by color.",
    )
    parser.add_argument("--body-color", default=None, help="Choose the largest effective component with this color code as body.")
    parser.add_argument("--body-index", type=int, default=None, help="Choose the 1-based recognized component index as body.")
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
            "Blocking generated-surface intrusion ratio. Ratios above 2% but not above "
            "this default 4% limit are retained as advisory findings."
        ),
    )
    parser.add_argument("--visual-max-material-mismatch-ratio", type=float, default=0.03)
    parser.add_argument(
        "--visual-max-local-material-mismatch-ratio",
        type=float,
        default=0.10,
        help="Maximum generated-material coverage of any one source part across all validation views.",
    )
    parser.add_argument("--visual-min-coverage-ratio", type=float, default=0.65)
    parser.add_argument("--recognize-only", action="store_true", help="Only parse and print recognized parts; do not export the final 3MF.")
    parser.add_argument("--preflight-only", action="store_true", help="Only run preflight checks.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.preflight_only and not args.input:
        parser.error("--input is required unless --preflight-only is used")
    if args.min_faces < 1:
        parser.error("--min-faces must be at least 1")
    if args.smooth_iterations < 0:
        parser.error("--smooth-iterations must be non-negative")
    if args.boundary_fairing_radius_mm <= 0:
        parser.error("--boundary-fairing-radius-mm must be positive")
    if args.boundary_max_displacement_mm < 0:
        parser.error("--boundary-max-displacement-mm must be non-negative")
    if not 0 <= args.boundary_feature_angle_deg <= 180:
        parser.error("--boundary-feature-angle-deg must be between 0 and 180")
    if args.boundary_fidelity_weight <= 0:
        parser.error("--boundary-fidelity-weight must be positive")
    if args.fit_clearance_mm < 0 or args.lead_in_mm < 0 or args.sibling_clearance_mm < 0:
        parser.error("clearance and lead-in values must be non-negative")
    if args.clearance_feature_ratio <= 0 or args.clearance_min_mm < 0:
        parser.error("adaptive clearance ratio must be positive and minimum must be non-negative")
    if args.exterior_view_count < 6:
        parser.error("--exterior-view-count must be at least 6")
    if args.exterior_depth_map_resolution < 64:
        parser.error("--exterior-depth-map-resolution must be at least 64")
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
    if args.debug_recursive_3mf and (
        args.assembly_mode not in {"tree", "flat"} or args.assembly_tree_strategy != "recursive-minimal"
    ):
        parser.error("--debug-recursive-3mf requires tree/flat assembly with recursive-minimal strategy")
    input_path = Path(args.input).expanduser() if args.input else None
    if input_path is None:
        progress("预检", "检查 Python 依赖")
    else:
        progress("预检", "检查输入文件和 Python 依赖", input=str(input_path))
    checks = preflight(input_path)
    print("preflight=" + json.dumps(checks, ensure_ascii=False), flush=True)
    failures = preflight_failures(checks)
    if failures:
        print("preflight_failures=" + json.dumps(failures, ensure_ascii=False), file=sys.stderr)
        if checks.get("missing_dependencies"):
            print("依赖未就绪，请确认后运行：" + checks["install_command"], file=sys.stderr)
        raise SystemExit(2)
    progress(
        "预检",
        "依赖检查通过" if input_path is None else "依赖与输入检查通过",
        python=checks["python"],
    )
    if args.preflight_only:
        return
    assert input_path is not None
    load_core_dependencies()
    from .pipeline import SplitPipeline

    config = SplitConfig(namespace=args, input_path=input_path, preflight_checks=checks)
    SplitPipeline(config, parser).run()
