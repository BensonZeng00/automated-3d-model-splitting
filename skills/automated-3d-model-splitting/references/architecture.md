# Architecture

The active path has five stages: preflight, vendor 3MF loading, painted-region recognition with frozen simplified boundaries, pairwise contact planning, and direct tenon/mortise construction. Publication reloads the final 3MF. There is no body selection, parent tree, recursive build, or Boolean cutter.

## Ownership

| Module | Responsibility |
|---|---|
| `scripts/split_painted_3mf.py`, `split3mf/cli.py` | Parse the current CLI, preflight, dispatch, and exit codes. |
| `split3mf/pipeline.py` | Sequence stages and keep side effects at stage boundaries. |
| `split3mf/project.py`, `package_io.py` | Read vendor paint and project settings; write colored assembly 3MF. |
| `split3mf/recognition.py`, `recognition_metadata.py`, `source_region_review.py` | Find visible painted regions and collect user-confirmed labels. |
| `split3mf/application/boundary_snapshot_builder.py` | Freeze the simplified shared contours once during recognition. |
| `split3mf/contact_interface_planner.py` | Decide the tenon and mortise side for each contact using Stage 03 contours. |
| `split3mf/interface_surface.py` | Compute one area-normal construction axis, planar inset, depth, direct outer-to-tip strip, and matched recess. |
| `split3mf/part_interface_composition.py`, `interface_assembly.py` | Join generated interface faces to preserved source exteriors and build complete part meshes. |
| `split3mf/validation.py`, `interface_package_validation.py` | Check completed meshes and read the written 3MF back. |
| `split3mf/application/stage_artifacts.py` | Write inspectable JSON/NPZ artifacts with manifests. |

`scripts/run_stage04_from_artifacts.py` and `split3mf/resume_assembly_stage.py` build Stage 04 from completed 02/03 artifacts, preserving confirmed source classifications and accepting later semantic labels without reopening the input 3MF. `scripts/run_stage05_from_artifacts.py` and `split3mf/resume_interface_stage.py` replay Stage 05 from completed 02–04 artifacts. They do not reopen the input 3MF or recalculate recognized boundaries.

## Geometry contract

Stage 04 owns part identities, tenon/mortise roles, and the frozen outer contour. Stage 05 derives the interface area normal from that contour and uses Stage 04's direction only to choose its sign toward the mortise. The default inner outline is a 0.50 planar scale around the projected area centroid. If that projection intersects itself, only the inner outline is adjusted to a simple contour, retaining one point per outer sample.

The inner contour moves along the construction axis to a collision-limited depth of at most 5 mm. One triangle strip directly joins the frozen outer contour to the extended inner contour, and a planar cap closes the tip. The mortise uses the same topology, with its inner tip expanded and its floor deepened by the configured clearance. No intermediate annulus or independent inner wall is generated. See [assembly-algorithm.md](assembly-algorithm.md).

The source exterior is retained except for the local interface opening. A narrow geometric bridge joins that opening to the frozen outer contour without matching source vertex IDs. Interface surfaces are connected into each part without a Boolean union. Every complete part must be water-tight and consistently wound before export.

## Artifacts and publication

Early-stop stages write their own JSON/NPZ artifacts under a unique run ID. A complete run writes interface surfaces, complete part meshes, and a summary before publishing. The final 3MF is written to a temporary sibling, validated and reloaded, then moved to its requested output path. Source project colors and filament slots are retained.
