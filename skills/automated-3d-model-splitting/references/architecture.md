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
| `split3mf/interface_surface.py` | Build restored full-rim scaled profiles or curved direct closures, sharing paired topology and depth checks. |
| `split3mf/surface_collar_remesh.py`, `part_interface_composition.py`, `interface_assembly.py` | Rebuild source exterior triangles at the frozen contour, retain paint origins, and attach interface faces to complete parts. |
| `split3mf/validation.py`, `interface_package_validation.py` | Check completed meshes and read the written 3MF back. |
| `split3mf/application/stage_artifacts.py` | Write inspectable JSON/NPZ artifacts with manifests. |

`scripts/run_stage04_from_artifacts.py` and `split3mf/resume_assembly_stage.py` build Stage 04 from completed 02/03 artifacts, preserving confirmed source classifications and accepting later semantic labels without reopening the input 3MF. `scripts/run_stage05_from_artifacts.py` and `split3mf/resume_interface_stage.py` replay Stage 05 from completed 02–04 artifacts. They do not reopen the input 3MF or recalculate recognized boundaries.

## Geometry contract

Source boundary topology has first priority when orienting an interface. Match
frozen contour tangents to the tenon's outward-wound source half edges, then use
the opposite closing-cap orientation. The interior/proximity logic below is a
fallback for unmatched boundaries, not a way to override a matched material side.

Stage 04 owns part identities, tenon/mortise roles, and the frozen outer contour. `interface_direction.py` is shared by planning and construction. It uses the topology priority above; unmatched boundaries retain source-interior rays and paired-region proximity as fallback evidence. Stage 05 repeats the role-aware check for legacy plans without changing ownership, and records the actual construction axis separately. Both sides use that axis. The default inner outline is a 0.50 planar scale around the projected area centroid. If that projection intersects itself, only the inner outline is adjusted to a simple contour, retaining one point per outer sample.

`interface_shape.py` classifies the frozen contact rim with arc-length weighted plane fitting, one 10% tail trim and weighted 90th-percentile absolute distance. Tolerance is the larger of 0.05 mm and 0.5% of the smaller in-plane span. Original open-edge samples may inform material-side orientation but never replace the frozen contour. Part-exterior normals are not contact-plane normals. Classification never changes ownership, axis or the frozen rim.

Planar interfaces use the restored full-rim-to-scaled-ellipse strip. Curved interfaces use the image-selected mean-plane shoulder followed by extension of the full scaled ellipse. The spanning-chord cap module is removed. The Stage 05 replay CLI supports explicit `--scaled-contour-interface IXXX` selections without changing shape evidence, roles or direction. Both paths reuse the same ellipse, one exit query, numeric depth backoff and exact rim composition. See [assembly-algorithm.md](assembly-algorithm.md).

The source exterior is clipped inside source triangles near each interface, using a distance field from the frozen Stage 03 contour. The new cut crosses triangle edges at interpolated points, avoiding a stair-step edge inherited from whole-face deletion. A geometric bridge joins the cut to the frozen outer contour without matching source vertex IDs. Clipped faces carry their original face paint. Interface surfaces are connected into each part without a Boolean union. Every complete part must be water-tight and consistently wound before export; visible seam continuity remains a separate inspection gate.

## Artifacts and publication

Early-stop stages write their own JSON/NPZ artifacts under a unique run ID. A complete run writes interface surfaces, complete part meshes, and a summary before publishing. The final 3MF is written to a temporary sibling, validated and reloaded, then moved to its requested output path. Source project colors and filament slots are retained.
