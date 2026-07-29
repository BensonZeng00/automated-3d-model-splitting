# Recognition, Classification, Assembly, and Geometry Rules

## Exterior-Only Recognition

Before visibility recognition, resolve the selected mesh through the package build/component graph. Require one unique project instance, compose all affine transforms, convert translation to millimeters, and bake the transform into the working vertices. Apply inward depths only after this conversion.

Decide recognition before generating geometry.

1. Sample deterministic viewing directions around the source mesh.
2. Project source-face centroids into a depth map for every view.
3. Mark a face visible when its depth is within tolerance of the frontmost sample in at least one view.
4. Keep original paint tokens on visible faces.
5. Reassign occluded faces to the explicit body color, source `DEFAULT`, or most common exterior color.
6. Group the resulting labels by shared mesh edges.

For connectivity labels, coalesce raw paint tokens only when trusted source metadata resolves them to the same filament slot. Keep the original token counts as provenance on the resulting component.

This filter changes recognition labels only. Preserve the source vertices, faces, and paint metadata.

## Component Recognition

Merge components below `--min-faces` before body selection, processing classification, and tree inference. Prefer shared-edge assignment; use nearest bounding boxes only when no shared edge exists.

Before automatic body selection, test high-confidence through candidates as possible structural separators. Exclude a candidate from automatic body selection only when all of the following hold:

- the through geometry gate and score pass;
- it has at least two large structural interfaces, where each neighbor has at least 25% of the candidate's face count and shares at least 32 source edges; and
- removing it splits the complete shell into exactly two meaningful regions whose smaller/larger face-count ratio is at least 0.50.

Record a body-candidate score that combines relative component size with penalties for through likelihood, balanced shell separation, and multiple large structural interfaces. After excluding definite strong separators, choose the highest-scoring remaining component across all colors. `DEFAULT`, base-color, filament-slot, and resolved material color must not filter or boost automatic body candidates. Explicit body flags always win.

## Structural Evidence and Processing Mode

Measure each component independently. Never infer structure from color name alone.

Measure:

- area-weighted normal resultant;
- balance between opposite-facing normal clusters;
- separation of those clusters along the dominant normal;
- component diagonal relative to the source model;
- largest boundary-loop span relative to the source model.

Treat those measurements only as body-selection evidence. Mark a strong structural separator candidate when both conditions hold:

1. removing the candidate from the current complete shell leaves at least two meaningful connected shell regions; and
2. the candidate has at least two large structural shared-boundary interfaces to other recognized parts.

Count a neighboring part only when it has at least 25% of the candidate's face count and shares at least 32 source edges. This filters small decorative inserts while retaining structural peers. Record every shared interface, thresholds, and gate result, then use the evidence to penalize or exclude the candidate from automatic body selection. It never creates a geometry mode: the selected body is `body`; every other part is `inward`.


## Recursive Parent Inference

Default to `recursive-minimal`:

1. Treat the current complete piece as a local assembly.
2. Identify its local body and direct child subassemblies from recognized exterior boundaries.
3. Compute the root local body cut and direct children as the first complete assembly state.
4. Serialize every direct child as an independent colored 3MF in the parent step, preserving triangle colors and filament-slot meanings.
5. Traverse pending child subassemblies in deterministic depth-first preorder.
6. Reload the exact standalone child 3MF emitted by its parent, consume that pending subassembly, and replace it with its local body cut plus direct children.
7. Keep cumulative 3MF packages as complete-state audits only; never use them as recursive child inputs.
8. Reject missing, already expanded, skipped-parent, color-changed, or descendant-provenance-mismatched targets.
9. Flatten local bodies and leaves into separate final objects.

Use the final recursive state as the final mesh source. Do not regenerate all final parts in an unrelated component-index loop.

Keep structural evidence independent from parent inference. After selecting the root body, infer one ordinary inward-recursive tree across all recognized components.

## Recursive Debug Trace

Debug output represents the actual strict inward recursion timeline. Every ordered step exports its changed parts as independent colored millimeter 3MF files and one cumulative millimeter 3MF containing the complete assembly state after that replacement. A pending child remains in its parent layer and its independent 3MF—not the latest cumulative package—is reloaded when that child is processed. Unprocessed descendants remain merged inside pending subassemblies with their original triangle colors; already processed branches remain unchanged. Debug packages use the active `ratio` or `strict` validation profile. Preserve the last valid standalone input and cumulative audit, and identify the exact failing step, input file, local body, and changed parts.

## Boundary Safety

- Preserve nesting only when child and parent-install contacts use separate boundary loops.
- Reparent a mixed-loop child to the common parent so one ring is not closed twice.
- Break cycles by preferring a valid larger external neighbor, then the body.
- Skip a socket when shared-boundary evidence is weak or non-closed; retain the semantic relation in reports.
- At one recursive layer, derive an inward install bottom only from the direct child root.

## Inward Geometry

Use one inward geometry behavior for every non-body part:

- do not inward-extrude the root body;
- before extrusion, fair each cut-loop position with constrained arc-length bi-Laplacian optimization; move only along the source-surface conormal, lock detected corners, add broad-extent anchors when a smooth loop has no corners, and enforce the configured maximum displacement;
- canonicalize every loop by source vertex id before fairing so reversed insert and parent loop order produces identical coordinates;
- extend inserts along the selected safe inward direction;
- derive an area-weighted local inward normal at every boundary vertex; retain the component direction only where it remains locally safe, otherwise blend toward the local inward normal;
- require every generated displacement to have a positive dot product with its local inward normal, and copy the same source-vertex direction map into the matching parent socket;
- measure parent thickness along the candidate inward field and set the safe maximum to `min(5 mm, parent thickness - 0.05 mm)`;
- use a 1.0 mm effective minimum unless parent thickness is below 1.05 mm, in which case reduce it to the safe maximum;
- default to `adaptive` caps; compare the global inward plane with an inward-facing loop best-fit plane, allow per-boundary-point depth to vary while keeping every bottom point coplanar, and accept the plane whenever its deepest required distance does not exceed the safe maximum;
- use smooth local-offset only when the required planar maximum exceeds the safe maximum, moving every boundary point along its safe local inward direction to that maximum, normally 5.0 mm;
- apply the same plane-first safety rule to recursive parents and nested leaves; owning child inserts is not by itself a reason to force local-offset;
- prevalidate every direct child's cap before building its parent socket, including root-level children and nested leaves; carry one `CapDecision` keyed by source vertex id across both builds, reuse its selected mode, direction field, and distance field for the socket bottom, reserve bottom clearance inside the measured safety ceiling, and never refit the parent socket plane independently;
- cap nested leaf effective fit clearance at 0.10 mm and record the original feature-adaptive value, preserving assembly clearance without exposing an excessive parent-wall band beside a deep detail;
- smooth local fallback directions over physical boundary arc length and triangulate single-loop caps with distributed short diagonals, never a synthetic center fan;
- let nested inserts use their effective nested cap mode;
- bridge hole rings into their containing outer ring and use boundary-preserving ear clipping for multi-loop bottoms;
- never use unconstrained Delaunay plus centroid filtering for concave or holed inward caps;
- generate matching parent sockets;
- keep fit clearance separate from fixed inward depth.


## Clearance

Treat 0.30 mm as the requested maximum fit clearance and 0.60 mm as the lead-in. Default to feature-adaptive clearance: clamp the requested value by 4% of the part's smallest positive bounding-box extent, with a 0.05 mm practical target, without moving the visible top surface. Record requested and effective values per part. Keep fixed clearance as an explicit compatibility profile.

## Validation

Validate every mesh in memory and after package reload. Record boundary-fairing mode, locked and movable vertices, maximum and RMS displacement, boundary-length change, target inward depth, cap mode, maximum generated inward travel, and outward-directed boundary vertices before and after correction for every part. Boundary displacement must not exceed its configured hard limit, and the accepted post-correction outward count is zero. Validate package entries, object/build counts, names, colors, and annotations before atomic replacement.

Also compare source and generated assembled surfaces across deterministic common depth views. Reject generated caps or sockets that become frontmost ahead of the source surface beyond tolerance, reject excessive global front-material ownership changes, reject any generated material that covers too much of one source part, and require minimum source coverage. The local material-pair ratio prevents a small but visually dominant feature from disappearing inside an acceptable whole-model average. Keep this check offline. When a slicer screenshot is needed, request it from the user instead of controlling the slicer.

## Visual Semantics

Apply explicit visual parent hints only after topology inference, mixed-loop repair, and cycle repair. Visual semantics never replace body evidence or mesh validation.
