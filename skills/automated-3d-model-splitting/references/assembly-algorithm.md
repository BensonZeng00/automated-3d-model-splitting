# Recognition, Classification, Assembly, and Geometry Rules

## Exterior-Only Recognition

Before visibility recognition, resolve the selected mesh through the package build/component graph. Require one unique project instance, compose all affine transforms, convert translation to millimeters, and bake the transform into the working vertices. Apply inward depths only after this conversion.

Decide recognition before generating geometry.

1. Sample deterministic viewing directions around the source mesh.
2. Project source-face centroids into a depth map for every view.
3. Mark a face visible when its depth is within tolerance of the frontmost sample in at least one view.
4. Keep original paint tokens on visible faces.
5. Reassign occluded faces to the explicit body color, source `DEFAULT`, or most common exterior color. Before reassignment, preserve any connected occluded patch whose complete shared-edge rim consists of visible faces with the same source color, or whose rim has at least two-thirds visible same-color support and no visible competing material. This protects centroid-depth visibility holes at both patch interiors and material boundaries without restoring open or wholly hidden paint.
6. Group the resulting labels by shared mesh edges.

For connectivity labels, coalesce raw paint tokens only when trusted source metadata resolves them to the same filament slot. Keep the original token counts as provenance on the resulting component.

This filter changes recognition labels only. Preserve the source vertices, faces, and paint metadata.

## Component Recognition

Use `--noise-review-max-faces 100` and `--small-region-review-max-faces 999` only to select semantic-review candidates. Long strips remain candidates at every face count. A complete source-matched decision file classifies every candidate as `noise`, `part`, or `uncertain`; all three classifications preserve the source region and enter the same body-selection, tree-inference, and interface-planning flow. Never merge, delete, recolor, repair, or filter source geometry because of its classification.

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

Debug output represents the actual strict inward recursion timeline. Every ordered step exports its changed parts as independent colored millimeter 3MF files and one cumulative millimeter 3MF containing the complete assembly state after that replacement. A pending child remains in its parent layer and its independent 3MF—not the latest cumulative package—is reloaded when that child is processed. Unprocessed descendants remain merged inside pending subassemblies with their original triangle colors; already processed branches remain unchanged. For Bambu projects, every standalone pending package carries both standard 3MF triangle material properties and per-triangle `paint_color` tokens; the object-level extruder is only the fallback/default material. Debug packages use the active `ratio` or `strict` validation profile. Preserve the last valid standalone input and cumulative audit, and identify the exact failing step, input file, local body, and changed parts.

## Boundary Safety

### User-confirmed visible-edge retreat

When the source paint boundary crosses a thin visible tangent but the intended
physical split belongs farther inside the model, use an opt-in
`interface_retreats` visual-semantic record. This changes physical ownership of
an existing source-surface patch; it does not recolor that patch. Run the
operation after recognition and body selection but before final component
classification, tree inference, or cap generation.

Start from a user-confirmed 3-D seed near the affected shared seam. Select only
parent-owned source faces inside the seed radius, require that the seed region
touch the current child/parent boundary, then grow over the parent surface with
deterministic edge-length Dijkstra distance up to `retreat_distance_mm`.
Transfer those source face ids to the child while preserving every face's
original material token and filament slot. Absorb parent fragments completely
enclosed by the transferred patch so the result does not retain isolated
surface islands.

Reject the retreat unless all of these gates pass:

- the declared child and parent exist and share a boundary;
- the seed reaches that boundary and all numeric inputs are finite and positive;
- transferred parent area remains below `maximum_parent_face_fraction`;
- neither child nor parent gains a disconnected surface region;
- a non-empty child/parent shared boundary remains after transfer.

Report transferred face count, closure count, old/new shared-edge counts,
region counts, bounding box, and material preservation. The ordinary thickness,
cap, socket, topology, and multi-view audits still apply afterward. Prefer the
normal local connector when its annuli validate; `boundary-extrusion` remains a
valid two-part fallback for dense concave contours because both strategies keep
the same retreated visible source patch and preserve the assembly union.

- Preserve nesting only when child and parent-install contacts use separate boundary loops.
- Reparent a mixed-loop child to the common parent so one ring is not closed twice.
- Break cycles by preferring a valid larger external neighbor, then the body.
- Skip a socket when shared-boundary evidence is weak or non-closed; retain the semantic relation in reports.
- At one recursive layer, derive an inward install bottom only from the direct child root.

## Inward Geometry

Use one inward geometry behavior for every non-body part:

- do not inward-extrude the root body;
- use the sole smooth policy and shared fitted seam with surface-band checks in [boundary-smoothing.md](boundary-smoothing.md); perform small-anomaly ownership cleanup before fitting;
- reuse identical source-id coordinates on child and parent sides even when their local loop winding is reversed;
- extend inserts along the selected safe inward direction;
- derive an area-weighted local inward normal at every boundary vertex; retain the component direction only where it remains locally safe, otherwise blend toward the local inward normal;
- resolve the sign of the component-level insertion axis independently at every closed interface rim before thickness or connector planning. Treat a coherent, consistently wound source-rim normal as authoritative: if the proposed axis points into the outward hemisphere, flip it for that interface only, retain the component direction when rim evidence is missing or incoherent, record the decision, and keep the final outward-axis audit as a hard publication gate. This prevents a merged paint component spanning differently oriented surfaces from turning one interface into an exterior extrusion;
- require every generated displacement to have a positive dot product with its local inward normal, and copy the same source-vertex direction map into the matching parent socket;
- measure parent thickness along the candidate inward field and set the safe maximum to `min(10 mm, parent thickness - 0.05 mm)`;
- use the same authoritative ray horizon for taper-candidate screening and final replay; a shorter ray cannot safely classify an entry whose matching exit lies beyond that horizon. When an oriented ray starts in free space and later has a paired entry/exit, limit travel by the remote entry distance rather than mistaking the entry-to-exit shell interval for thickness available at the source;
- screen unique taper candidates in descending inset order with the authoritative thickness horizon, stop once the largest inset preserving the preferred depth is proven, then replay the selected field through the same audit and retain disagreement as a blocking invariant before geometry generation;
- when the baseline safe depth is below the 0.45 mm load-bearing minimum, generate deterministic hidden-interface fields by blending the current field toward several interior targets along the active parent centroid direction; project and smooth every field into each boundary vertex's local inward hemisphere, rank it by authoritative safe depth and coherence, then rerun the ordinary cap and reserve gates before accepting it;
- keep the selected shared visible boundary ring locked during hidden-interface recovery, record the baseline and selected safe depths plus candidate evidence, and reuse the exact accepted per-source-vertex field for the parent socket; if no candidate reaches the load-bearing threshold, retain the existing blocking failure rather than publishing a cosmetic shell;
- when a high-confidence `guided_internal_cut` is present, interpret its entry direction as the front internal transition and its target plane as the deeper shared surface; never move the selected shared visible rim to imitate the drawn line;
- probe parent thickness around the whole loop, seed only genuinely thin arcs, and blend the requested entry inset around those arcs with a smooth circular falloff; unaffected arcs keep ordinary fit clearance instead of receiving a whole-loop retreat;
- rank bounded parent-interior direction fields and target-parallel plane shifts by safe depth, coherence, and side-wall quality; require the selected bottom ring to remain coplanar and within its minimum depth, maximum depth, and maximum parallel-shift limits;
- treat the guided prevalidated fit ring as authoritative by source vertex id for both the insert and socket. Bridge the locked shared visible rim to the bottom through a midpoint loft, choose each quad diagonal geometrically, and reject any strip with degenerate faces, long circumferential bridges, excessive stretch, or a widespread normal conflict;
- use a 3.0 mm preferred minimum unless parent thickness is below 3.05 mm, in which case reduce the effective minimum to the safe maximum;
- default to `adaptive` caps; compare the global inward plane with an inward-facing loop best-fit plane, allow per-boundary-point depth to vary while keeping every bottom point coplanar, and choose the deepest plane whose minimum and maximum travel remain inside the effective and safe bounds;
- use smooth local-offset only when no coherent plane satisfies both bounds, moving every boundary point along its safe local inward direction to the measured safe maximum, normally up to 10.0 mm;
- apply the same plane-first safety rule to recursive parents and nested leaves; owning child inserts is not by itself a reason to force local-offset;
- prevalidate every direct child's cap before building its parent socket, including root-level children and nested leaves; carry one `CapDecision` keyed by source vertex id across both builds, reuse its selected mode, direction field, and distance field for the socket bottom, reserve bottom clearance inside the measured safety ceiling, and never refit the parent socket plane independently;
- preserve the original source-loop points in each `CapDecision`; match exact source vertex ids first, then reconcile only coordinate aliases, T-joint subdivisions, and short alternate paint-boundary routes by projecting one source loop onto the other; cap that projection at `max(planar-arc safe band offset, effective interface fit clearance)`, interpolate the already validated cap field, keep visible top rings unchanged, record the reconciliation, and reject anything beyond the bound;
- when the selected interface is `local-connector`, preflight the exact production geometry instead of a boundary-extrusion proxy: build the simplified paired backing contours, one-to-one ruled wall, compact peg or zero-engagement floor, and private Boolean cutters in memory; require exact face accounting, nondegenerate generated triangles, bounded cross-edges, and watertight cutter validation before any large Boolean; strict slope validation additionally requires a central 30-75 degree backing profile with at most 2% isolated nearest-projection outliers and no sustained outlier arc, while explicit advisory mode records that slope finding without weakening the other gates;
- construct the load-bearing backing as a paired contour: first simplify the outer projected boundary under the configured geometric error bound, then obtain the inner boundary by homothetic scaling about the measured interior center. The two generated rings must retain identical vertex counts and cyclic order, so their wall is sewn one-to-one rather than by a 28k-to-dozens cross-ring zipper. If the scaled contour is not contained (for example, a non-star-shaped outline), reduce the feasible backing depth or omit the optional peg; never admit projection folds, area mismatch, or long cross-ring edges by increasing a tolerance;
- treat the compact peg and socket as optional reinforcement. Omit both whenever measured thickness cannot provide the minimum printable engagement after backing and bottom-clearance budgets; retain the full-boundary backing rather than manufacturing a mechanically fictional thin tenon;
- diagnose local-connector backing depth from two independent measurements: axial distance to the opposing parent shell and projected in-plane clearance available to the taper. A globally thick model can still have a narrow painted interface, so report `local_connector_backing_limiting_constraint` and both nominal-depth booleans before changing inward depth; increasing extrusion cannot fix a lateral-clearance limit. Treat 45 degrees as a neutral reference rather than a constant: repair selects a 30--75 degree taper from the interface-clearance distribution, and missing local-normal exits are retried by bounded blends toward the already validated inward axis before containment is audited;
- judge planar-arc adjustment using both physical displacement and scale-free loop-relative displacement. When a relatively small change (such as roughly 5.975 mm on a large interface) is represented by a generated hidden ring, preserve the visible source rim and carry the validated requested displacement into hidden-ring correspondence instead of requiring the visible surface band itself to be that wide;
- keep all nested pre-cut fit clearances at zero; subtract exact full-size children and apply final whole-part uniform scaling once, as described in [uniform-fit.md](uniform-fit.md);
- smooth local fallback directions over physical boundary arc length and triangulate single-loop caps with distributed short diagonals, never a synthetic center fan;
- let nested inserts use their effective nested cap mode;
- bridge hole rings into their containing outer ring and use boundary-preserving ear clipping for multi-loop bottoms;
- never use unconstrained Delaunay plus centroid filtering for concave or holed inward caps;
- generate matching parent sockets;
- on a multicolor pending subassembly, derive generated parent-contact wall and cap materials from the source faces incident to each local boundary segment; never flood all generated faces with the wrapper root color;
- keep fit clearance separate from fixed inward depth.


## Clearance

The sole fit workflow is [exact subtraction followed by final uniform scaling](uniform-fit.md).
Pre-cut fit and floor clearances are zero. Subtract the complete unscaled child
solid without exterior proxies, shifted backing cutters, or compact socket
cutters. After all recursive cuts finish, scale complete final inward parts to
99% by default in XYZ about each own bounding-box center; never scale the root.
Keep the original full-size recursive artifacts as provenance, not final
scaled deliverables. Do not interpret historical pre-scale seam measurements
as measurements of the scaled assembly.

## Validation

Validate every mesh in memory and after package reload. Record the planar-arc sample count, pass count, band width, maximum and RMS target offset, ripple before/after, target inward depth, cap mode, maximum generated inward travel, and outward-directed boundary vertices before and after correction for every part. Target offset must stay within its configured safe band fraction, and the accepted post-correction outward count is zero. Validate package entries, object/build counts, names, colors, and annotations before atomic replacement.

Also compare source and generated assembled surfaces across deterministic common depth views. Reject generated caps or sockets that become frontmost ahead of the source surface beyond tolerance, reject excessive global front-material ownership changes, reject any generated material that covers too much of one source part, and require minimum source coverage. The local material-pair ratio prevents a small but visually dominant feature from disappearing inside an acceptable whole-model average. Keep this check offline. When a slicer screenshot is needed, request it from the user instead of controlling the slicer.

## Visual Semantics

Apply explicit visual parent hints only after topology inference, mixed-loop repair, and cycle repair. Visual semantics never replace body evidence or mesh validation.
