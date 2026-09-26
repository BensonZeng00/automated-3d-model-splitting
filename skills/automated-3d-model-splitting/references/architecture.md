# Object-Oriented Architecture

## Stable entry point

`scripts/split_painted_3mf.py` is the public command-line entry point. It only adds the script directory to `sys.path`, imports `split3mf.cli.main`, and invokes it. Keep existing command lines compatible.

## Pipeline

`split3mf.pipeline.SplitPipeline` currently coordinates one complete split run and writes a durable artifact after every completed application stage. Its stage order and `--stop-after-stage` controls are documented in [application-stages.md](application-stages.md):

1. read and normalize the source project;
2. recognize connected painted source regions, review <=100-face noise candidates, 101–999-face small-region candidates, and long strips, then preserve every confirmed noise/part/uncertain region for normal interface planning;
3. use the recognition-frozen simplified boundaries to enumerate every contacting part pair and plan tenon/mortise sides plus inward directions;
4. stop after the Stage 04 relation artifact until later stages are migrated to the pairwise relation contract;
Stages 05 and later are not invoked until their consumers are migrated from
the removed parent-tree contract to `contact-interface-plan/v1`.

The pipeline may coordinate policy but must not duplicate geometry, XML, ZIP, or validation algorithms.

## Domain and configuration

`split3mf.domain` is a package with one cross-stage data contract per file:

- `SplitConfig`: validated CLI namespace and paths;
- `LoadedProject`: normalized mesh and project metadata;
- `RecognitionResult`: recognized components and provenance;
- `AssemblyPlan`: parent, child, depth, and recursive-layer records;
- `CapDecision`: selected cap policy, measured travel, and the original source-loop points used for bounded parent/socket correspondence;
- `PartBuildResult`: built mesh plus annotations;
- `ValidationReport`: package and topology validation result.

Prefer these explicit records when data crosses stage boundaries. Do not introduce new unstructured global dictionaries for pipeline state.

## Services

- `cutting_reference.py` maps the existing sequential Boolean cutter-step
  intersection volumes to final child annotations. These original removal
  volumes stay fixed through scaling and seating. `overlap_policy.py` applies
  the default strictly-below-1% ratio without changing numerical precision.
- `assembly_review.py` measures all unresolved pair intersections and source-front
  occlusion for the default manual-adjustment handoff. It never repairs geometry
  and supports read-only reassessment under the current acceptance policy.
  `assembly_visibility.py` catches only typed
  seating failures for this path and retains validated difference candidates.
- `assembly_case.py` persists checksum-bound, already-scaled source/part geometry,
  color annotations, tree identities and options for `tools/replay_assembly.py`.
  Replaying the final fit stage must not apply another uniform scale.

- `backing_repair.py` validates actual complete-child backing before each parent
  Boolean and owns the explicitly enabled source-following reconstruction.
  `backing_thickness.py` screens every interior source-face centroid along its
  own normal, without filtering by the component-average axis. `local_ray_probe.py`
  provides exact finite ray tests with a conservative spatial broad phase.
  `curved_backing.py` retains front triangles and back-face material provenance.
  `part_mesh_building.py` passes accepted replacements through the normal colored finalizer;
  recursion consumes the new full-size solid, not any pre-repair cutter.

- `boolean_parent.py` supplies the complete current recursive source shell for
  exact child subtraction, retaining curved child patches until the Boolean.
- `surface_rays.py` performs exact parallel ray/triangle queries without an
  optional spatial-index dependency. `insert_visibility.py` checks front
  ownership even for nearly coincident parent membranes.
- `assembly_visibility.py` validates compact final insert fronts and applies
  only measured, bounded leaf seating through `assembly_seating.py`. Rigid pose
  records are separate from uniform scaling; shapes and cutters remain intact.
- The default-enabled `post_fit_difference.py` repair runs before seating.
  It returns private tree-ordered candidates, subtracting final ancestors from
  scaled inserts with face/material provenance. `post_fit_audit.py` owns exact
  nearest-surface queries and sampled local-normal thickness checks. Caller
  meshes are committed only after assembly validation succeeds; the pipeline
  then refreshes mesh statistics and runs the normal visual/package gates.
  This shape change is recorded separately from uniform scaling. The CLI offers
  `--no-post-fit-parent-difference` for an explicit diagnostic opt-out.

- `ThreeMFReader` reads vendor packages and resolves build/component transforms.
- `VendorPaintDecoder` restores composite `paint_color` subdivision streams.
- `PartRecognizer` groups material-equivalent, edge-connected exterior paint.
- `source_region_review.py` renders deterministic whole-model and local-zoom PNG sheets for every face-count or long-strip candidate, writes the classification manifest/template, and validates complete user-confirmed noise/part/uncertain decisions without changing source geometry.
- `contact_interface_planner.py` reads the retained simplified boundary loops and emits one independent tenon/mortise relation per contacting pair. It does not select a body or infer a tree. Stage 05 consumes the ordered frozen boundary from that relation directly; it must not remap the contour to source-edge chains or expand it by arc length.
- `assembly.py` still contains legacy consumers for later stages; Stage 04 does not call its parent-tree planner.
- `InwardDirectionPlanner` creates locally safe, smoothed inward directions.
- `HiddenInterfacePlanner` proposes deterministic parent-interior direction fields only when the baseline interface is thinner than the 0.45 mm load-bearing minimum. Parent-thickness ray checks use at most 768 equal-arc samples from the ordered boundary inputs; cap geometry still uses the complete boundary. No complete-boundary thickness audit follows the sampled depth decision.
- `GuidedInternalCutPlanner` converts a high-confidence image-guided internal-section constraint into a symmetric child/socket `CapDecision`. It localizes entry inset to measured thin arcs, ranks bounded parent-interior direction and plane-shift candidates, locks the already-retopologized visible rim, and emits the source-id fit ring and diagnostics consumed unchanged by both sides.
- `InterfaceRetopologyService` owns the actual shared visible seam and its topology-connected C2 surface-band deformation. It moves child and parent seam vertices to one canonical planar-arc target, preserves vertices outside the band, and blocks degenerate, topologically inconsistent, mismatched, excessively stretched, or materially clustered fold results. Boundary-ear repair first performs one vectorized source/result-normal broad phase, then runs topology-aware local repair only for reversed candidates; the complete vectorized quality audit remains authoritative. On a large band only, at most 0.05% nondegenerate source-normal outliers may remain advisory when each result angle is at least 3 degrees and edge-connected clusters contain at most two faces.
- `ConnectorSurfaceService` regularizes the hidden 45-degree annulus. Index-aligned rings use complete intermediate rings; unequal rings use conforming internal refinement followed by bounded convex-quad edge flips that break inherited radial spoke chains without moving either boundary. Both shared rim rings remain immutable after visible retopology.
- `LocalConnectorPlanningService` owns immutable manufacturing/depth policy,
  rim and footprint safety selection, and the pure priority allocator that
  reduces compact-peg engagement to zero before reducing backing; it does not
  mutate meshes.
- `ConnectorTopologyService` triangulates and audits exact polygon-with-hole
  annuli, then propagates consistent face winding.
- `annulus_projection.py` checks oriented area coverage and nonincident edge
  crossings for every strip strategy. Proven source projection ears are
  restored around a separately audited simple core; generated folds have no
  print-area exemption.
- `surface_direction.py` guards the material side using oriented source-rim
  normals. `average_outward_normal` uses the model center only if the normal
  sum vanishes; recessed surfaces must not be reversed by a radial heuristic.
- `backing_shape.py` measures within-layer dihedrals independently of ring
  transitions. Missing samples are reported as unevaluated, not zero degrees.
- `rim_chord_repair.py` separates generated diagonals co-owned by source
  triangles using conforming inward collars, without moving source vertices
  or rim edges. `source_ear_repair.py` restores bounded source-ear paths of up
  to four triangles under one cumulative physical patch budget.
- `InterfaceRetopologyService` owns the only cut-boundary policy: stable-plane projection, 384 equal-physical-arc samples, a 24-control periodic cubic B-spline in the plane, an 8-control periodic cubic B-spline for stable-plane-normal height, no source-extrema restoration, source-id-aligned evaluation, and safe-direction 3-D restoration. `planar_arc.py` contains the reusable numerical kernel. Targets outside 45% of the configured local band are rejected without partial backoff.
- `AdaptiveCapPlanner` tries flat caps first and selects local-offset fallback only when needed.
- `BoundaryTriangulator` closes single-loop and holed boundaries without center fans.
- `PartMeshBuilder` builds inserts, body cuts, sockets, and subassemblies.
- `FullTreePreflightService` visits every recursive parent and materializes its
  interface decisions before Boolean execution; it owns no mesh mutation.
- `RecursiveStageCache` stores only validated content-addressed recursive
  stages and verifies every artifact SHA-256 before returning a hit.
- `ThreeMFWriter` serializes the standard colored multi-object package.
- `ValidationService` checks in-memory meshes and reloads the written package.
- `ValidationService` also performs offline multi-view surface/depth consistency checks; it never controls a slicer UI.

Services should be stateless where practical. Inject or replace collaborators through `SplitPipeline` rather than reaching into CLI parsing.

## Performance invariants

- Vendor paint-selector expansion is required to recover the actual material
  boundary: one source triangle may contain multiple painted leaf regions, so
  skipping conformance would change part ownership.  The expanded mesh is an
  immutable recognition/source representation, not a license to run unrelated
  global repair.  Later work must remain interface-local and use acceleration
  structures rather than repeatedly scanning every expanded face.
- Physical search convergence uses the active print tolerance and a bounded
  iteration guard; it must not chase sub-micron numerical differences that
  cannot change millimetre-scale FDM output.  Accepted geometry still receives
  the complete topology and safety audit.
- Small-impact completion applies only to generated/interface-local geometry.
  Source faces and material regions remain preserved regardless of whether
  their affected ratio is below one percent; they may be diagnosed or bypassed
  when irrelevant to an interface, never silently deleted.

- `boundary_correspondence.py` proves a bounded sampled source-loop match
  before the cap remapper increases its geometric projection allowance.
  Exact-ID handling and visible source coordinates remain unchanged.
- `subdivision_proof.py` verifies complete edge-fan coverage using actual
  replacement geometry; aggregate area alone is not a coverage proof.
- `cap_backoff.py` computes a proposed translation from one measured depth
  constraint. `cap_template.py` reuses a direction-independent conservative
  broad-phase candidate superset across retries, but every retry reruns the
  directional capsule filter, exact ray intersections, and complete thickness
  safety policy. It may publish only a result passing that fresh measurement.

- `region_review.py` partitions physical long-strip candidates and ordinary
  face-count candidates consistently for recognition, image review, and
  confirmed decisions. It measures visible source faces and never merges.
- Production preserves audited hidden annuli before optional refinement;
  direct refinement calls retain their diagnostic defaults. The CLI selects
  production preservation through `PrintTolerance`.
- An empty semantic-review list returns before global adjacency/projection
  work. Disconnected needle spans share one total affected-area budget.

- Parent-thickness ray checks use no more than 768 equal-arc samples from the
  ordered boundary input, including candidate screening and final depth
  selection. Generated cap geometry continues to use every boundary vertex.
- Parent-thickness broad phases must be conservative: active-only triangle
  buckets and segmented capsule covers may reduce exact ray/triangle tests but
  must never exclude a triangle that can intersect the finite probe segment.
- Dense polygon ear clipping may spatially query reflex vertices and update
  only neighbors of a removed ear. It must still preserve every source boundary
  edge and emit exactly `n - 2` nondegenerate faces for a simple `n`-point loop.
- Keep generic mesh cleanup unless a replacement proves both faster and
  topology-equivalent on a real vendor benchmark. A faster invalid mesh is not
  an optimization.
- Run `tools/run_inward_performance_harness.py` together with the connector
  harness and the full unit suite after changing these paths.

## Module ownership

- `cli.py`: arguments, preflight, dependency loading, exit behavior;
- `project.py`: project metadata, material slots, unit and transform resolution;
- `recognition.py`: paint decoding and component recognition;
- `explicit_merge.py`: explicit multi-material body merging after recognition;
- `interface_retreat.py`: opt-in, user-confirmed source-face ownership retreat
  across an existing child/parent seam, with deterministic geodesic growth,
  material preservation, and connectivity/size safety gates;
- `source_region_review.py`: source-region review rendering, manifest generation, and classification validation;
- `selection.py`: structural evidence and root-body selection;
- `assembly.py`: parent inference, cycle repair, recursive layers;
- `direction_field.py`, `assembly_references.py`, `interface_thickness.py`,
  `cap_planning.py`, `surface_construction.py`, `connector_building.py`, and
  `part_mesh_building.py`: separated geometry algorithms and mesh construction;
- `part_geometry.py`: explicit import surface for split geometry modules; it
  contains no geometry implementation;
- `hidden_interface.py`: pure hidden-interface candidate generation, equal-arc boundary sampling, local-inward projection, and evidence records;
- `connector_planning.py`: immutable connector depth policy, independent
  backing/engagement budgets, elastic priority allocation, and compact-footprint
  safety planning;
- `connector_geometry.py`: planar projection, line-preserving/Clipper2 inset,
  and continuous 45-degree backing lead/floor rings;
- `connector_topology.py`: constrained annulus strategies, exact boundary/area
  audit, and shared-edge face-orientation propagation. Rings with at least
  1024 source vertices get one linear nearest-seam probe before the
  constrained solver, and dense star-shaped annuli get a linear zipper probe;
  both fast paths must pass the unchanged strict audit. Dense failed solver
  cases stop before the quadratic visible-bridge fallback. Hidden backing
  annuli with at least 2048 already-audited faces bypass optional recursive
  flat-shading refinement, preserving their accepted topology and both rings;
- `projected_micro_folds.py`: isolates bounded local self-crossings in a curved
  rim's planar projection, retaining the exact 3-D ears. The topology service
  solves the reduced simple annulus, restores every ear and original boundary
  edge, then runs its unchanged full 3-D strip audit. All ears share the physical
  micro-patch area budget; no source vertices move. `tools/replay_connector_strip.py`
  replays captured strip inputs without restarting the model pipeline.
- `local_connectors.py`: deterministic compact peg/socket plans, Boolean
  sequencing, ratio audit, and the immutable per-interface assembly strategy;
- `boolean_cutters.py`: non-serialized complete-child exterior proxies,
  deterministic remote source-patch re-entry detection, and a boundary-cap
  proxy fallback which keeps generated attachments but cannot punch through a
  second parent exterior wall; one isolated area-weighted
  source-normal cancellation may be recovered from its valid one-ring or
  largest incident oriented face, while clusters and count excesses remain
  blocking source-patch defects;
- `boolean_case_cache.py`: compressed parent/cutter regression cases for the
  standalone Boolean replay harness;
- `planar_arc.py`: stable-plane equal-arc reconstruction and source-id canonicalization;
- `interface_retopology.py`: production facade for shared child/parent interface targets;
- `surface_quality.py`: reusable scale-independent source-triangle quality,
  directed shared-edge winding, and local retriangulation-rim audits used by
  visible interface retopology;
- `mesh.py`: generic mesh and boundary operations, including per-closed-shell
  outward-orientation repair so a dominant positive body cannot hide an
  inverted detached micro-shell, plus provenance-gated removal of new
  near-coplanar zero-thickness Boolean shells already covered by the retained
  body surface;
- `mesh_finalization.py`: source-face-prefix-preserving cleanup, selective
  open-boundary welding, residual closure without global source-face deletion,
  and strict topology audit;
- `source_ear_repair.py`: restores a generated closure chord onto two existing
  open source edges when a protected micro-ear was skipped. It changes only
  generated triangles, shares one physical patch budget and retains every
  source face and vertex; finalization still requires closed oriented topology.
- `package_io.py`: 3MF component-assembly serialization and package metadata;
- `validation.py`: topology, per-shell outward-orientation, and package validation;
- `debug_export.py`: strict parent-emitted-part 3MF recursion plus explicitly requested standalone colored-part and cumulative audit exports;
- `recursive_preflight.py`: whole-tree interface-plan orchestration and typed
  pass/warn/block records;
- `stage_cache.py`: run/stage fingerprints, atomic validated checkpoint
  manifests, and artifact-integrity checks;
- `reporting.py`: user and machine-readable result reporting;
- `common.py`: small shared constants and pure helpers only.

Avoid circular imports. A lower-level module must not import `pipeline.py` or `cli.py`.

Performance ownership follows the same boundaries. `part_mesh_building.py` decides
whether provisional connector cutters are needed at all; deferred recursive
parents receive only their source-plane preclosure. `connector_topology.py`
owns spatial seam lookup, linear candidate ranking, and the unchanged strict
strip audit. `local_connectors.py` keeps the accepted Manifold parent resident
across sequential differences while still exporting and auditing every
intermediate result. These optimizations may remove redundant work, but may
not weaken topology, volume, or winding gates. Default ratio validation may
preserve zero-volume Boolean seam faces up to 0.5% only while the solid remains
watertight, winding-consistent, and free of boundary or over-shared edges.

Assembly construction uses one fit policy, owned by `uniform_fit.py`:
exact complete-child subtraction first, uniform scaling after all recursive
steps. `debug_export.py` constructs exact cutter copies without overshoot or
auxiliary clearance tools. `pipeline.py` applies the final transform before
mesh/visual/package validation. Public legacy clearance switches and translated
attachment-cutter functions are removed. The geometry builders remain reusable.

Every accepted sequential Boolean is also compared with its immediate pre-cut
parent before the resident Manifold is advanced. A detached shell may be
removed only when it is absent from the parent vertex provenance, is bounded
to the documented micro-shell face/volume/thickness limits, and lies entirely
within the cover tolerance of the dominant result shell. The cleaned mesh must
pass the same watertightness, winding, collapsed-face, and Boolean-volume audit
and must be re-imported as the resident Manifold for later cutters. This is not
a generic component filter: source-supported detached shells remain immutable.

## Connector harness

`tools/run_connector_harness.py` exercises square, curved, dense concave-V, and
unequal-count ripple boundaries without loading a vendor model. Each ordinary
case calls the complete production male-backing builder in addition to the
public planning, backing-geometry, and topology services. Run it before a full
3MF:

```bash
python tools/run_connector_harness.py --case all --strategy auto
```

Every case must retain a continuous 3.0 mm backing with no vertical-skirt
fallback, keep the compact connector inside the backing, preserve both annulus
boundaries exactly, match outer-minus-hole area, record measured taper angles,
emit closed backing-clearance and compact-socket cutters in that order with
0.30 mm per-side and bottom relief, and contain no degenerate, over-shared,
needle, or annulus-leaving faces.

`tools/run_elastic_connector_harness.py` separately exercises the complete
3.0 mm backing plus 5.0 mm engagement case, engagement-only reduction,
zero engagement, backing reduction after zero engagement, and a thin-rim/
thick-center split budget. Every case must emit closed, consistently wound
synthetic child and parent solids. Zero engagement must omit both compact peg
and compact socket rather than generating coincident zero-depth rings.
The same harness also plans a 2089-point dense boundary. Polygon containment
and edge-clearance queries must use the bounded-memory vectorized service and
finish inside the harness time budget; reintroducing per-point/per-edge Python
loops is a performance regression even when the resulting geometry matches.

## General tolerance services (2.0.6)

`region_review.py` owns face-count and long-strip candidate selection. These measurements request semantic review only. The production pipeline does not invoke `micro_regions.py` or `micro_openings.py`: source regions and openings remain unchanged, while interface validation alone decides whether a split can proceed.

Boundary cycle decomposition uses a heap-backed Euler walk and an online
path/position stack, so every adjacency and trail occurrence is consumed a
bounded number of times even when thousands of simple cycles touch at one
vertex. Inward conormal evidence is accumulated once per component and reused
by every requested ring while retaining source order. `prepare_debug_directory`
returns an atomically reserved new `Path`; callers must use that returned path.
A collision never removes earlier artifacts. See
[tolerance-and-runtime-policy.md](tolerance-and-runtime-policy.md) for ratios,
reporting, and execution recovery.

Boundary inward-direction smoothing keeps its physical-radius Gaussian window
and all 32 inward-hemisphere projection rounds, but executes the closed-loop
weighted sums through compiled wraparound convolution rather than allocating
one `np.roll` array per offset. This is a performance substitution, not a
higher-resolution geometry mode: print-scale tolerances remain authoritative,
and no CAD-style sub-resolution refinement is introduced.

Strict recursion still writes every changed node as a standalone 3MF and
performs a real disk reload, package/material/provenance validation, and
SHA-256 identity check before descendants consume it. The validated parsed
object is retained by `VerifiedRecursiveArtifactCache`; later consumption uses
that exact object after identity verification instead of parsing the same 3MF
again. Stage-cache hits remain content-addressed and independently validated.
Do not remove the first disk round trip to gain speed: it is the serialization
quality gate that prevents a valid in-memory mesh from becoming an invalid
print artifact.

## Compatibility invariants

### Conditional boundary clarification

`boundary_clarity.py` builds shared-edge adjacency once per geometry assessment.
Pairwise branching seams trigger reviewed local ownership search; endpoint-only defects first use automatic nearest stable adjacent ownership and the general residual >1% warning policy;
ordinary triangulation, waviness, and valid three-material junctions do not.
This is a topology-based ambiguity detector, not a complete semantic segmentation
or self-intersection detector. Existing printable-geometry checks still apply.

`boundary_review.py` gates raw input, recognized root ownership before planning,
and every newly computed recursive stage after reloading its current mesh. Clear
boundaries return without candidate search or preview generation. Endpoint-only cases never require a review preview: `nearest_boundary.py` applies at most three improving local ownership rounds, preserves parts and connectivity, and reports residual ratios. Remaining branching cases
try at most three local weighting variants inside a two-hop face band. Candidate
labels cannot remove a part or increase its disconnected-region count; vertices,
triangles, and paint are not edited by this service. Original/generated face roles
are not inferred from color when provenance is absent.

`boundary_preview.py` emits assembly and local X-ray candidate comparisons. CLI
returns exit code 4 before cutting when no candidate can be accepted automatically.
The agent then applies completion-first impact assessment and authorized repair;
this diagnostic return does not itself require a user confirmation. Candidate topology
passing is not semantic or print acceptance. `--boundary-review-json` accepts an
explicit user-confirmed decision, bound to both input and regenerated candidate
fingerprints; a `decisions` list supports independent recursive-stage approvals.
Never set `user_confirmed` without actual user approval. Unmatched approvals do
not approve another stage. `--boundary-check-only` stops after the input gate.
Changes to these modules and approval-file contents invalidate stage caches.

Boundary mode is always `smooth`; source-mode and immutable visible-seam bypasses
have been removed. Already generated parent-contact interfaces remain shared and
immutable to preserve assembly correspondence.

`boundary_budget.py` freezes original per-region surface areas and tracks cumulative
changed faces. `boundary_simplification.py` repairs small projected crossing lobes
by local ownership transfer, retaining the mesh and paint. Strictly below 1% is the
automatic cleanup path. At or above 1%, `boundary_review.py` evaluates admissible
local branch candidates by impact: retained parts, unchanged surface and paint,
no added disconnections, and a clear boundary. Such candidates continue as
`completion_priority_local_merge`; percentage alone never forces a stop.
Complex remaining failures follow `completion-first.md` before user handoff.

Regression command: `python -m unittest discover -s tests -p test_boundary_clarity.py`.

### Conditional projected-curve clarity (2026-09-05)

Ownership topology and curve geometry are separate checks. A degree-two seam
can still generate crossing orange target segments. `planar_arc.py` now checks
the actual source-ID-ordered target before the displacement gate or surface work.
`curve_clarity.py` proposes local small-lobe removal only when projected crossings
exist: at most three width profiles, bounded loop cuts, cumulative area budget,
and no convex hull or global refit. Clear curves allocate no candidate mapping.
Sparse weights retain candidate-to-old-target provenance, not mesh vertex IDs.
The depth difference between crossing segments is reported; a 2-D crossing is
not proof of a 3-D self-intersection. Every topology-changing proposal requires
review and surface remeshing; it is never forced onto the original vertex list.

`CurveClarityRequired` reaches the context's `curve_review_sink`, which exports
actual before/after curves plus numerical NPZ/JSON through `curve_preview.py` and
raises `BoundaryReviewRequired` (CLI exit 4). Existing ownership approvals do not
approve a new geometric contour. Explicit immutable-seam mode still preserves
the user-selected original seam and bypasses spline construction. A separately
user-confirmed `apply_clear_curve` decision must match both the source/target
fingerprint and candidate fingerprint. The approved contour is resampled without
treating proposal rows as source IDs, then passed through the existing surface-band
remesher and all authoritative face, topology, connector, Boolean, and output audits.
Approval therefore permits an attempted remesh; it never approves the final 3MF.

Regression: `python -m unittest discover -s tests -p test_curve_clarity.py`.
Replay: `python tools/review_curve_clarity.py CAPTURE.npz OUTPUT_DIRECTORY`.
The replay expects `source`, `target`, and optional `guide` arrays, and plots the
actual edited polyline, not a cosmetically simplified representation.

An architecture-only refactor must preserve:

- command-line arguments, defaults, exit codes, and progress annotations;
- recognition ids, colors, body id, and assembly parents;
- generated vertex/triangle order and object annotations;
- validation decisions and atomic-output behavior;
- complete output 3MF bytes for deterministic golden inputs.

Before installing an architecture change, compile all modules, run `--version` and `--preflight-only`, then compare complete output SHA-256 values against the pre-refactor Panda and Speedboat baselines.

## Printing tolerance and replay (2.1)

`visual_ray_confirmation.py` is the narrow-phase check for centroid-screen intrusion candidates. It reuses `surface_rays.first_surface_hit` for signed depth and original face identity, and tests complete assembled triangles for occlusion. `validation.py` retains threshold policy and aggregate counters. Exact confirmation changes the evidence quality, not the acceptance limits.

Curved leaf recovery uses `scripts/repair_curved_insert.py` as a separate orchestration entry point. `curved_repair_io.py` owns normalized stage loading and oriented source matching; `curved_backing.py` owns source-following solid construction; `backing_thickness.py` owns directional thickness evidence. They reuse uniform scaling, exact Boolean operations, seating, visual validation and package I/O. `assembly_visibility.py` checks both ownership and solid intersection; `assembly_seating.py` owns the bounded axial/contact-pose search and structured failure measurements. No module identifies a repair case by filename or color name. See `curved-backing-repair.md` for supported scope and acceptance gates.

`print_tolerance.py` owns a context-scoped immutable physical tolerance. CLI selects it; geometry modules read this policy without model-name checks or task-specific approvals. `surface_preservation.py` distinguishes original triangle identity from preserved surface coverage. `edge_index.py` supports local incremental boundary closure. `quad_regularization.py` batches candidate geometry while retaining sequential flip selection. `finalization_case.py` stores non-pickle input/candidate snapshots and validated, implementation-matched local results; `tools/replay_finalization.py` replays this stage independently. Existing recursive cache remains responsible for whole successful recursive steps. Default source seams bypass unnecessary visible smoothing; optional smoothing remains explicit.
