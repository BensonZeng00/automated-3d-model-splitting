# CLI Reference

## Assembly validation and manual handoff

`--assembly-fit-validation manual|strict` defaults to `manual`. After normal
geometry repair and post-fit difference, unresolved seating and assembled-view
issues are reported for user observation and judgment about printing, with
manual adjustment only if needed, and do not stop final export. Raw
checks remain failed and the package is labeled `manual_adjustment_required`.
Individual geometry, thickness, colors and package reload still must pass.
See [manual-assembly-fit.md](manual-assembly-fit.md) for recovery and replay.

## Final seating tolerances

`--assembly-ignore-overlap-ratio` defaults to `0.01` (1%). A pair's total
intersection divided by its original actual cutting volume must be strictly
below this fraction to be accepted silently, without
post-fit cutting, seating or a manual-adjustment warning solely for that
overlap. Sum disconnected fragments per pair. Values at the boundary still
enter normal checks; machine-roundoff-near-boundary values count as equal.
`0` disables this policy, independently of `manual|strict` export behavior.
Retain both volumes and their ratio in machine reports and all other
geometry/visibility gates. The denominator is frozen from the original
sequential socket Boolean before scaling and seating. Missing/zero cutting
references disable ratio acceptance. The former fixed-volume CLI flag is
removed; see [manual-assembly-fit.md](manual-assembly-fit.md) for pair selection.

`--seating-penetration-tolerance-mm 0.01` enables a conservative local
intersection slab-thickness bound in millimeters. Every connected intersection
must fit within that width and the summed intersection surface area must stay
within `--micro-defect-area-mm2`. This is not a measured maximum penetration
depth. Default zero disables this physical acceptance rule.

`--seating-overlap-tolerance-mm3` retains a separate numerical volume threshold
(default `1e-8`). Length and volume must never be converted by relabeling units.
Accepted nonzero intersections remain recorded; ownership checks still apply.

Confirmed coupled seating processes parents before children, solves the union
of a branch against all other parts, checks cumulative travel, and rechecks all
final pairs and ownership before committing poses. `--recovery-dir` retains
failure details and explicitly scaled input NPZ snapshots. These snapshots are
diagnostic inputs, not unscaled recursive parts or printable deliverables.

Current release: `2.1.0`.

`--hidden-surface-refinement preserve|refine` defaults to `preserve`: keep
already-audited hidden annulus triangles without density subdivision. The
`refine` option retains the prior edge-length refinement diagnostic. Neither
option skips downstream solid, wedge, thickness, color or visual validation.

Under the default semantic review policy, long-thin exposed regions also
require confirmed image-review decisions regardless of their triangle count.
The physical screening thresholds and their interpretation are in SKILL.md.

The only fit policy is [exact subtraction then XYZ uniform scaling](uniform-fit.md).
Use `--post-split-uniform-scale 0.99` (default 0.99); this is a numeric factor,
not a mode selector. Legacy fit/flat/bottom/sibling clearance options and
`--clearance-profile`, `--clearance-feature-ratio`, `--clearance-min-mm`, and
`--clearance-mode` are removed from the public parser. Their historical
descriptions below are not supported command options in this revision.

Use `python scripts/split_painted_3mf.py --help` as the source of truth.

## Input and Safety

- `--input PATH`: required source 3MF.
- `--output PATH`: final colored 3MF; default `<source>_split_parts.3mf`.
- `--format-profile auto|vendor-paint`
- `--model-entry PATH`
- `--preflight-only`
- `--recognize-only`
- `--overwrite`
- `--full-tree-preflight` / `--no-full-tree-preflight` (enabled by default)
- `--resume off|auto|strict` (default `off`)
- `--cache-dir PATH` (required when resume is enabled)
- `--debug-recursive-3mf`
- `--debug-recursive-stl` (deprecated compatibility alias)
- `--diagnostic-preview`

Whole-tree preflight plans every recursive interface before the first expensive Boolean and exits with code `3` on a blocking thickness, cap, or interface decision. Resume checkpoints are content-addressed and include the source, normalized geometry configuration, recognized component identities, assembly tree, parent artifact hash, and relevant implementation fingerprint. Cached standalone 3MF files are hash-, provenance-, color-, and topology-checked before use. Cache flags cannot be combined with recursive debug export because debug mode must materialize the complete deterministic trace.

## Exterior Recognition

- `--recognition-surface-profile exterior-visible|all-faces`
- `--exterior-view-count 32`
- `--exterior-depth-map-resolution 768`
- `--exterior-depth-tolerance-mm 0.08`

`exterior-visible` reassigns occluded paint to the recognition base color before connected-region grouping. It never edits the source mesh.

Before either recognition profile runs, vendor composite `paint_color` streams are decoded into their actual recursively split leaf faces. The parser conforms T-joints along internal and neighboring source-triangle edges, then reports source faces, decoded leaves, conformed faces, leaf states, and parse failures. Review face thresholds apply to restored leaf-face connectivity, never to raw composite token strings.

## Processing Mode

- `--part-processing-mode auto|inward`: default `auto`; both use inward geometry for every non-body part.
- `--part-mode-overrides P10=inward`: legacy-compatible inward-only override.
- `--accept-ambiguous-inward`: retained compatibility flag; geometry is already inward-only.

Automatic body selection records normal coherence, opposite-normal balance, dominant-normal separation, component span, loop span, through-like score, shell-removal balance, and large structural interfaces. Through likelihood contributes only to strong-separator evidence; it is not an unconditional penalty after a candidate fails the balanced shell-split gate. These measurements affect only body choice; they never create a through geometry path.

`auto-score` compares every eligible component regardless of raw paint token, resolved color, filament slot, or `DEFAULT` status. Color filtering is allowed only through an explicit `--body-color` override.

## Components and Colors

- `--noise-review-max-faces 100`
- `--small-region-review-max-faces 999`
- `--region-review-json PATH`
- `--region-review-dir PATH`
- `--region-review-resolution 320`

The face thresholds select mandatory semantic-review candidates only. Regions through 100 faces are noise candidates, regions from 101 through 999 faces are small-region candidates, and long strips require review at every face count. Confirm every candidate as `noise`, `part`, or `uncertain`. All classifications remain unchanged source geometry and follow the same normal interface-planning path. No classification merges, deletes, recolors, repairs, or filters a region. The former `--min-faces` and `--tiny-component-*` interface is intentionally unsupported.

High-confidence part semantics may include `force_inward_vector: [x,y,z]`, `force_parent_direction: true`, or a `guided_internal_cut` object. A guided cut supplies `entry_direction`, `target_plane_normal`, `target_plane_point_mm`, optional `minimum_depth_mm`, `maximum_depth_mm`, `entry_inset_mm`, and `maximum_parallel_shift_mm`. The entry vector is the front internal transition; the plane is the deeper shared cut. The planner keeps the visible rim locked, localizes the entry inset to measured thin arcs, searches bounded parent-interior variants, reuses one source-id fit ring on both parts, and blocks poor side-wall triangulation. These are audited geometry constraints, not model-axis defaults.

The default inward planner automatically invokes a hidden-interface fallback when the baseline safe depth is below the 0.45 mm load-bearing minimum. It evaluates deterministic locally inward fields aimed at interior parent targets, reruns the full thickness and cap audit, keeps the visible source rim unchanged, and applies the selected field symmetrically to the insert and socket. This is automatic and adds no model-specific CLI option.

## Inward Geometry and Fit

- `--post-split-uniform-scale 0.99`: scale complete final inward parts after exact full-size subtraction, around each own bbox center.
- `--interface-geometry local-connector`: sole public interface strategy.
- `--max-extension-mm 3.0`, `--max-planar-travel-mm 10.0`: preferred depth and measured safety ceiling.
- `--cap-mode flat|adaptive|tilted|offset`, `--nested-cap-mode inherit|flat|adaptive|tilted|offset`.
- `--lead-in-mm 0.60`: backing/peg geometry, not socket enlargement.
- `--assembly-mode tree|flat|legacy-flat`, `--assembly-tree-strategy recursive-minimal|strongest-path`.
- `--output-layout assembly|separate-items`, default assembly.

All pre-cut fit/floor/sibling clearances are zero. No fit-allocation mode,
radial overcut, or translated clearance cutter is available. See [uniform-fit.md](uniform-fit.md).

## Boundary shape and planar arc retopology

`--boundary-shape smooth` is the sole supported value and default. Shared seams use the spline and surface-band behavior below; see [boundary-smoothing.md](boundary-smoothing.md). Source ownership is never changed by seam smoothing.

`--seam-smoothing-profile source-conservative|print-balanced|print-smooth` selects one coherent topology and shape-quality budget; `print-balanced` is the default. `--maximum-boundary-displacement-mm` independently sets both the maximum and P95 seam-motion limit and defaults to 10 mm. `print-balanced` permits at most 1% of source area, at most `min(5000, 5% of part vertices)` collateral vertices, at most 8 topology layers, edge stretch up to 16x, and introduced sparse normal reversals up to 0.10% when every connected cluster has at most 32 faces, every result angle is at least 0.01 degree, including isolated seam-adjacent faces created while smoothing a pitted boundary. Existing source defects are diagnostic and do not consume the introduced-defect budget. Open edges, over-shared edges, inconsistent directed topology, degenerate faces, over-budget seam-contact reversals and larger folded clusters remain blocking. `source-conservative` tightens these budgets; `print-smooth` expands them for explicitly print-first smoothing.

A hidden generated-backing thin patch no larger than 2 mm² is recorded as a print-scale advisory and does not block or trigger source reconstruction. Larger generated-backing failures still require the explicit `--repair-thin-backing` workflow; this exception never authorizes changing source faces.

- `--boundary-target-samples 384`
- `--boundary-smooth-passes 28`
- `--boundary-retopology-band-mm 3.0`
- `--connector-slope-validation strict|advisory` (default `advisory`)
- `--surface-band-validation strict|advisory` (default `strict`)
- `--connector-surface-validation strict|advisory` (default `strict`)

`--surface-band-validation advisory` is reserved for a surface band the user has visually reviewed. It reports source-relative normal changes instead of treating them as geometric inversion, allows the target to use up to 60% of the requested real band, and replaces area/vertex coverage limits with the actual displacement envelope only when both maximum and P95 displacement remain within `--maximum-boundary-displacement-mm` (default 10 mm). It locks isolated shared-loop junction vertices to their original source positions, lowers the minimum result-angle floor for an otherwise eligible sparse isolated inversion from 3 degrees to 0.01 degree, raises the bounded edge-stretch ceiling from 8x to 128x, reuses a source-ID-matched prevalidated hidden fit ring when a later equivalent surface-band pass changes only its local conormals within the reviewed band offset, permits up to `0.00005 mm³` numerical volume drift when removing already accepted redundant Boolean micro-shells, and permits at most `0.005 mm³` kernel-to-export volume roundoff only when the audited Boolean is watertight, winding-consistent, has zero boundary/over-shared edges, and has zero degenerate faces. Its final local material-pair raster check blocks only when both the per-source-part ratio and accumulated absolute-pixel budgets are exceeded; a one-budget exceedance is recorded as a user-reviewed advisory so tiny parts are not rejected by an inflated percentage alone. Watertightness, winding, edge topology, collapsed-face ratio, exact boundary match, substantive Boolean validation, and assembly validation remain blocking. Every exercised advisory is recorded in the quality report.

`--connector-surface-validation advisory` is reserved for user-reviewed geometry. It retains an already topology-audited hidden connector annulus without optional flat-shading subdivision and records the measured internal-edge resolution. Together with connector-slope advisory, it may retain the intrinsic high-aspect side triangles of a peg-free minimal closure, whether produced by the constant-offset or axial-fallback branch, only when the rim has at most four vertices, depth is at most `0.05 mm`, strip topology is present and valid, no bridge leaves the annulus, fanout is at most 64, and reported needle edges are at most `8 mm`. Degeneracy, topology, winding, thickness, cutter, Boolean, and assembly checks remain blocking.

The generated interface target is 45 degrees. By default, only a strictly greater than 1% share of finite slope samples outside inclusive 30–75 degrees on the current interface produces a warning. Exactly 1% is recorded without warning. Median, central 90%, and cyclic-run diagnostics do not block advisory mode. Explicit `strict` retains the previous central-90%, 2%-outlier, and cyclic-run acceptance checks for diagnostic use.

`advisory` is the general default and keeps slope measurements in the report without making departures fatal. Degenerate, inverted, non-manifold, Boolean, and assembly checks remain blocking.

In `smooth` mode, the implementation fits one stable local SVD plane, resamples the projected closed loop by equal physical arc length, fits a 24-control periodic cubic B-spline in-plane plus an independent 8-control height spline, and evaluates them at locally regularized source physical-arc phases. It does not restore projected source extrema. Source-id canonicalization makes reversed child and parent winding deterministic. The visible shared seam is the fitted target and its displacement diffuses through the topology-connected surface band. A target requiring more than 45% of the configured band width is rejected, and the algorithm never falls back to a rougher curve.

A three-edge closed micro-loop is preserved exactly and recorded as `exact-minimal-loop`: a periodic cubic spline cannot be defined without inventing geometry. The surrounding topology, cap, Boolean, and assembly audits still run normally.

Default cap and nested-cap modes are `adaptive`. For each boundary and candidate inward field, the implementation measures parent thickness and computes `safe maximum = min(10 mm, parent thickness - 0.05 mm)`. The preferred minimum is 3.0 mm unless parent thickness leaves less safe depth. It first tries the deepest coherent plane whose point depths may differ while remaining within both the effective minimum and safe maximum. When no such plane fits, it uses local-offset at the measured safe maximum. The subsurface entry ring derives its depth from the actual lateral fit offset for an approximately 45-degree outer-large, inner-small taper, capped by `--lead-in-mm`. An explicit `--planar-extra-limit-mm` can narrow the global ceiling. There is no CLI fallback for center-fan caps; unsafe synthetic-hub closure is forbidden.

`--debug-recursive-3mf` writes every changed part as an independent colored `_mm.3mf` and one complete `_CUMULATIVE_mm.3mf` audit after each strict depth-first recursion step. A pending child's independent file remains in the parent layer where it was emitted and becomes that child's later input. Per-triangle colors and original filament-slot indices are preserved. Cumulative packages show complete assembly state but are never recursive inputs. The old `--debug-recursive-stl` spelling invokes the same 3MF behavior and emits no STL.

## Validation

- `--validation-profile ratio|strict`: default `ratio`.
- `--max-topology-defect-ratio 0.001`: default 0.10%.
- `--visual-validation-profile strict|report|off` (default `strict`)
- `--visual-validation-view-count 32`
- `--visual-validation-resolution 384`
- `--visual-depth-tolerance-mm 0.12`
- `--visual-max-intrusion-ratio 0.04` (2% remains an advisory level)
- `--visual-max-material-mismatch-ratio 0.03`
- `--visual-max-local-material-mismatch-ratio 0.10`

Ratio mode retains the 0.10% ordinary limit. A built-in localized-short-open-edge branch can accept up to 0.20% only for consistently wound, pure open-edge findings that also pass scale-relative single-edge and total-length limits. Over-shared and inconsistent-orientation edges always block.
- `--visual-min-coverage-ratio 0.65`

Visual validation is deterministic and offline. Do not use Computer Use to inspect a slicer. When human slicer confirmation is needed, ask the user to provide screenshots of the loaded final file, expanded assembly tree, and sensitive detail.

Exit code `3` means mesh or package validation failed. Exit code `4` means mandatory small-component image review is waiting for user confirmation. Default execution writes one final 3MF; review PNG/JSON artifacts are generated for sub-threshold or long-strip candidates requiring confirmation. Recovery artifacts follow [manual-failure-handoff.md](manual-failure-handoff.md); optional previews still require the corresponding switch.

Keep `--max-topology-defect-ratio` at or below `0.001` for publication. Do not increase it to rescue generated geometry that failed the default threshold.

## Print tolerance and stage replay

- `--boundary-shape smooth`: sole supported mode, enabled by default; `source` is rejected.
- `--micro-defect-area-mm2 1.0`: whole affected surface patch threshold.
- `--print-surface-tolerance-mm 0.05`: sampled local source-surface distance limit.
- `--recovery-dir PATH`: persistent finalization input/candidate snapshots and local validated results.
- `python tools/replay_finalization.py --case PATH`: replay a logged finalization case without decoding or recursive planning.
# Default post-fit parent difference

## Pre-Boolean backing repair

Complete local-connector children now receive an all-source-face-centroid
local-normal thickness screen before their parent is cut. The 0.6 mm rim taper
band is excluded; the remaining ray segments must reach 0.45 mm after the
configured final scale, subject to the existing single affected-area budget.
This is a bounded directional screen, not a global minimum-thickness proof.

`--repair-thin-backing` explicitly enables source-following reconstruction for
failed backing. Valid children are unchanged. Failed children retain every
source triangle and material, using locally inward vertex normals and a
60-degree rim-distance taper capped by 3 mm and parent exit minus 0.05 mm. The candidate
must pass topology, connectivity, complete-parent containment and the repeated
thickness screen, followed by an isolated exact socket/scale/difference check.
That precheck does not replace the final complete-assembly validation.
It replaces the old hidden backing/peg; the ordinary recursive
pipeline subtracts this new full-size solid and serializes it before recursion.
The two-part recovery command's restrictions do not apply to this pipeline
adapter: source-face material owners are retained and pending subassemblies
continue through their original tree. Final visual validation cannot be off.
Failures retain NPZ/JSON evidence; no final package is published on failure.

## Post-fit subtraction

`--post-fit-parent-difference` is enabled by default after the full recursive
split and final uniform scaling, before seating. In tree order it subtracts
the final root and other ancestors from each scaled insert, leaving ancestor
meshes unchanged. It does not enlarge cutters or reinterpret the 0.01 mm
penetration tolerance as cubic millimeters. Sibling collisions remain blocking.

Retained faces keep Boolean face provenance and filament slots; new cut faces
inherit their nearest original insert surface's material. Empty solids, changed
material-component counts (excluding cavity shells), volume errors, remaining intersection, and failed source-surface
or thickness checks block publication. Thickness uses 512 equal-area source
centroid samples along local normals, excluding the 0.6 mm rim band; it is not
a global minimum-wall-thickness proof. The 0.45 mm interior threshold and the
configured affected-area budget stay active. Final seating, all-pair collision,
multi-view validation and package reload must still pass.

Use `--no-post-fit-parent-difference` for an explicit diagnostic opt-out.
While enabled, final visual validation cannot be `off`; disabling it requires
the explicit difference opt-out and no other enabled shape repair.
It records `post_fit_difference` separately from
the original 99% scaling and retains unvalidated candidates in the recovery
directory. With `--resume off`, no prior recursive geometry cache is reused.

When a physical penetration tolerance is explicitly enabled, an intersection
already inside both its containing-slab bound and the summed area budget is
retained and recorded as `tolerance_accepted_contacts`, without a needless
Boolean. The slab screen first uses PCA, then up to 256 actual face-normal
planes for small rejected patches. Every proposed slab contains all component
vertices; this tightens the conservative bound, not the acceptance threshold.
The final assembly check uses the same bounds. Contacts outside either limit
still require repair; numerical volume checks are unchanged for differences.

Post-fit differences reuse the existing provenance/cover-gated micro-shell
cleanup with unchanged limits (100 faces, 1e-5 mm³, 0.01 mm equivalent thickness,
0.05 mm cover distance). Retained face indices preserve material ownership.
Removed volume is recorded separately from the pre-cleanup Boolean partition
audit. Source-supported or non-covered fragments remain blocking.

The assembly orchestrator may defer a rejected difference to the existing
bounded rigid seating check, retaining the valid input, never the failed
Boolean. This does not accept a collision or a failed volume partition.
Non-leaf seating still requires explicit coupled-seating authorization; all
descendants move together, total travel remains at most 0.1 mm, and final
all-pair, ownership, visual and package gates must pass before any commit.

Independent intersection and difference computations can disagree at nearly
coplanar Boolean seams. The mass-balance check retains its numerical floor
(`max(1e-6 mm³, input volume * 1e-9)`). Beyond that floor, a discrepancy is
accepted only up to `1e-5 mm³` and only after constructing the symmetric
difference of the reconstructed partition with the input, bounding every
connected discrepancy patch by the explicitly requested penetration tolerance
and the summed area budget. Kernel-to-export volume agreement and final
parent intersection are checked separately. No unmeasured tolerance increase
or additional assembly clearance is introduced.
