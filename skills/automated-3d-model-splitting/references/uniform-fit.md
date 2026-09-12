# Sole fit policy: exact subtraction, then uniform scaling

The user selected this as the only assembly-fit workflow. Do not offer or
silently restore socket overcut, feature-adaptive clearance, or translated
backing-clearance cutters.

1. Recognize parts and build full-size inward male solids with the ordinary
   boundary, thickness, and backing algorithms.
2. Subtract a private exact copy of each complete emitted male solid from its
   closed parent. Do not scale the cutter, replace its visible patch with a
   proxy cap, overshoot its rim, or add separate clearance/socket cutters.
3. Complete the full recursive split with these unscaled solids.
4. Only afterward scale each final inward part uniformly in XYZ around its own
   bounding-box center. Leave the root body unchanged. Scale the entire part,
   including its exterior; do not substitute an internal-only deformation.
5. By default, subtract final ancestor solids from each scaled insert in tree
   order before seating, silently retaining per-pair intersections below
   `--assembly-ignore-overlap-ratio` (default 1% of the original actual
   interface cutting volume). Preserve ancestor meshes and audit each candidate's
   topology, thickness, source surface and material ownership. Record this
   separately as `post_fit_difference`; use `--no-post-fit-parent-difference`
   only for an explicit diagnostic opt-out. This cannot repair thin backing.
6. Validate and export the final assembly, preserving all colors and
   filament slots. Historical interface measurements describe pre-scale
   construction; `post_split_uniform_scaling` records the final transform.

`--post-split-uniform-scale` controls only the factor, default `0.99`. Values
must be finite and strictly between zero and one. It is not a mode switch.
All pre-cut fit, floor, flat, and sibling clearances are zero. The old public
clearance options are removed. The public interface builder is local-connector.

Scaling about a center does not guarantee containment for concave solids.
Measure final parent/child interference and visible-seam gaps when reviewing a
new model. Watertight individual parts do not prove a gap-free assembly.
Report measured issues honestly and request slicer screenshots if needed.

For a visibly obstructed final leaf insert, a bounded rigid seating correction
may follow scaling. `assembly_visibility.py` measures exact parent/insert ray
ownership over equal-area interior samples, independent of the legacy visual
depth allowance. If cover exceeds both 1% of checked samples and the configured
affected-area budget, `assembly_seating.py` searches outward along the measured
source-front axis first. Overlap outside the physical volume policy also triggers this check,
even when the visible patch is unobstructed. If the axial search fails, search
rigid translations inside a 60-degree outward cone with total travel at most
0.1 mm. Re-measure the resulting solid; optimizer success alone is insufficient.
Each final pair must satisfy the configured physical overlap policy (or the
separate numerical tolerance, 1e-8 mm³, when the physical rule is disabled)
and pass the ownership check again. Record the rigid translation separately
from the 99% scale. Never modify the cutter, parent, or insert shape. Never move
a non-leaf independently of its children. Unresolved seating is recorded for
user observation and judgment about printing, adjusting only if needed;
`--assembly-fit-validation strict` blocks unresolved issues.
See [manual-assembly-fit.md](manual-assembly-fit.md).

Before subtraction, use the complete current recursive input as the parent
solid. A rim-only temporary cap can bridge over a recessed source patch and
survive exact subtraction as an opaque membrane. `boolean_parent.py` preserves
the actual curved source patch and inherited contact shell until the Boolean.

Micro-shell deletion is optional: apply it only if its result passes the
unchanged audit. Otherwise retain the already validated pre-cleanup Boolean
mesh and resident solid together, and record the rejected cleanup. Never
silently accept failed topology or volume checks.

Implementation: `uniform_fit.py` owns configuration, exact cutter copies, and
post-split transforms; `debug_export.py` owns actual recursive subtraction;
`pipeline.py` applies scaling after all recursive steps and before final checks.

Regression: `python -m unittest discover -s tests -p test_uniform_fit.py` and
`python tools/run_connector_harness.py --case all --strategy auto`.
