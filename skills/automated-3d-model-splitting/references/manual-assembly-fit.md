# Default assembly observation handoff

The default `--assembly-ignore-overlap-ratio 0.01` silently accepts a measured
total intersection volume **strictly below 1% of the corresponding cutting volume**.
This is volume divided by volume. The denominator is the actual intersection
removed by that original sequential full-size child cutter, frozen before
uniform scaling or final fitting. It is not the whole body or cutter volume.
`assembly_cutting_reference` preserves the parent and measured denominator.
Direct parent/child pairs use their interface's reference. For other pairs,
use the smaller incoming cutting volume, excluding a root's absent incoming
cut. Missing references on non-root parts or a zero denominator disable ratio
acceptance; do not invent a geometric proxy or restore the old 1 mm³ default.
Sum all disconnected intersection fragments for that pair before comparison.
Do not aggregate unrelated pairs, including during coupled seating. Keep the
measured volume in machine-readable records, but do not cut or move parts,
mark manual adjustment, or notify the user solely for these ignored overlaps.
Values indistinguishable from the boundary by floating-point roundoff count
as the boundary. `0` disables this physical acceptance policy. It is separate
from Boolean numerical precision (`1e-8 mm³`), thickness, surface area and
visibility gates. Exact full-size socket construction remains unchanged.

An overlap at or above the threshold still enters normal difference/seating
checks. If unresolved, finish the export and provide the measured volume;
ask the user to observe the model and judge whether it affects printing,
adjusting manually only if necessary. Do not equate a fit discrepancy with
a failed print. Use “请观察模型并判断是否影响打印，必要时手动调整。”

`--assembly-fit-validation manual` is the production default. Execute the
normal backing checks, exact full-size parent subtraction, final 99% scaling
and default post-fit ancestor difference first. If bounded automatic seating
cannot solve the fit, retain the valid meshes and finish exporting the split
assembly. Do not ask for another approval or stop solely for fit interference,
front occlusion or assembled-view mismatch. The user will adjust these manually.

Keep the raw intersection volumes, source-front coverage and multi-view errors.
An unresolved check remains `valid=false`; export is not proof of fitted assembly.
Mark the package title and every object's `assembly_fit` annotation as
`manual_adjustment_required` (retained machine status); use `ASSEMBLY REVIEW`
in the package title. Explain the affected parts and issues in the final
response using the observation wording above, and link the final 3MF.
Never call an unresolved result assembly-validated.

This policy does not accept invalid individual meshes, wrong colors or slots,
lost source surfaces, failed backing thickness, empty difference results or
disconnected parts. Rejected Boolean candidates never replace valid inputs.
The usual topology, winding, source-surface, thickness and package-reload checks
remain active. Thin backing requires geometry repair before assembly handoff.

`--assembly-fit-validation strict` restores blocking on unresolved fit and
assembled-view errors. Coupled seating remains separately optional; a missing
coupled-seating opt-in now produces a measured manual handoff in default mode.
Strict mode still uses the physical overlap policy; also specify
`--assembly-ignore-overlap-ratio 0` to test numerical-only intersections.

## Local replay

Before final assembly checks, `--recovery-dir` saves `assembly_case/case.json`
and `assembly_case/scaled_inputs.npz`. The checksum-bound pair contains source
vertices/faces/part labels, all scaled part meshes, materials, annotations,
tree identities and effective options. They are already scaled: never scale
these inputs again. `assembly_result.json` stores the complete fit report.

Use `python tools/replay_assembly.py --case <assembly_case> --output <fresh-dir>`
to run only fit checks and repair. `--strict` requests a blocking replay. The
tool writes reports and new recovery inputs; it does not claim to publish a
validated final package. Retain source-slot and topology verification when
integrating replayed geometry into a final assembly.
Replay preserves a saved ratio when present, otherwise uses the current
default. `--assembly-ignore-overlap-ratio` explicitly overrides it. Legacy
fixed-volume options are discarded. Old cases without cutting references need
their original Boolean measurements restored before ratio acceptance can apply.
