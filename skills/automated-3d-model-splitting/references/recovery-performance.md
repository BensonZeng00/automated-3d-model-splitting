# Local recovery and bounded-memory export

Use these helpers when replaying a failed backing or complete-parent cache.
They improve execution and diagnostics; they do not establish a correct split
tree, matching shared seams, printable backing, or a valid final assembly.

## Exit queries and early rejection

`LocalRayProbe` bounds candidates with a union of short enclosing spheres
along the complete finite ray. It retains the original outward-triangle,
barycentric, epsilon and travel-bound predicates. Triangle-radius buckets
prevent a large cap from widening every query.

`parent_ray_probe.parent_exit_distances` uses the native kernel ray query only
on a nonempty, valid closed parent. Older kernels without `ray_cast` use the
exact segmented probe. The local-normal curved backing path checks parent
exits before expensive rim-distance calculation and retains the 0.05 mm
reserve. `ParentSafetyError.record` identifies the first rejected sample,
point, direction, exit distance, required reserve and tested/total counts.
`complete=false` means a partial failed audit, never full thickness coverage.
The backing recovery adapter persists this record as `failure_detail`.

## Recover the intact current parent

For a cached **current** complete parent, use:

```text
python tools/repair_complete_parent.py --input <complete_parent.npz> --output-dir <new-recovery-directory>
```

The NPZ must contain millimeter `vertices` and `faces` in assembly coordinates.
The helper reuses source-preserving finalization, records source SHA-256,
before/after topology and kernel status, and conservatively bounds all added
or retriangulated surface area by the current micro-area budget. It writes
`intact_parent_validated.npz` only on success, plus `report.json` on accepted
or rejected geometry. Original files remain unchanged. Existing output
directories are rejected to avoid overwriting a recovery checkpoint.

This is an opt-in geometry-cache repair, not a colored deliverable or a cut
body. It does not replace the recursive parent's ordinary preconditions.
Do not substitute the original unsmoothed whole model for a current parent
whose child seams have moved. Reconcile shared surface coordinates first;
then subtract exact full-size children and repeat the normal color, topology,
backing, mating-interface and assembly checks. Closure alone proves none of
those downstream properties. Broad missing surfaces remain failures.

## Export and reload

The ordinary `export_colored_parts_3mf` now spools mesh XML to a temporary
binary stream, then copies bounded chunks into the final ZIP. It preserves
the existing serialization-grid orientation step, coordinate precision,
face order, per-face colors, paint tokens, palette order, component layout
and Bambu metadata. No extra XML tree is allocated per vertex or triangle;
the caller's mesh arrays and ordinary validation still consume memory.

`load_colored_mesh_objects_3mf` supplies `color_hex`, source slot and color
resolution provenance as well as per-face colors, allowing direct re-export
without missing-field errors or collapsing duplicate-color filament slots.
Metadata color labels do not override the palette's actual object index.

A successfully reopened package proves serialization, not manufacturability.
Do not reclassify failed candidates as accepted parts merely by packaging all
of them together. Existing diagnostic-preview authorization and acceptance
requirements remain in force.

Regression coverage: `test_recovery_ray_queries.py`,
`test_complete_parent_repair.py`, `test_streamed_package_roundtrip.py`,
plus the existing backing, recursive, package and assembly tests.
