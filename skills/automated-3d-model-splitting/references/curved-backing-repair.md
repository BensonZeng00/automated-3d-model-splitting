# Source-following curved backing recovery

Use this route for a curved leaf insert whose planar backing approaches the visible front over a broad area, or whose scaled final pose still intersects its parent. Do not select it by model name, color, or part id.

The standalone command below remains limited to two single-material parts.
For an explicitly authorized complete-tree rebuild, the ordinary split CLI now
offers `--repair-thin-backing`; see [cli-reference.md](cli-reference.md#pre-boolean-backing-repair).
That adapter retains per-face color owners and repairs before exact recursive
parent subtraction. It uses source-local normals, not the standalone command's
single axis. No failed geometry is silently accepted or published.

## Preconditions and construction

Use normalized millimeter recovery stages with one mesh and identity transforms: the complete uncut source, original full-size insert, and parent template. The command currently supports a two-part assembly with single-color parts. Do not apply it to nested children or multi-material templates. Match source triangles by coordinates **and oriented normals**; coincident reversed backing triangles are not source faces.

`curved_backing.py` retains the matched source front and boundary, and offsets a conforming duplicate backing inward. Depth is bounded by distance to the rim, preferred depth, and measured parent exit minus 0.05 mm. Interior edges whose endpoints both lie on the rim need conforming midpoint splits; otherwise sharing front/back rim vertices can create four-owner edges. Missing parent exits, inverted/degenerate geometry, or failed closed-solid checks block the repair.

`backing_thickness.py` reports directional source-to-backing chords and area-weighted coverage. Separate the natural taper within a 0.6 mm rim band from the interior; do not call this a global minimum-wall certificate. The default interior target is 0.45 mm after 99% scaling. Missing eligible measurements and broad thin regions are failures.

Subtract the complete full-size rebuilt insert from the complete source, then scale the emitted insert by 0.99 about its bounding-box center. Apply the bounded rigid seating policy in [uniform-fit.md](uniform-fit.md). Do not enlarge the socket or relax collision thresholds.

## Run and acceptance

Run from the skill directory with the configured Python environment. The single-line command works in PowerShell and POSIX shells; use `python3` on macOS when that names the selected interpreter.

```text
python scripts/repair_curved_insert.py --source recovery/editable_source_unscaled.3mf --insert recovery/full_size_insert.3mf --parent recovery/parent_template.3mf --output recovery/repair_run/repaired_assembly.3mf
```

Use a new output namespace. `--max-depth` defaults to 3 mm; use the recorded original depth where appropriate. `--taper-slope` defaults to 1 (45 degrees); any changed slope must pass the same thickness and visual checks. Do not repeatedly sweep parameters without inspecting the recorded failure.

Acceptance requires thickness coverage, source containment measurements, watertightness, winding, rigid seating overlap at most 1e-8 mm³, source-surface ownership, unchanged multi-view material checks, palette preservation, and package reload validation. Report a failed multi-view check even if the overlap is zero. Never advertise such a candidate as printable or repaired.

The centroid depth-map screen can flag hidden backing through gaps in its sparse source samples. Before diagnosing these flags as exposed sidewalls, `visual_ray_confirmation.py` verifies each flagged surface point against complete source and assembled triangles. A nearer assembled surface proves occlusion; a source hit within the existing depth allowance proves no excessive intrusion. Source ownership comes from the actual hit triangle. Keep the same depth/pixel thresholds, retain unresolved missing-source hits, and record how many flags were dismissed and why. This confirms flagged samples; it is not an exhaustive wall or per-pixel visual certificate. Regression must include a real exposed cap that still fails.

The command always writes a JSON report with source hash, timings and failure details. After seating it saves a clearly unvalidated NPZ candidate for local inspection. Only passing candidates produce final 3MF files. Keep failed candidates in the recovery directory; never replace the user source.

Regression: `python -m unittest discover -s tests -p test_curved_backing.py`, existing visibility and uniform-fit tests, plus connector, planar-arc and inward-performance harnesses. Synthetic passes establish general invariants; they do not override a real model's visual failure.
