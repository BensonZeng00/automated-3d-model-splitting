# CLI Reference

Current release: `1.3.5`.

Use `python scripts/split_painted_3mf.py --help` as the source of truth.

## Input and Safety

- `--input PATH`: required source 3MF.
- `--output PATH`: final colored 3MF; default `<source>_split_parts.3mf`.
- `--format-profile auto|vendor-paint`
- `--model-entry PATH`
- `--preflight-only`
- `--recognize-only`
- `--overwrite`
- `--debug-recursive-3mf`
- `--debug-recursive-stl` (deprecated compatibility alias)
- `--diagnostic-preview`

## Exterior Recognition

- `--recognition-surface-profile exterior-visible|all-faces`
- `--exterior-view-count 32`
- `--exterior-depth-map-resolution 768`
- `--exterior-depth-tolerance-mm 0.08`

`exterior-visible` reassigns occluded paint to the recognition base color before connected-region grouping. It never edits the source mesh.

Before either recognition profile runs, vendor composite `paint_color` streams are decoded into their actual recursively split leaf faces. The parser conforms T-joints along internal and neighboring source-triangle edges, then reports source faces, decoded leaves, conformed faces, leaf states, and parse failures. `--min-faces` applies to restored leaf-face connectivity, never to raw composite token strings.

## Processing Mode

- `--part-processing-mode auto|inward`: default `auto`; both use inward geometry for every non-body part.
- `--part-mode-overrides P10=inward`: legacy-compatible inward-only override.
- `--accept-ambiguous-inward`: retained compatibility flag; geometry is already inward-only.

Automatic body selection records normal coherence, opposite-normal balance, dominant-normal separation, component span, loop span, through-like score, shell-removal balance, and large structural interfaces. Through likelihood contributes only to strong-separator evidence; it is not an unconditional penalty after a candidate fails the balanced shell-split gate. These measurements affect only body choice; they never create a through geometry path.

`auto-score` compares every eligible component regardless of raw paint token, resolved color, filament slot, or `DEFAULT` status. Color filtering is allowed only through an explicit `--body-color` override.

## Components and Colors

- `--min-faces N`
- `--tiny-component-policy merge|ignore`
- `--body-strategy auto-score|largest|none`
- `--body-color CODE`
- `--body-index N`
- `--color-map-json PATH`
- `--visual-semantics-json PATH`
- `--visual-semantic-min-confidence LOW|MED|HIGH|0..1`

High-confidence part semantics may include `force_inward_vector: [x,y,z]` or `force_parent_direction: true`. These are audited geometry overrides, not model-axis defaults.

## Inward Geometry and Fit

- `--max-extension-mm 1.0` (preferred effective minimum)
- `--max-planar-travel-mm 5.0` (global safety ceiling)
- `--cap-mode flat|adaptive|tilted|offset`
- `--nested-cap-mode inherit|flat|adaptive|tilted|offset`
- `--planar-extra-limit-mm N` (optional narrower compatibility override)
- `--force-flat-parts P06,P12`
- `--fit-clearance-mm 0.30`
- `--clearance-profile feature-adaptive|fixed` (default `feature-adaptive`)
- `--clearance-feature-ratio 0.04`
- `--clearance-min-mm 0.05`
- `--lead-in-mm 0.60`
- `--sibling-clearance-mm N`
- `--clearance-mode insert-shrink|socket-overcut|split`
- `--assembly-mode tree|flat|legacy-flat`
- `--assembly-tree-strategy recursive-minimal|strongest-path`
- `--min-assembly-shared-edges N`
- `--assembly-direction-override-dot N`
- `--output-layout assembly|separate-items` (default `assembly`)

## Boundary Fairing

- `--boundary-fairing-mode constrained|taubin|off` (default `constrained`)
- `--boundary-fairing-radius-mm 1.20`
- `--boundary-max-displacement-mm 0.075`
- `--boundary-feature-angle-deg 35.0`
- `--boundary-fidelity-weight 1.0`
- `--smooth-iterations`, `--lambda-factor`, and `--mu-factor` remain available only for explicit legacy `taubin` mode.

Constrained mode uses an arc-length-scaled curvature objective, source-surface conormal motion, feature locks, broad-extent anchors, a centroid constraint, and a hard per-vertex displacement bound. Source vertex ids establish one canonical loop order so matching insert and parent boundaries fair identically even when their local winding is reversed.

Default cap and nested-cap modes are `adaptive`. For each boundary and candidate inward field, the implementation measures parent thickness and computes `safe maximum = min(5 mm, parent thickness - 0.05 mm)`. The effective minimum is 1.0 mm unless parent thickness is below 1.05 mm. It first tries a coherent plane whose point depths may differ but whose bottom is coplanar. Only when the required deepest distance exceeds the safe maximum does it use local-offset at that safe maximum. An explicit `--planar-extra-limit-mm` can narrow the global ceiling. There is no CLI fallback for center-fan caps; unsafe synthetic-hub closure is forbidden.

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

Exit code `3` means mesh or package validation failed. Default execution writes one final 3MF; other artifact types require explicit debug or preview switches.

Keep `--max-topology-defect-ratio` at or below `0.001` for publication. Do not increase it to rescue generated geometry that failed the default threshold.
