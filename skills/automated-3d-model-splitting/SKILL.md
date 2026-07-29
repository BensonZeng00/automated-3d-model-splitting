---
name: automated-3d-model-splitting
description: Automated painted-3MF model-part splitting for vendor projects with externally visible per-triangle paint. Use when Codex must recognize printable color regions, infer a root body and nested insert tree, generate inward parts and sockets, preserve filament slots, export one grouped assembly 3MF, and validate topology plus multi-view visual surface consistency.
---

# 自动化3d模型拆件

Current release: `1.3.5`, positioned as automated painted-3MF model-part splitting.

## Protect the Source

Read only the user-specified source 3MF. Do not inspect other model files unless the user explicitly authorizes them.

Use `scripts/split_painted_3mf.py`. Do not use Blender. By default, generate no STL, GLB, JSON, Markdown, preview, or diagnostic files. Deliver one final grouped multi-part `.3mf`.

Do not use Computer Use to open or inspect Bambu Studio or another slicer. Complete deterministic offline validation first. When slicer confirmation is needed, ask the user to open the final 3MF and provide screenshots showing the model, the expanded assembly tree, and any visually sensitive region.

The script is a stable compatibility entry point. Implementation lives in the object-oriented `scripts/split3mf/` package and is coordinated by `SplitPipeline`. Read [references/architecture.md](references/architecture.md) before changing module responsibilities, service boundaries, shared state, or pipeline orchestration. Keep the entry point thin and preserve its CLI contract.

## Separate Recognition, Body Selection, and Cutting

1. Decode each vendor `paint_color` value first. A multi-nibble value is a reversed hexadecimal `TriangleSelector` subdivision stream, not one color token. Restore its leaf subtriangles, leaf states, shared midpoints, and T-joint conformity before connected-boundary recognition.
2. Recognize colors and connected boundaries from externally visible restored faces only.
3. Reassign occluded or deeply recessed paint faces to the recognition base color without editing the source mesh.
4. Measure through-like structure only as body-selection evidence: opposite-facing surfaces, shell-removal regions, balance, and large shared interfaces.
5. Penalize or exclude a strong structural separator from automatic body selection.
6. After the root body is chosen, classify the body as `body` and every other component as `inward`.
7. Run the ordinary recursive-minimal inward workflow for the complete assembly. Do not create macro bodies, through roots, or direct through cuts.

Use normals, opposite-facing surface balance, separation along the dominant normal, component span, and boundary-loop span for automatic classification. Do not classify from color name or area alone. Treat the root body as `body`.

There is no printable `through` processing mode. `--part-processing-mode` accepts `auto` or compatibility value `inward`; per-part overrides accept only `P10=inward`.

Default exterior recognition uses 32 deterministic views, a 768-pixel depth map per view, and a `0.08mm` depth tolerance. Report visible, occluded, and reassigned faces by original paint token. Use `--recognition-surface-profile all-faces` only as an explicit compatibility override.

Read [references/assembly-algorithm.md](references/assembly-algorithm.md) before changing recognition, body evidence, tree inference, inward-cap behavior, or inward-direction logic.

## Check Dependencies

Run `--preflight-only` on installation or first use. The script checks `numpy`, `scipy`, `trimesh`, and `networkx`. If a dependency is missing, explain the interpreter-scoped install command and request permission before running it. Never install silently.

Relay the script's Chinese preflight, color, recognition, classification, assembly, split, validation, and export annotations during long work. Keep machine-readable JSON lines in logs.

## Follow the Three-Stage Workflow

1. Run recognition and classification first:

   ```bash
   python scripts/split_painted_3mf.py --input "/path/to/model.3mf" --recognize-only
   ```

2. Report the input profile, source unit, model entry, exterior-recognition statistics, excluded paint tokens, effective parts, body, colors, face counts, bounding boxes, centers, boundary loops, selected processing mode, confidence, and evidence.

3. Stop only when the body, topology, or semantic parent is ambiguous. Processing mode is always inward for non-body parts. Do not invent visual semantics.

4. Export after the inventory is accepted or unambiguous:

   ```bash
   python scripts/split_painted_3mf.py \
     --input "/path/to/model.3mf" \
     --output "/path/to/model_split_parts.3mf"
   ```

5. Treat exit code `3` as a failed mesh/package deliverable. Write a temporary 3MF, reload its objects and colors, and atomically replace the final path only after validation passes.

## Apply Hybrid Geometry Rules

- Group recognition colors by shared-edge connectivity after exterior filtering.
- Resolve package build/component transforms before recognition and bake the unique project instance into millimeter-space vertices. Do not read a mesh submodel at raw scale while ignoring its parent build transform.
- Treat raw paint tokens that source metadata maps to the same filament slot as one material for edge connectivity. Preserve all contributing raw tokens in provenance; do not create a separate printable part merely because `DEFAULT` and a vendor token serialize the same material differently.
- Merge components below `--min-faces` before body, classification, and tree inference.
- Before automatic body selection, score every eligible candidate using relative size and strong-separator strength. Use through likelihood only inside separator evidence; do not subtract it unconditionally from candidates whose removal does not produce a balanced shell split. Exclude a strong structural separator when it passes the through geometry gate, has at least two large structural interfaces, and its removal splits the complete shell into exactly two meaningful regions whose face-count ratio is at least 0.50. Choose the highest-scoring remaining component regardless of color. `DEFAULT` and base-color labels must never filter or boost automatic body candidates. Respect explicit body overrides.
- Never inward-extrude the body.
- For `inward`, extend inward, close the bottom, cut child sockets, and default to `adaptive`, a 1.0 mm effective minimum depth, a measured 5.0 mm maximum safety ceiling, 0.30 mm requested fit clearance, and a 0.60 mm lead-in.
- Default to feature-adaptive clearance: clamp the requested fit clearance by the smallest local part extent while leaving the visible top surface unchanged. Retain fixed clearance only as an explicit compatibility profile.
- Do not reuse one component-average inward vector blindly across curved or multi-loop boundaries. At every boundary vertex, compare the candidate direction with the area-weighted local inward normal, blend unsafe directions into the local inward hemisphere, and require zero outward-directed generated vertices. Reuse the same per-source-vertex directions for the matching parent socket.
- Accept high-confidence `force_inward_vector` and `force_parent_direction` visual-semantic overrides as audited escape hatches. Normalize and validate every override, reject zero or non-finite vectors, never apply an override to the root body, and still require geometry and multi-view validation to pass. Never turn one model's axis into a global default.
- Fair cut-boundary positions with constrained arc-length bi-Laplacian optimization. Use source-mesh surface normals, lock detected corners, preserve broad extents on smooth loops, cap every displacement at 0.075 mm, and canonicalize by source vertex id so inserts and parent sockets receive identical coordinates. Keep legacy Taubin only as an explicit compatibility mode.
- Measure parent thickness from every boundary point along each candidate inward field before generating a cap. Set `safe maximum = min(5 mm, parent thickness - 0.05 mm)`. Use a 1.0 mm effective minimum unless measured parent thickness is below 1.05 mm; only then reduce the minimum to the remaining safe maximum.
- Through-like evidence never changes geometry mode or creates a printable separator; retain it only in body-selection reports.
- Keep one strict tree-recursive body-splitting flow after body selection. When a parent step creates a direct child, serialize that child immediately as an independent millimeter 3MF with its per-triangle colors and original filament-slot meanings. Process internal nodes in deterministic depth-first preorder, and require every later child step to reload the exact 3MF emitted by its parent step; a sibling branch or the latest cumulative package is never a substitute. Replace exactly one pending subassembly in the active assembly state and carry every untouched mesh forward. Reject missing, already expanded, skipped-parent, color-changed, or descendant-provenance-mismatched inputs. The final assembly must reuse the meshes materialized by this recursion rather than rebuilding all parts in a separate batch.
- After reloading a pending child, rebuild that step's component faces, boundary-neighbor graph, vertex normals, component centers, and model center from the reloaded mesh. Treat its already generated parent-contact shell as immutable source geometry for the local body cut. Never reuse the original whole-model arrays or regenerate the parent-contact extrusion at a later recursive step.
- Adaptive caps must try one coherent bottom plane first, comparing the component-global inward plane with an inward-facing loop best-fit plane. Boundary points may travel different distances, but their bottom points must be coplanar. Accept the plane whenever its deepest required distance is between the effective minimum and measured safe maximum.
- Use smooth `local-offset` only when the deepest required planar distance exceeds the measured safe maximum. In that fallback, move every boundary point along its safe local inward direction to the safe maximum, normally 5.0 mm. Do not force local-offset merely because a part owns child inserts, is nested, or previously produced a small ratio-accepted topology finding.
- Prevalidate nested caps before building parent sockets. The parent socket must reuse the child's final cap mode and actual cap distance field, keep any requested bottom clearance inside the same measured safety ceiling, and place its bottom behind the child cap rather than independently fitting another plane.
- Cap nested leaf effective fit clearance at 0.10 mm after feature-adaptive clearance. This keeps a printable assembly gap while reducing parent-wall exposure beside deep planar details; record the pre-cap and effective values.
- Local-offset fallback directions must be smoothed over physical boundary arc length, and single-loop caps must use distributed short-diagonal triangulation without a synthetic center fan.
- Multi-loop caps must bridge holes and use boundary-preserving triangulation. Do not use unconstrained Delaunay plus centroid filtering for concave or holed boundaries.
- Default to `tree` plus `recursive-minimal`; flatten local bodies and leaf parts into separate mesh objects inside one top-level 3MF component assembly.

Read [references/cli-reference.md](references/cli-reference.md) for options.

## Preserve Colors

Treat `paint_color` values as serialized vendor paint data, not decimal palette indices. Decode simple leaf tokens and composite recursive subdivision trees before color connectivity. Preserve child order, midpoint sharing, and cross-face T-joints; rejecting or atomizing a valid composite token is an input-decoding failure. Resolve decoded leaf states against project filament metadata. Resolve `DEFAULT` from source object or component-parent extruder metadata; never assume slot 1.

Distinguish `source_metadata`, `explicit_override`, and `fallback_estimate`. A fallback requires user acceptance. Synthetic sockets, sides, and inward caps inherit the owning part's resolved color.

Preserve the complete source filament palette in its original slot order. Set each source-mapped object's 3MF color index from its resolved zero-based `filament_slot_index`; never rebuild the palette from first object appearance or collapse duplicate-colored slots. Append only explicit overrides or fallback colors that do not belong to a source slot. Treat a source slot/color mismatch as an export failure.

For a Bambu Studio source project, preserve the non-internal source project settings and generate matching `Metadata/project_settings.config`, `Metadata/model_settings.config`, and `Metadata/slice_info.config`. Write each object's one-based Bambu extruder from its resolved source slot. Never emit a `BambuStudio-*` application marker without a nonempty project configuration; Bambu Studio interprets that combination as an obsolete project, loads geometry only, and can discard the intended filament assignment.

For every final object, report object/part id, color name, actual hex value, raw token, filament slot, mapping source, and resolution status. Do not collapse repeated colors.

## Validate and Deliver

Default to ratio validation with `--max-topology-defect-ratio 0.001` (0.10%). Label accepted nonzero defects and the final result `ratio_accepted`; keep `--validation-profile strict` for zero-defect audits.

Do not raise `--max-topology-defect-ratio` above `0.001` merely to publish a failed generated part. A separate localized-short-open-edge rule may accept up to 0.20% only when every defect is an open edge, winding is consistent, there are no over-shared or inconsistent edges, the longest open edge is at most `max(0.05 mm, 0.6% of bbox diagonal)`, and their total length is at most 7.5% of the bbox diagonal. Report this as `localized_short_open_edges`, never as strict watertightness.

Require one mesh object per printable part and, by default, one top-level assembly build item containing every mesh object. Validate component order, names, mesh counts, color references, grouped Bambu part metadata, and required package entries. Keep `separate-items` only as an explicit compatibility layout. Record recognition provenance, processing mode and evidence, parent/depth, inward-cap method, requested/effective fit settings, and validation result in each object annotation.

Run deterministic multi-view surface validation before package publication. Compare the source painted surface with the assembled split output under common depth maps. Cull back-facing and grazing centroid samples before building the one-pixel source envelope so hidden caps and oblique raster artifacts do not become false intrusion or material-cover failures. Keep raw ratios in the report: generated intrusion above 2% is advisory and above 4% is blocking by default. The local pair check remains a guard for small high-contrast details such as tongues, eyes, and logos. This validation must remain offline and must not invoke a slicer UI.

For Bambu-source outputs, reload all three Bambu metadata entries and require the project filament palette and every object extruder to match the resolved source slots before publication.

Before export, make adjacent face winding consistent without moving vertices or changing triangle membership, including on the large-part fast path. Require zero inconsistent shared edges and outward orientation for every watertight object. Winding defects are always blocking and cannot be ratio-accepted. Reload the written package and repeat the winding and source-slot color-index checks before atomic publication.

Record measured parent thickness, the 0.05 mm reserve, preferred/effective minimum, safe maximum, required planar maximum, selected cap mode, and generated minimum/maximum inward travel for every part. Read [references/output-3mf.md](references/output-3mf.md) for the package contract.

Use `--debug-recursive-3mf` only when requested for layer debugging; retain `--debug-recursive-stl` as a deprecated command-line alias only. Folders must follow deterministic depth-first recursion order. Export every changed local body, leaf, and pending subassembly as an independent colored `_mm.3mf`. A pending child's file must preserve its internal per-triangle colors and source filament slots, remain in the parent layer where it was born, reload successfully, and become that child's later recursive input. Also export one cumulative `_CUMULATIVE_mm.3mf` after every step for complete-assembly inspection, but never use a cumulative file as a recursive part input. Preserve the last valid standalone input, cumulative audit, and failing candidate when a step fails. Validate every standalone and cumulative 3MF with the selected strict or ratio profile. Use `--diagnostic-preview` only for an explicitly requested non-deliverable failure preview.

In the final response, state the final path, object count, body id, exterior-recognition statistics, excluded paint tokens, every part's processing mode, preserved colors, output layout, assembly mode/strategy, inward-cap and requested/effective clearance settings, topology result, and multi-view visual result. If slicer screenshots are needed, explicitly ask the user to provide them; do not control the slicer.
