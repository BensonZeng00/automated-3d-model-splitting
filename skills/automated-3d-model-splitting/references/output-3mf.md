# Final 3MF Contract

By default, write exactly one final file. Do not write STL, GLB, JSON, Markdown, recursive previews, or diagnostics without the corresponding explicit switch.

## Recognition and Classification Provenance

Record:

- vendor composite-paint decoding profile, raw source-face count, decoded leaf-face count, conformed face count, T-joint repairs, and decoded leaf-state counts;
- recognition profile and visibility settings;
- visible and occluded face counts;
- reassigned occluded faces by original token;
- selected processing mode (`body` or `inward`);
- body-selection status, confidence, and structural evidence;
- explicit body or inward compatibility overrides.

Recognition never edits the source mesh.

## Geometry

Bake the source package's unique build/component instance transform into the vertices before generating geometry. Output coordinates and inward-depth settings are millimeters.

Every non-body part uses the same recursive inward extrusion, cap, socket, clearance, and lead-in behavior. Through-like structural measurements are body-selection evidence only and must not create macro partitions, direct cuts, or root separator objects.

Execute recursion as a strict parent-to-child 3MF transition. A parent step must serialize each direct child as an independent colored 3MF, including per-triangle color properties and source filament slots. When the child is later visited in depth-first preorder, reload that exact parent-emitted file as the recursive input; do not extract the child from the most recent cumulative package. Replace exactly one pending subassembly and retain every untouched mesh. When debug output is requested, also serialize and reload the complete cumulative state after each replacement as an audit only. Use the final recursive state as the final package mesh source instead of rebuilding parts in a separate batch.

Default to a 1.0 mm effective minimum and a measured safe maximum of `min(5 mm, parent thickness - 0.05 mm)`. Permit different boundary-point depths when all bottom points lie on one coherent plane. Use local-offset only when that plane's deepest required distance exceeds the safe maximum. Record measured parent thickness, reserve, preferred/effective minimum, safe maximum, required planar maximum, selected cap mode, and minimum/maximum generated inward travel for every part.

For recursive parents, keep the parent's own outer cap in local-offset mode while allowing matching child sockets to follow each child's bounded cap policy. Prevalidate every direct child, including root-level children and nested leaves, before constructing the parent socket. Carry one source-vertex-keyed `CapDecision` into both builds so the parent socket uses the child's final selected mode, direction field, and actual distance field. Reserve bottom clearance inside the total-travel budget, and record any topology-driven retry with before/after defect-edge counts. Replanning an existing child cap from the parent path is a package-blocking consistency error.

Within the 3.0 mm global ceiling, use a 0.40 mm default extra allowance for nested leaf details (2.4 mm total at the default target) to control oblique parent-wall exposure. Record this effective per-part allowance separately from the global ceiling.

Cap nested leaf effective fit clearance at 0.10 mm. Record both the pre-cap feature-adaptive clearance and the effective clearance used by the insert and its matching socket.

Default cut-loop position smoothing uses constrained arc-length fairing with a 1.20 mm physical scale and a 0.075 mm hard displacement limit. Record the selected fairing mode and, for every loop, its feature locks, maximum and RMS displacement, length change, and solver status. Matching parent and insert loops must use the same source-normal field and source-id canonical order.

For curved or multi-loop boundaries, use per-boundary-vertex safe local inward directions. When directions differ within one ring, use a fixed-depth `local-offset` cap instead of claiming a shared flat plane. Record corrected and remaining outward-directed vertex counts; remaining must be zero. Parent sockets must reuse the child's direction map keyed by source vertex id.

## Package

The ZIP must contain `[Content_Types].xml`, `_rels/.rels`, and `3D/3dmodel.model`. Use millimeters and a standard 3MF color group. Represent every printable part as one mesh object. Default to one additional component-assembly object and exactly one build item referencing that assembly. `separate-items` is the explicit compatibility layout where every mesh object is a build item.

When the input is a Bambu Studio project, retain its `BambuStudio-*` application marker and record `automated-3d-model-splitting` separately as the generator. Retain the non-internal source project settings. In assembly layout, write one Bambu object containing one part record per mesh component, one plate instance, and one assemble item; preserve every part's one-based source extruder. Write `Metadata/project_settings.config`, `Metadata/model_settings.config`, and `Metadata/slice_info.config`. Never write the Bambu marker alone: current Bambu Studio treats a nonzero Bambu version plus an empty project configuration as an obsolete project and loads geometry only.

Each object annotation must include part id, raw token, resolved color, filament slot, mapping source, resolution status, recognition basis, occluded-paint exclusion, selected processing mode and body evidence, assembly parent/depth, inward cap and depth settings, fit settings, edge diagnostics, and validation level.

Synthetic geometry inherits the owning part's resolved color.

Preserve every source filament color in original zero-based slot order, including unused and duplicate-colored slots. A source-mapped object's `pindex` must equal its resolved `filament_slot_index`, and the color stored at that index must equal the resolved source color. Do not derive palette order from object order. Append only colors produced by explicit overrides or fallback resolution.

## Atomic Validation

Default ratio validation accepts `(open_edges + over_shared_edges) / unique_edges <= 0.001` and labels accepted nonzero defects `ratio_accepted`. Strict mode requires zero defects.

Normalize face winding before serialization without moving vertices or changing triangle membership. Require every generated and reloaded object to have consistent winding and zero inconsistent shared edges. Winding failures are always blocking; the ratio profile applies only to open and over-shared topology edges.

Validate meshes in memory, write a temporary sibling 3MF, reload it, and verify package entries, object/build counts, names, source-slot color indices, colors, vertex/triangle counts, annotations, winding, and topology. For Bambu-source projects, also parse all three generated Bambu metadata files, require a nonempty project config, verify its filament palette, and verify every model-settings extruder against the object's resolved source slot. Verify that exactly one selected body exists and every other object records `selected_processing_mode=inward`.

Before writing the temporary package, compare source and assembled generated face centroids from deterministic common views. Use a one-pixel source envelope to avoid centroid-raster false positives. Require sufficient projected coverage, generated cap/socket intrusion below the configured global ratio, frontmost material mismatch below its global tolerance, and every source-part/generated-part mismatch pair below the configured local coverage ratio. Record per-view, per-part, and mismatch-pair evidence in the run summary. This is an offline validation stage; never invoke Computer Use. Ask the user for slicer screenshots when human confirmation is required.

Only atomically replace the requested final file after validation succeeds. On failure, remove the temporary file, preserve any existing final file, and exit with code `3`. `--diagnostic-preview` is the only exception and must be labeled non-validated.
