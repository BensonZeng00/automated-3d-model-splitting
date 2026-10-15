# Final 3MF contract

The output contains one colored mesh per recognized part. By default, one top-level component assembly and one build item group those meshes. `--output-layout separate-items` is an explicit alternate layout. Geometry coordinates are millimeters with the input build/component transform baked into them.

Stage 05 preserves source exterior faces and connects them locally to the frozen Stage 04 contour. Each part's generated interface consists of one direct outer-to-extended-inner side surface and a planar inner cap. A matching mortise has an outward inner-tip allowance and a deeper floor. No intermediate annulus, recursive subassembly, or Boolean result is serialized.

Preserve the source filament palette and slot order, including duplicate or unused colors. For Bambu Studio source projects, retain compatible project metadata, the application marker, and the original source filament assignments. The writer may repair face winding without moving vertices or changing triangle membership.

Before publication, each complete part must be water-tight and consistently wound without newly introduced degenerate faces. Write to a temporary sibling `.3mf`, validate the package structure and colors, reload every part, and compare part identities and geometry counts. Publish only the passing file. Stage artifacts and diagnostics are separate from the final deliverable.
