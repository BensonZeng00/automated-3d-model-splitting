# Format Support

## Supported

- Read only the user-selected `.3mf` archive.
- Select an explicit internal `.model` entry or default to the entry with the most triangles.
- Read triangle meshes with vendor `paint_color` assignments.
- Resolve one unique build instance through nested `<components>` and affine build/component transforms, then bake it into millimeter-space vertices.
- Convert `micron`, `millimeter`, `centimeter`, `inch`, `foot`, and `meter` coordinates to millimeters.
- Read filament colors from `Metadata/project_settings.config` when available.
- Resolve triangles without `paint_color` from source object/component extruder metadata.
- Override display names and colors with `--color-map-json`.
- Use externally visible source paint for recognition while retaining the original source mesh.
- Export one standard colored multi-object 3MF using the 0.4.2 split geometry.

## Rejected Deliberately

- Standard 3MF material/color properties referenced only through `pid`, `pindex`, `p1`, `p2`, or `p3` without `paint_color`.
- Multiple independent project instances whose transforms cannot be reduced to one unambiguous source model.
- Build-item/object-resource mismatches.
- Unknown units.

Reject unsupported input with a specific error. Never reinterpret it as a default-colored mesh.

## Recognition Profile

Recognition always filters occluded paint using deterministic multi-view depth maps before connected-region grouping. There is no surface-profile selector.

The filter does not claim to reconstruct a hidden volumetric material field. It answers only which source paint should participate in exterior part-boundary recognition.

## Extension Rule

Add a separate parser adapter before expanding input support. Preserve these normalized outputs:

- vertices in millimeters;
- triangle indices;
- one source color token and one effective recognition color per triangle;
- resolved default-filament provenance without assuming slot 1;
- selected model entry, detected profile, and source unit;
- exterior visibility and reassignment diagnostics.

Input and output profiles differ: vendor `paint_color` is accepted as input, while output uses standard 3MF color-group properties for slicer interoperability.
