# Pairwise interface assembly

Stage 04 is the sole source of interface ownership and direction. Stage 05 consumes its pairwise relations and the immutable simplified-boundary snapshot. It has no root-part selection, parent tree, or recursive layer scheduler.

## Per-interface construction

For each Stage 04 relation:

1. Validate the tenon and mortise part IDs, the ordered shared-boundary IDs and points, both side directions, and the mating axis. The ordered simplified boundary in Stage 04 is authoritative; Stage 05 consumes it directly and does not match it back to source boundary rings or expand it by arc length. Check self-intersections in world XYZ before projection. At a true 3D crossing, retain the longer perimeter loop and remove the smaller loop. A crossing visible only after projection is diagnostic and must not delete its XYZ-separated arcs.
2. Keep the frozen boundary in XYZ as the canonical outer ring. Scale around its interface-plane area centroid by `interface_scale_ratio` (default `0.50`), preserving each point's axial height. Build the annular band from corresponding outer and inner samples as a triangle strip. Do not flatten the outer or inner ring to create the band.
3. Probe the inner ring and its cap samples toward the mortise along Stage 04's socket direction. Start at 10 mm maximum travel. If any probe intersects or passes through the mortise shell, halve the travel and probe again; never try below 0.2 mm. If 0.2 mm still intersects the shell, keep the minimum-depth geometry and record the collision.
4. Build the tenon first and record its exact profile, placement, and safe depth. Build the matching mortise as a negative cavity from that tenon: expand its inner profile by the full configured clearance and deepen its floor by the same amount. Probe the expanded cavity against the mortise shell during depth selection so it cannot break through. Keep the outer boundary fixed and close each inner boundary with a curved cap whose triangles use the XYZ ring vertices. Prefer a simple auxiliary projection for cap connectivity; if every tested projection crosses, use a projection-independent 3D centre fan. The auxiliary projection never changes or deletes boundary geometry.
5. Do not run a second boundary-identification or boundary-simplification pass in Stage 05.
6. Record boundary, projection, winding, closure, collision, and clearance findings as per-interface diagnostics. These geometry-quality findings do not block Stage 05 output; malformed or missing Stage 04 inputs that make construction impossible remain execution errors.

The default inner-ring scale is `0.50`; the default additional mortise side and floor clearance is `0.20 mm`. Both values are CLI-configurable. Record the tenon depth, deeper mortise recess, every depth-halving attempt, measured side and floor allowance, IDs, directions, and topology audit in the Stage 05 summary.

## Stage transaction and publication

Stage 05 builds candidates in memory. It writes interface-surface NPZ data, complete colored-part meshes, and a JSON summary only after construction succeeds. A full run then performs closed-mesh checks and serializes a temporary 3MF. Stage 08 reloads and validates that package; only a passing package replaces the final output path. Failure removes the temporary 3MF and leaves no final package marked successful.

Source mesh defects away from a planned interface are outside this stage's repair scope. They are not silently patched to make interface generation pass.
