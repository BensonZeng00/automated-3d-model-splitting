# Pairwise interface assembly

Stage 04 is the sole source of interface ownership and direction. Stage 05 consumes its pairwise relations and the immutable simplified-boundary snapshot. It has no root-part selection, parent tree, or recursive layer scheduler.

## Per-interface construction

For each Stage 04 relation:

1. Validate the tenon and mortise part IDs, the ordered shared-boundary IDs and points, both side directions, and the mating axis. The ordered simplified boundary in Stage 04 is authoritative; Stage 05 consumes it directly and does not match it back to source boundary rings or expand it by arc length. Check self-intersections in world XYZ before projection. At a true 3D crossing, retain the longer perimeter loop and remove the smaller loop. A crossing visible only after projection is diagnostic and must not delete its XYZ-separated arcs.
2. Keep the frozen XYZ boundary as the canonical outer ring. Its area normal is the construction axis; use the Stage 04 tenon/mortise relation to orient that normal toward the mortise. Scale the projected outline around its area centroid by `interface_scale_ratio` (default `0.50`) and make the inner ring planar. If that projected inset crosses itself, replace only the inner outline with a simple arclength-parameterized contour while retaining the frozen outer ring and point correspondence.
3. Probe the inner ring and cap samples along the construction axis. Start at a 5 mm maximum travel. If a probe hits the mortise shell, halve the travel and probe again; never try below 0.2 mm. An unhit ray permits the full attempted depth. A collision at the minimum depth stops the interface.
4. Extend the inner ring to the accepted depth. Connect corresponding vertices of the frozen outer ring and extended inner ring directly with one tapered triangle strip, then cap the extended inner ring. There is no intermediate annulus or separate inner wall. Build the mortise with the same topology, an outward inner-tip offset of the configured clearance, and a floor deeper by that clearance. The shared outer ring stays fixed; the allowance therefore grows from zero at the mouth to the configured value at the inner tip.
5. Do not run a second boundary-identification or boundary-simplification pass in Stage 05.
6. Record boundary, projection, winding, closure, collision, and clearance findings as per-interface diagnostics. The complete-part export still requires watertight, consistently wound meshes and a valid 3MF readback.

The default inner-ring scale is `0.50`; the default additional mortise side and floor clearance is `0.20 mm`. Both values are CLI-configurable. Record the tenon depth, deeper mortise recess, every depth-halving attempt, measured side and floor allowance, IDs, directions, and topology audit in the Stage 05 summary.

## Stage transaction and publication

Stage 05 builds candidates in memory. It writes interface-surface NPZ data, complete colored-part meshes, and a JSON summary only after construction succeeds. A full run then performs closed-mesh checks and serializes a temporary 3MF. Stage 08 reloads and validates that package; only a passing package replaces the final output path. Failure removes the temporary 3MF and leaves no final package marked successful.

Source mesh defects away from a planned interface are outside this stage's repair scope. They are not silently patched to make interface generation pass.
