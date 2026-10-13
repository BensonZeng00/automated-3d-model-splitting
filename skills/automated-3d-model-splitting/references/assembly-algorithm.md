# Pairwise interface assembly

Direction precedence: match the frozen rim tangents against outward-wound
half edges on the confirmed tenon's source patch. With at least 80% arc coverage
and 85% signed orientation consensus, use the opposite closing-cap orientation.
This topology evidence takes priority over the source-interior/proximity fallback
described below: a wrapping host exterior can lie nearer on the wrong side of
the seam. Both Stage 04 and Stage 05 use this rule; reversing rim order or rigidly
rotating the model must not reverse the material side. A role label alone is
not evidence that the generated connector protrudes into the intended host.

Stage 04 is the sole source of confirmed tenon/mortise ownership. Both stages use `interface_direction.inward_interface_axis`. After the topology rule above, unmatched boundaries retain the legacy source-interior and paired-region fallback: probe 0.2 mm either side of the seam centroid with three first-hit rays, then use paired-region proximity when interior evidence is symmetric. Misses do not certify interior. Ambiguous fallback evidence blocks construction. Stage 05 repeats the direction check for legacy plans, records the applied construction axis separately from the saved Stage 04 direction, and never swaps roles. Both generated sides use the same axis, and the connector depth audit still checks the source far wall. These checks do not certify whole-assembly containment. There is no root-part selection, parent tree, or recursive layer scheduler.

## Per-interface construction

1. Validate confirmed roles, directions and the frozen shared XYZ rim. Do not move the rim or infer a new role from planarity.
2. Default planar interfaces use the restored full-rim-to-scaled-ellipse strip. Reuse the original arclength-parameterized ellipse fit in `interface_profile.py`; every rim sample connects to its corresponding tip sample. There is no local cylindrical peg or shoulder.
3. Curved interfaces connect the frozen simplified rim to the scaled ellipse at its mean axial position and close that root directly. No protruding local connector or deeper socket is generated. Retain radial clearance; additional bottom clearance and connector depth are zero.
4. Classification and construction consume the same frozen simplified rim. Never remap to original source edges or select a strategy by interface ID. The replay CLI no longer accepts per-interface construction overrides.
5. Planar interfaces retain one exit-ray batch and numerical depth backoff, including plane alignment in the 5 mm budget. Curved shoulders retain patch topology checks without adding depth probes. These checks do not certify whole-assembly containment.
6. Clip source exterior triangles along a distance field from the frozen shared contour; interpolate crossings within original edges and preserve source face paint. Rebuild one exterior transition to that same contour for every interface. Check the visible seam as well as closed-part/package-reload results.

Classification uses arc-length weighted plane fitting and one trimmed refit. The distance tolerance is max(0.15 mm, 1% of the smaller in-plane span). Classify as curved only when the weighted 90th-percentile plane distance is at least 2 mm and 20% of that span, and at least 90% of weighted boundary length exceeds the distance tolerance. Exterior skin normals are not contact-plane normals.

The unified straight-wall and spanning-chord cap constructions are removed. The mean-plane curved shoulder is explicitly restored by the user's image selection; zero-depth fallback and ratio-based collision acceptance remain removed. Default scale and clearance remain 0.50 and 0.2 mm.
