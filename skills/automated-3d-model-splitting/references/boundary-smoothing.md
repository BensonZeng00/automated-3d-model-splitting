# Explicit boundary smoothing

This is the sole production boundary policy. `smooth` is the default; the source mode has been removed. First apply the area-based local cleanup and completion-first judgment in [completion-first.md](completion-first.md), then fit the shared seam.

Recognized component boundary loops are cleaned once during recognition. Sparse
isolated spike vertices are removed using a local edge-scale and chord-deviation
check. Only the largest perimeter loop is considered for each recognized
component; all secondary loops are removed. The selected loop is also removed
when its largest axis-aligned span is below 1 mm. If this leaves a component
without an accepted ring, recognition excludes that component before it creates
the final region list; plotting receives only that final list. Both simplified
and unsimplified topology snapshots omit rejected loops, preventing later seam
planning from restoring them. A surviving ring is simplified by approximately
5% sampling at equal physical arc-length intervals, with at least three source
vertices. There is no geometric deviation tolerance. The review image applies
a light cyclic smooth to the retained samples for readability; source mesh
vertices are not moved. Later interface construction consumes the frozen
source-vertex rings and does not simplify them again.

For every production split, use `planar-arc-retopology` for each cut boundary. Canonicalize the closed loop by source vertex id, fit a stable local plane with SVD, project to 2-D, and resample at 384 equal physical arc-length positions. Fit those projected samples with a 24-control least-squares periodic cubic B-spline and fit the stable-plane-normal height independently with an 8-control periodic cubic B-spline.

A three-edge closed loop is already the smallest valid polygon and cannot support a cubic spline: preserve that exact micro-loop instead of inventing a fourth control point, and record the exception. Never restore source projected extrema after fitting; record the signed extent deltas instead. Locally regularize source physical edge fractions for up to 8 cyclic passes on loops up to 300 vertices and 2 passes on larger loops, eliminating isolated microscopic rim edges without rotating semantic source ids several millimetres around a deliberately nonuniform loop. Evaluate the fitted splines directly at those regularized source phases and keep child/parent source-id correspondence exact.

When the target stays within the safe visible envelope (45% of the band in strict mode or 60% after review in advisory mode), make it the shared visible seam and diffuse its displacement through a topology-connected surface band with a C2 fade to zero; vertices outside the band remain unchanged. When it exceeds that envelope but remains within `--maximum-boundary-displacement-mm`, keep the source rim immutable and retain the fitted ring for generated inward walls and caps only. This prevents a large visible-surface reconstruction merely to accommodate a distant manufacturing target. The default transition band is 3 mm (not a vertex displacement allowance), while the maximum and P95 fitted-target offset default to 10 mm. Audit modified source faces for degeneracy, seam equality, bounded edge stretch, and directed-edge topology. Only newly introduced source-normal reversals consume the selected profile budget; pre-existing source defects remain diagnostic.

When multiple child rims share source IDs, first reconcile their proposals through [shared-layer-seams.md](shared-layer-seams.md). Its joint target and locked junctions are authoritative for every participant, and its single complete-layer deformation replaces independent per-child surface passes. Ordinary offset and surface-quality gates still apply; report the actual joint displacement separately from each loop's original proposal.

After the user has visually approved that band, `--surface-band-validation advisory` may report source-relative normal changes, lower the sparse-outlier result-angle floor from 3 degrees to 0.01 degree, accept isolated shared-loop junction vertices by locking them to their original source positions, raise the bounded edge-stretch ceiling from 8x to 128x, and reuse a source-ID-matched prevalidated hidden fit ring when a later equivalent surface-band pass changes only its local conormals within the reviewed band offset. Exact seam match, zero degenerates, directed-edge topology, Boolean, and assembly gates remain blocking, and every advisory must be recorded.

If the target exceeds the configured displacement limit or the remaining hard checks fail, block the interface—never weaken the target, partially back off, or invoke another smoothing algorithm. Build the constant-distance inner ring from the retained fitted target and keep the requested slope near 45 degrees. Record whether the visible seam moved or the source rim was preserved, including the explicit three-edge preservation exception when used.

## Printable smoothing profiles

Production defaults to `print-balanced`, which evaluates the displacement distribution and only counts defects introduced by smoothing. The CLI accepts P95 and maximum boundary motion through 10 mm by default. Coverage budgets are distribution-aware: each affected source triangle may account for at most 2% of source area while all affected triangles together may account for at most 15%; collateral vertices are similarly capped at 15% (and 5000 vertices), with at most 8 topology layers. Introduced source-normal reversals may occupy at most 15% of the audited band in total and no edge-connected cluster may exceed 2%, so many separate small triangles are not mistaken for one large defect. Every result minimum angle must remain at least 0.01 degree, and isolated seam-adjacent faces are permitted. Edge stretch through 16x is allowed. Open, over-shared, inconsistent or degenerate topology remains blocking. `source-conservative` rejects introduced reversals and tightens the stretch limit; `print-smooth` expands the stretch limit for an explicitly print-first result. None of the profiles authorizes changes to source ownership or unrelated source geometry.
