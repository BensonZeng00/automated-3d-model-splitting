# Frozen simplified boundaries

Before geometry processing, apply the complexity gates in [execution-policy.md](execution-policy.md). Branching and loop/point counts alone do not reject a model; actual loop extraction and bounded-contact validation determine whether construction can proceed.

Stage 03 uses equal-arclength source samples, retains more points for small or shape-sensitive rings, and applies the prescribed 4 mm periodic Gaussian smoothing only when its shape guard passes. The smoothed coordinates are actual frozen geometry, not merely a display effect. This corrects the earlier description that smoothing affected only the preview.

The deterministic point-contact cleanup has a cumulative discarded-perimeter budget and contour-displacement budget of 2 mm. It must preserve source-point identity, including for multi-vertex contact loops. Residual contacts are audited under the same bounded policy as Stage 05; unsafe crossings reject construction. A general crossing that requires invented source vertices is never cleaned.

Stage 04/05 use the frozen ordered points and fingerprint. Stage 05 records tolerable micro-contact diagnostics without remapping, smoothing or replacing the frozen outer rim. Existing finite recovery candidates may run; exhausted candidates or failed final validation end the run. Review images remain evidence for the existing semantic gates.
