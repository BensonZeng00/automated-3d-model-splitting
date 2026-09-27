# Frozen simplified boundaries

Recognition extracts ordered closed color boundaries from the source mesh. It removes sparse isolated spikes using local edge-scale and chord-deviation checks. A ring is rejected only when its largest axis-aligned span is below 1 mm. A component with no accepted ring is excluded before the final region list is produced.

Each accepted ring is simplified once during Stage 03 to approximately 5% equal-physical-arc samples, with at least three points. The review image may smooth its displayed line, but that display operation does not move source vertices or change the saved contour. The saved ordered points and fingerprint are authoritative for Stage 04 contact planning and Stage 05 interface construction.

Stage 05 may adjust its generated **inner** contour when the projected inset crosses itself. The frozen Stage 04 outer contour remains unchanged. No later stage repeats boundary detection, spline fitting, source-edge remapping, or global source-surface smoothing.
