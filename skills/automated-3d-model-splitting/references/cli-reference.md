# CLI Reference

Run the production workflow with `scripts/split_painted_3mf.py`. The CLI owns argument parsing, input checks, stage selection, and exit codes; geometry is built by reusable services.

## Main workflow

```powershell
python scripts/split_painted_3mf.py --input model.3mf --preflight-only
python scripts/split_painted_3mf.py --input model.3mf --recognize-only
python scripts/split_painted_3mf.py --input model.3mf --output model_parts.3mf `
  --interface-scale-ratio 0.50 --interface-clearance-mm 0.20
```

The production path is **01 preflight → 02 load → 03 recognize → 04 pairwise interface planning → 05 interface construction and assembly fit → 08 package validation and publish**. The old recursive 05/06/07 path and its public switches have been removed.

## Useful options

- `--input PATH`: required source 3MF.
- `--output PATH`: final colored 3MF; defaults beside the source.
- `--format-profile auto|vendor-paint`: identify or require supported painted 3MF input.
- `--overwrite`: replace an existing final output.
- `--preflight-only`, `--recognize-only`: stop after preflight or recognition.
- `--stage-artifacts-dir PATH`: choose the root for inspectable JSON/NPZ stage artifacts.
- `--stop-after-stage preflight|load|recognize|assembly|interface-assembly`: stop after the named stage is recorded.
- `--interface-scale-ratio R`: homothetic inner-loop ratio for Stage 04 interfaces; must be greater than 0 and less than 1.
- `--interface-clearance-mm MM`: additional side and floor clearance added to each mortise relative to its tenon; must be non-negative.
- `--max-extension-mm MM`, `--max-planar-travel-mm MM`, `--cap-mode MODE`: control the safe inward extension and its cap.
- `--boundary-review-json PATH`, `--recognition-review-json PATH`, `--region-review-json PATH`: apply explicit reviewed decisions.
- `--recovery-dir PATH`: retain geometry failure evidence for replay.

Use `--help` for the complete current option list. Removed recursive-tree, whole-part uniform-scale, post-fit seating, and multi-view visual-validation switches are not accepted; their former stages no longer participate in this workflow.

## Interface contract

Stage 04 is the authority for each pair, including tenon/mortise ownership and directions. Stage 05 projects its frozen simplified shared boundary onto the interface plane, scales an inner ring about the area centroid (default ratio `0.50`), and probes toward the mortise up to `10 mm`. On any shell hit it halves the candidate tenon depth, with `0.2 mm` as the minimum; a hit at that minimum fails the interface. It then derives the mortise as a negative cavity from the tenon profile, adding the configured clearance to the side profile and floor depth, and checks that expanded cavity against the mortise shell. The inner end planes are capped and the annular bands retain matching topology. The stage records every depth attempt and measured side/floor allowance. It does not infer a root part or recurse through a parent tree. Stage 08 writes the final 3MF only after mesh and package validation pass.
