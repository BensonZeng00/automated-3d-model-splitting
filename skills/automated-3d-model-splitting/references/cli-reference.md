# CLI reference

Run `python scripts/split_painted_3mf.py --help` for the active options. The source must be a vendor-painted `.3mf`.

```powershell
python scripts/split_painted_3mf.py --input model.3mf --preflight-only
python scripts/split_painted_3mf.py --input model.3mf --recognize-only
python scripts/split_painted_3mf.py --input model.3mf --output model_parts.3mf --overwrite
```

| Option | Purpose |
|---|---|
| `--stage-artifacts-dir DIR` | Save inspectable JSON/NPZ artifacts under a new run ID. |
| `--stop-after-stage STAGE` | Stop after `preflight`, `load`, `recognize`, `assembly`, or `interface-assembly`. |
| `--region-review-json FILE` | Apply confirmed small-region classifications. |
| `--recognition-review-json FILE` | Apply fingerprint-bound recognition actions and confirmation. |
| `--boundary-review-json FILE` | Apply confirmed boundary-ownership decisions. |
| `--visual-semantics-json FILE` | Supply optional part labels and physical partitions. |
| `--interface-scale-ratio NUMBER` | Projected inner-contour scale; default `0.50`. |
| `--interface-clearance-mm NUMBER` | Extra mortise tip radius and floor depth; default `0.20`. |
| `--output-layout assembly|separate-items` | Choose the grouped assembly (default) or separate build items. |

Visibility and review sampling remain configurable with `--exterior-view-count`, `--exterior-depth-map-resolution`, `--exterior-depth-tolerance-mm`, `--noise-review-max-faces`, `--small-region-review-max-faces`, and `--region-review-resolution`. `--model-entry`, `--format-profile`, `--color-map-json`, `--region-review-dir`, and `--recovery-dir` control their corresponding input or artifact locations.

The CLI exits with code 4 when user review is required, 3 for interface/build/export failure, and 2 for preflight failure. Stage 05 can also be replayed from saved 02–04 artifacts via `scripts/run_stage05_from_artifacts.py`; see [application-stages.md](application-stages.md).
