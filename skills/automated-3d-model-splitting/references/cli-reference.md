# CLI reference

Current release: `3.0.0`.

Resolve all relative script paths against the directory containing the loaded `SKILL.md`. From another working directory, invoke the selected Python interpreter with the absolute script path and absolute input, output, and review-file paths. Install runtime dependencies from that skill directory's `requirements.txt`; see [installation and entry location](../SKILL.md#安装与入口定位).

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
| `--resume-region-review-from RUN_DIR` | Continue from saved Stage 02 and, when present, the pre-review Stage 03 checkpoint; requires `--region-review-json`. |
| `--recognition-review-json FILE` | Apply fingerprint-bound recognition actions and confirmation. |
| `--boundary-review-json FILE` | Apply confirmed boundary-ownership decisions. |
| `--visual-semantics-json FILE` | Supply optional part labels and physical partitions. |
| `--interface-scale-ratio NUMBER` | Projected inner-contour scale; default `0.50`. |
| `--interface-clearance-mm NUMBER` | Extra mortise tip radius and floor depth; default `0.20`. |
| `--output-layout assembly|separate-items` | Choose the grouped assembly (default) or separate build items. |

Visibility and review sampling remain configurable with `--exterior-view-count`, `--exterior-depth-map-resolution`, `--exterior-depth-tolerance-mm`, `--noise-review-max-faces`, `--small-region-review-max-faces`, `--max-region-review-candidates`, and `--region-review-resolution`. At most the largest 10 qualifying regions by total source area are shown for review by default; qualifying candidates beyond the limit are automatically classified as noise and excluded from part recognition while their source faces remain unchanged. `--model-entry`, `--format-profile`, `--color-map-json`, `--region-review-dir`, and `--recovery-dir` control their corresponding input or artifact locations.

The CLI exits with code 4 when user review is required, 3 for interface/build/export failure, and 2 for preflight failure. Stage 05 can also be replayed from saved 02–04 artifacts via `scripts/run_stage05_from_artifacts.py`; see [application-stages.md](application-stages.md).


## Best-effort quality output

Geometry checks warn by default and retain their measured values and original thresholds.
`--strict-quality` restores blocking geometry validation for full and artifact-replay CLIs.
Read `split_report` in the CLI JSON summary for the Markdown report; a companion JSON
contains full findings. PLY part checkpoints are saved in the report's `_partial_parts`
directory. Missing interfaces and reference source meshes are explicitly labeled.
File/identity/role-table errors, review requirements, resource budgets and I/O still stop.
The final model requires user inspection; successful export does not certify printability.
