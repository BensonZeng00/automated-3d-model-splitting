# Stages and artifacts

Selection, complexity gates, strict failure behavior and log reading follow [execution-policy.md](execution-policy.md). These rules also apply to artifact replay.

`split3mf.pipeline.SplitPipeline` runs the current painted-3MF workflow. `StageArtifactStore` writes stage JSON/NPZ files under a unique run ID. The source 3MF is read only once in a full run.

| Stage | Input | Artifacts |
|---|---|---|
| 01 preflight | Source path and environment | `01_preflight.json` |
| 02 load | Vendor-painted 3MF | `02_loaded_project.npz`, `02_loaded_project_summary.json` |
| 03 recognize | Source mesh, exterior paint and review decisions | `03_region_review_checkpoint.npz` when source-region review pauses; then `03_recognition_regions.npz`, `03_recognized_boundaries.npz`, `03_recognition_summary.json`, review image and decision files |
| 04 assembly | Frozen simplified boundaries and region geometry | `04_assembly_plan.json`, `04_interface_table.md`, `04_assembly_review.json` |
| 05 interface-assembly | Stage 04 relations, frozen boundaries, scale and clearance | `05_interface_surfaces.npz`, `05_interface_assembly_meshes.npz`, `05_interface_assembly_summary.json` |
| 08 publication | Closed parts and source project metadata | `08_validated_publish.json`, grouped 3MF |

Stage 03 selects equal-arc source samples (approximately 5% for large loops, with a 0.5 mm maximum target spacing), increases the count when sampling would distort the contour, applies 4 mm periodic Gaussian smoothing when its scale-normalized shape guard passes, and freezes the ordered points. Small loops retain the existing higher sampling fractions. Stage 04 and Stage 05 consume that snapshot without remapping it onto the original edge chain. Recognition actions and user confirmation are fingerprint-bound. Region review is capped at the 10 largest qualifying candidates by source surface area by default; additional qualifying regions are recorded as automatic noise. A first full run may stop with exit code 4 and write a review template. Confirm the current result before continuing to Stage 04. Stage 04 has its own fingerprint-bound review gate: inspect the interface table, record any wrong pairing or role in `correction_requests`, revise Stage 04, then explicitly confirm the revised plan before Stage 05. Both full runs and Stage 05 artifact resumes stop with exit code 4 while this review is pending.

Boundary pairing uses one symmetric coverage rule for both closed-loop counterparts and fragmented host boundary unions. Distances are measured from samples to line segments in both directions; each direction must have at least 80% coverage within the existing scale-derived tolerance. Reports retain tolerance, P50/P90/maximum distances, and directional coverage. Fragmented-host candidates retain the existing unique-host check.

Use `--stop-after-stage` with `preflight`, `load`, `recognize`, `assembly`, or `interface-assembly` to inspect a stage. `--stage-artifacts-dir` selects its output root. Each NPZ has a manifest listing its arrays and checksums.

When a source-region review pauses Stage 03, confirm its `user_decisions.json` and continue with the saved run ID:

```powershell
python scripts/split_painted_3mf.py `
  --input .\model.3mf `
  --region-review-json .\review\user_decisions.json `
  --resume-region-review-from .\artifacts\model_stages\<run-id>
```

The checkpoint preserves visibility, connectivity, candidate identities and automatic noise exclusions. Resume validates source SHA-256, Stage 02/03 manifests, recognition options and producer code before applying decisions. Runs created before this checkpoint existed reuse Stage 02 and recalculate candidate work, still avoiding 3MF parsing. The continued run writes a new artifact directory and follows the normal Stage 03 review gate.

To rerun Stage 05 from completed 02–04 artifacts without rereading the input model:

```powershell
python scripts/run_stage05_from_artifacts.py `
  --source-run-dir .\artifacts\model_stages\<run-id> `
  --output-root .\artifacts\model_stage05 `
  --assembly-review-json .\artifacts\model_stages\<run-id>\04_assembly_review.json `
  --interface-scale-ratio 0.50 `
  --interface-clearance-mm 0.20
```

The replay validates existing artifact manifests and the frozen boundary fingerprint. It builds complete colored parts and exports a grouped 3MF in a new run directory. The resulting meshes and reloaded package are recorded in the Stage 05 summary.

Curved simplified boundaries automatically use a root-capped shoulder without a protruding connector. Shape classification uses scale-normalized, arc-length-weighted sagitta from the contact rim's best-fit plane. It requires at least 2 mm and 20% of the smaller in-plane span in sustained deviation over at least 90% of the rim; lesser bows use the ordinary scaled tapered connector. No per-interface routing flags are required or accepted. When changing the classifier, rerun the complete connector harness so square, curved, and concave-V production cases remain valid.

If Stage 03 completed but the run stopped before Stage 04 because semantic labels
were missing or supplied later, reuse its saved mesh, region, and frozen-boundary
artifacts to generate Stage 04 without parsing the source 3MF again:

```powershell
python scripts/run_stage04_from_artifacts.py `
  --source-run-dir .\artifacts\model_stages\<run-id> `
  --input .\model.3mf `
  --visual-semantics-json .\review\part_semantics_proposed.json `
  --recognition-review-json .\artifacts\model_stages\<run-id>\03_recognition_review.json
```

The command verifies the source hash, Stage 02/03 artifact manifests, saved
recognition fingerprint, saved source-region classifications, and semantic labels.
If Stage 03 is still pending confirmation, pass a confirmed recognition review
JSON. The command compares its source, actions, and fingerprints with the saved
result, records the confirmation, and continues without rereading the 3MF mesh.
It adds Stage 04 artifacts to the same run only when that run has no existing
Stage 04 plan. Existing plans are preserved for review or Stage 05 resume.

Stage 05 uses one geometry path for every pair: the Stage 04 outer contour connects directly to a planar extended inner contour; the mortise is the matching recess. See [assembly-algorithm.md](assembly-algorithm.md). The frozen Stage 03 contour also controls each part's visible exterior near the interface. `surface_collar_remesh.py` clips source triangles at interpolated crossings of a contour-distance field, preserving outer faces and per-face paint origins; it then joins the resulting cut to the same frozen contour. Whole-triangle deletion and its stair-step rim are not used. Adjacent-interface spacing bounds the local reconstruction width. The one universal candidate loop compares transition self-crossings with the original connection and reduces the width when it would introduce more crossings. There is no recursive scheduler or Boolean merge.
