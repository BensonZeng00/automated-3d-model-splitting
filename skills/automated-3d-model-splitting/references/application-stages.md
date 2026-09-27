# Stages and artifacts

`split3mf.pipeline.SplitPipeline` runs the current painted-3MF workflow. `StageArtifactStore` writes stage JSON/NPZ files under a unique run ID. The source 3MF is read only once in a full run.

| Stage | Input | Artifacts |
|---|---|---|
| 01 preflight | Source path and environment | `01_preflight.json` |
| 02 load | Vendor-painted 3MF | `02_loaded_project.npz`, `02_loaded_project_summary.json` |
| 03 recognize | Source mesh, exterior paint and review decisions | `03_recognition_regions.npz`, `03_recognized_boundaries.npz`, `03_recognition_summary.json`, review image and decision files |
| 04 assembly | Frozen simplified boundaries and region geometry | `04_assembly_plan.json` with pairwise tenon/mortise roles |
| 05 interface-assembly | Stage 04 relations, frozen boundaries, scale and clearance | `05_interface_surfaces.npz`, `05_interface_assembly_meshes.npz`, `05_interface_assembly_summary.json` |
| 08 publication | Closed parts and source project metadata | `08_validated_publish.json`, grouped 3MF |

Stage 03 simplifies each accepted boundary to approximately 5% of its source samples and freezes the ordered points. Stage 04 and Stage 05 consume that snapshot without remapping it onto the original edge chain. Recognition actions and user confirmation are fingerprint-bound. A first full run may stop with exit code 4 and write a review template. Confirm the current result before continuing to Stage 04.

Use `--stop-after-stage` with `preflight`, `load`, `recognize`, `assembly`, or `interface-assembly` to inspect a stage. `--stage-artifacts-dir` selects its output root. Each NPZ has a manifest listing its arrays and checksums.

To rerun Stage 05 from completed 02–04 artifacts without rereading the input model:

```powershell
python scripts/run_stage05_from_artifacts.py `
  --source-run-dir .\artifacts\model_stages\<run-id> `
  --output-root .\artifacts\model_stage05 `
  --interface-scale-ratio 0.50 `
  --interface-clearance-mm 0.20
```

The replay validates existing artifact manifests and the frozen boundary fingerprint. It builds complete colored parts and exports a grouped 3MF in a new run directory. The resulting meshes and reloaded package are recorded in the Stage 05 summary.

Stage 05 uses one geometry path for every pair: the Stage 04 outer contour connects directly to a planar extended inner contour; the mortise is the matching recess. See [assembly-algorithm.md](assembly-algorithm.md). Source exteriors are retained and locally connected to that outer contour. There is no recursive scheduler or Boolean merge.
