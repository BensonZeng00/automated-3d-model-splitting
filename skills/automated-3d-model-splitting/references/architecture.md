# Object-Oriented Architecture

## Stable entry point

`scripts/split_painted_3mf.py` is the public command-line entry point. It only adds the script directory to `sys.path`, imports `split3mf.cli.main`, and invokes it. Keep existing command lines compatible.

## Pipeline

`split3mf.pipeline.SplitPipeline` owns one complete split run. It coordinates services in this order:

1. read and normalize the source project;
2. recognize connected painted components;
3. select the root body from geometry and separator evidence;
4. infer the recursive-minimal assembly tree and deterministic depth-first execution steps;
5. plan safe inward directions and adaptive caps;
6. execute strict tree recursion, reloading each pending subassembly from the independent colored 3MF emitted by its parent step while replacing one active subassembly at a time;
7. compare source and assembled generated surfaces across deterministic depth views;
8. write the temporary grouped multi-object 3MF;
9. reload, validate, and atomically publish the result.

The pipeline may coordinate policy but must not duplicate geometry, XML, ZIP, or validation algorithms.

## Domain and configuration

`split3mf.domain` contains the cross-stage data contracts:

- `SplitConfig`: validated CLI namespace and paths;
- `LoadedProject`: normalized mesh and project metadata;
- `RecognitionResult`: recognized components and provenance;
- `AssemblyPlan`: parent, child, depth, and recursive-layer records;
- `CapDecision`: selected cap policy and measured travel;
- `PartBuildResult`: built mesh plus annotations;
- `ValidationReport`: package and topology validation result.

Prefer these explicit records when data crosses stage boundaries. Do not introduce new unstructured global dictionaries for pipeline state.

## Services

- `ThreeMFReader` reads vendor packages and resolves build/component transforms.
- `VendorPaintDecoder` restores composite `paint_color` subdivision streams.
- `PartRecognizer` groups material-equivalent, edge-connected exterior paint.
- `BodySelector` scores body candidates and excludes definite structural separators.
- `AssemblyPlanner` infers and repairs the recursive-minimal parent tree, plans depth-first steps, and validates state transitions.
- `InwardDirectionPlanner` creates locally safe, smoothed inward directions.
- `BoundaryFairingService` fairs cut-loop positions with physical arc-length weights, feature locks, source-normal constraints, and a hard displacement bound.
- `AdaptiveCapPlanner` tries flat caps first and selects local-offset fallback only when needed.
- `BoundaryTriangulator` closes single-loop and holed boundaries without center fans.
- `PartMeshBuilder` builds inserts, body cuts, sockets, and subassemblies.
- `ThreeMFWriter` serializes the standard colored multi-object package.
- `ValidationService` checks in-memory meshes and reloads the written package.
- `ValidationService` also performs offline multi-view surface/depth consistency checks; it never controls a slicer UI.

Services should be stateless where practical. Inject or replace collaborators through `SplitPipeline` rather than reaching into CLI parsing.

## Module ownership

- `cli.py`: arguments, preflight, dependency loading, exit behavior;
- `project.py`: project metadata, material slots, unit and transform resolution;
- `recognition.py`: paint decoding and component recognition;
- `selection.py`: structural evidence and root-body selection;
- `assembly.py`: parent inference, cycle repair, recursive layers;
- `inward.py`: safe directions, cap planning, inward mesh construction;
- `boundary_fairing.py`: cut-loop position fairing and source-id canonicalization;
- `mesh.py`: generic mesh and boundary operations;
- `package_io.py`: 3MF component-assembly serialization and package metadata;
- `validation.py`: topology and package validation;
- `debug_export.py`: strict parent-emitted-part 3MF recursion plus explicitly requested standalone colored-part and cumulative audit exports;
- `reporting.py`: user and machine-readable result reporting;
- `common.py`: small shared constants and pure helpers only.

Avoid circular imports. A lower-level module must not import `pipeline.py` or `cli.py`.

## Compatibility invariants

An architecture-only refactor must preserve:

- command-line arguments, defaults, exit codes, and progress annotations;
- recognition ids, colors, body id, and assembly parents;
- generated vertex/triangle order and object annotations;
- validation decisions and atomic-output behavior;
- complete output 3MF bytes for deterministic golden inputs.

Before installing an architecture change, compile all modules, run `--version` and `--preflight-only`, then compare complete output SHA-256 values against the pre-refactor Panda and Speedboat baselines.
