# Best-effort quality reporting and low-volume diagnostics

The CLI inventories, selects, assesses, recognizes, plans and builds in that order.
CLI geometry quality checks are non-blocking by default. Checks keep their original
thresholds and report actual findings; a completed export is not a printability claim.
Use `--strict-quality` to restore blocking geometry checks. Python library calls
outside a report scope remain strict. Full and Stage 05 replay CLIs share this policy.

Construction keeps existing bounded recovery. A failed interface is skipped; a failed
part composition retains its source region as a reference mesh. A failed Boolean
keeps the pre-carve host. Other parts continue. Never label a reference mesh or a
missing connector as completed. Palette export errors retry a basic colored package,
with loss of vendor settings/material slots explicitly reported. Geometric readback
findings warn, but an unreadable package is never published.

Every invocation writes a Markdown split report and companion JSON. Part PLY files
are checkpointed beside the report so a later failure preserves completed work.
The CLI JSON summary supplies `split_report` and `quality_policy`. Report findings
include stage, identity, measured values/original limits when available, fallback
actions and skipped interfaces. Visual consistency remains `not_checked` until run.
Users inspect the exported model to decide printability.

Input parsing, missing dependencies, resource budgets, object selection, source and
artifact identity, role-table integrity, review gates and I/O remain blocking.
Do not modify source files, code, thresholds or confirmed roles during a split.

## Selection

```powershell
python scripts/split_painted_3mf.py inspect --input model.3mf
python scripts/split_painted_3mf.py assess --input model.3mf --object-id '3D/Objects/object_1.model#1'
python scripts/split_painted_3mf.py --input model.3mf --object-id '3D/Objects/object_1.model#1'
```

Use actual IDs from the inventory. With multiple mesh objects, missing or ambiguous selection returns `selection_required` (exit 5) before geometry decoding. Show the list and ask which object, unless the user has already identified it unambiguously. One mesh is selected automatically. `--model-entry` remains an explicit selector only when it resolves uniquely; it never selects the largest object. Multiple meshes inside one entry are selected by `entry#mesh-id`. A build instance that cannot be resolved uniquely is rejected by the existing transform validator.

Inventory names come from object metadata or vendor part metadata. Dimensions are local source coordinates with their source unit; literal color tokens are not a decoded material count. Paint selectors require decoding, so inventory must not present them as confirmed material counts.

## Complexity budgets

Face counts, raw/effective region counts, loop counts and boundary point counts are informational only. Remove obsolete count keys from custom configurations. Tiny regions follow the existing noise rule; counting never triggers automatic merging or repair.

Assess actual structure at the stage with sufficient evidence: Branching alone does not reject a region. Actual loop extraction failures, invalid loops and unsafe self-contact provide part/loop evidence; selected-interface construction failures are isolated in best-effort mode. Multiple loops alone are supported; ambiguous relationships, conflicting interface constructions and fit failures are checked by the existing construction and validation stages, not inferred from counts. Load/connectivity assessment remains `undetermined`, with boundary, interface-relationship and fit checks pending. No stage certifies checks it has not performed.

`--complexity-config` now controls only explicit resource budgets: `paint_token_chars` (16384) and `model_xml_bytes` (268435456). Exceeding these reports `budget_exceeded`, not geometric complexity. Structural rejection reports `unsupported`; invalid/insufficient boundary data reports `undetermined`. All admission rejections retain exit code 6. Strict construction/fit failures use exit 3; best-effort construction failures are reported as warnings, never relabeled as complexity failures. Replay validates the interfaces selected by the assembly plan under the same structural rules.

The prescribed Stage 03 simplification/smoothing and bounded 2 mm point-contact cleanup remain. Audit only the shared boundaries actually selected for interfaces, using the established Stage 05 bounded-contact policy; do not reject all source/candidate loops in advance: tolerable contact is recorded without changing frozen coordinates; unsafe contact skips the affected interface in best-effort mode and exits in strict mode. Existing finite depth backoff, collar-width candidates, output closure and winding/inversion correction run inside their original constraints. Geometry and readback validation still apply. Do not add ad hoc repair, alter source files or loosen checks while executing a split.


## Log contract

Each CLI invocation writes a unique JSONL log plus a sibling `.summary.json`. `--log-file` chooses the log path; `--log-level` controls console verbosity, default `WARNING`. Full detail always remains local. Main, Stage 04 and Stage 05 CLIs share this contract.

Each line has `time`, `level`, `stage`, `event`, `object_id`, `interface_id`, `elapsed_seconds`, `message` and `details`. Unknown identities are null rather than guessed. INFO records contain scalar stage data; large messages and nested diagnostics are written as DEBUG detail records. Errors retain bounded summaries and full DEBUG evidence. Legacy output is captured as DEBUG rather than copied to the terminal.

Read the final summary and exit code first. On failure, filter before reading:

```powershell
rg -n -m 10 '"level":"ERROR"' run.jsonl
rg -n -m 20 '"interface_id":"I001"' run.jsonl
rg -n -m 20 '"level":"WARNING"' run.jsonl
```

Only expand the implicated stage/event and relevant DEBUG records if the error is not explained. Never read the whole log by default. Avoid frequent empty polling; inspect output at stage changes or every 30–60 seconds. The summary includes review/artifact paths when a gate requires human input.

Exit codes: 0 completed/inspected assessment operation; 2 invalid arguments/environment; 3 execution/geometry failure; 4 review required; 5 object selection required; 6 structural rejection, undetermined input or resource budget exceeded. Exit 0 means the requested operation completed, possibly with quality warnings; inspect the split report. An assessment with status `undetermined` is not a printability claim.

## Images

Review output adds `review_overview_XX.png`, five numbered candidates per page with six zoom views each. Read these first. Inspect individual `Fxxx_context_and_zoom.png` only when context, identity or small details remain unclear. Keep original images on disk and never replace uncertain semantics with guesses. Failed complexity gates generate no review images. Reuse fingerprint-validated stage artifacts instead of parsing the source repeatedly.
