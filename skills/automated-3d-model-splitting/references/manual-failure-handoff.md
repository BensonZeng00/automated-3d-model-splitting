# Bounded retries and user-edited cache recovery

## Retain before retrying

The agent owns attempt counting and user handoff. The current CLI does not
automatically retry three times, watch for edits, or graft an edited branch
into a completed assembly. Do not invent flags for these capabilities.

Before the first cutting attempt, preserve a persistent, task-owned recovery
directory. Keep the exact unscaled standalone input for each failing stage,
source SHA-256, parent/stage/part identity, palette, units, assembly coordinates,
options, implementation fingerprint, and failure logs. Keep a small manifest
recording each attempt's input hash, substantive change, failure reason and
artifact paths. Do not inspect unrelated user models.

Use existing `--debug-recursive-3mf` with a unique attempt output path to retain
recursive layer inputs/outputs; verify the actual debug paths reported by the
runner. This is authorized recovery work, not an unsolicited final preview.
Debug export currently cannot be combined with `--resume auto|strict`; use
`--resume off` for debug runs. When using validated checkpoint mode instead,
copy the last valid standalone input into recovery storage before a retry.
Never rely solely on a temporary directory that disappears on failure.

Keep three things distinct: immutable validated checkpoints, unvalidated failed
candidates, and the editable handoff copy. Save a failed candidate if one was
materialized and label it unvalidated. If preflight failed before any candidate
existed, provide the last valid input of the affected pending subassembly and
say explicitly that it is the input to the failed step, not a failed output.
Never fabricate a failed-part file or hand off a cumulative assembly as a
standalone recursive input.

## Local repair before handoff

Apply print-tolerance-and-replay.md first. Source triangle identity failures and tolerated cosmetic defects should be diagnosed/repaired locally, not sent to the user for manual editing. Runtime failures do not consume a geometric retry. User-authorized skill repair is a separate software task; validate changes with local replay.

## Stop and hand off

Count a completed unsuccessful stage attempt against the same unchanged input
hash and interface. Internal optimization candidates are not separate attempts;
successful siblings do not reset the failing stage's counter. Dependency or
permission errors and exit-code-4 user-review pauses are not geometry failures.
Each retry must have a reasoned change; do not repeat deterministic failures
unchanged merely to reach three. Stop earlier if authority or a user choice is
needed. After three failures, stop repeating that unchanged local strategy.
Apply completion-first.md: a supported local discard, merge, or materially different
repair may continue without user handoff when it preserves usable parts and assembly.
Attempt count alone does not require manual editing; hand off only when meaningful
bounded repairs are exhausted or the remaining tradeoff needs the user's choice.

On handoff, report in plain language:

- Failed part and parent/stage, attempt count, and a short measured reason.
- A clickable absolute path to the exact editable `.3mf`, not just its folder.
- A separate failed-candidate path if available, clearly marked unvalidated.
- A deterministic failure/boundary image when geometry is available; otherwise
  state why no failure geometry could be rendered. Do not launch another full
  split just to draw the failure.
- Ask the user to edit the provided copy, save it, and confirm the same path
  or provide their new path. Preserve millimeters, colors and assembly position;
  do not apply the final 99% shrink to the editable intermediate.

Example: “这个零件已尝试 3 次仍失败，自动重试已停止。请修改
[可编辑阶段 3MF](<absolute-path>) 后保存，告诉我‘已保存’；我会读取你
修改后的版本继续拆。原文件和已通过的缓存保留。” Replace the link and
part details with verified existing artifacts; never leave a placeholder.

## Continue only from the saved edit

Wait for user confirmation; no background file watcher is implied. Recompute
the editable file's SHA-256 and compare with the recorded handoff hash. If it
has not changed, ask the user to check saving or supply the actual edited path.
If changed, preserve that version and load it as a new source input with fresh
recognition, topology and color checks. Never overwrite it with an old cache.
Rebuild face IDs, boundary correspondence, local topology and stage identity;
do not reuse source-specific boundary approvals against changed geometry.

Start a new attempt sequence for the edited input hash and retain the prior
failure history. Use a fresh output/cache namespace so old descendants cannot
be mistaken for valid results. Continue on this standalone pending subassembly
with the ordinary splitter. Preserve unaffected sibling artifacts. Reuse or
reassemble them only after verifying provenance and mating-interface agreement;
if the edit changes its parent-contact shell, recompute the affected parent cut
from the preserved pre-cut parent. If that parent is unavailable, explain and
request the necessary input rather than silently attaching incompatible parts.
Report a standalone branch result as such until full assembly integration is
actually verified.

Use only exact full-size child subtraction followed by one final XYZ uniform
scale (default 0.99) of completed inward parts. Recursive cache and editable
stage inputs must remain pre-scale. A branch's local body is not automatically
the global body: for final assembly integration, determine the global part role
and apply the final shrink exactly once to every global inward part, leaving
the global root unchanged. Keep existing geometry, color and assembly checks;
manual repair is new input, not permission to declare an invalid mesh printable.
