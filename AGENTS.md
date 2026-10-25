# Repository work rules

## Plain-language installation requests

A request to install this Skill and its dependencies from this repository is sufficient to start the installation workflow. The user does not need to name an installer, provide shell commands, choose a Python path, specify a version, or repeat the consent policy in their request. Read [tools/INSTALL.md](tools/INSTALL.md) and use `tools/install_skill.py`; do not stop at giving the user manual commands or copy only `SKILL.md`. Select an available supported Python and the current assistant's Skill destination from the environment. For Codex, the installer supplies the default destination; for other assistants, pass their known destination with `--target` and ask only if it cannot be determined.

The initial request starts the workflow but does not satisfy the installation-consent gate below. Show the exact notice below and collect explicit consent in chat when there is no interactive terminal, then use `--accept-usage` only for that consent in the same session. After a successful installation, verify the installed CLI with `--version` and `--help` from an unrelated working directory before reporting success. If the environment cannot run the installer, report the concrete blocker rather than claiming installation succeeded.

## Knowledge base after pulling code

Whenever you run `git pull` or otherwise incorporate code from a remote branch into this checkout, review the incoming commits and update `knowledge/` in the same task. Extract reusable development guidance: invariants, failure patterns, diagnostic steps, decision rules, and trade-offs. Keep model-specific incidents, commit ranges, and test run logs in the task or commit record rather than the general guidance page. Link new or revised guidance from `knowledge/README.md`. If the pull yields no reusable lesson, record that finding briefly rather than inventing one.

Before pushing code changes made in this repository, update the knowledge base when they teach or change a reusable rule. Keep guidance tied to observed code and test evidence; distinguish confirmed rules from hypotheses. Correct outdated guidance explicitly instead of silently leaving contradictory advice.

Generated model files are ignored by default. Add one only when the user explicitly requests publication of that artifact, and document its input and validation in the task or release record.

## Installation consent

Before installation, display this exact notice and wait for explicit acceptance. Repeat the same notice after successful installation without requesting confirmation again:

> 拆件可能消耗较多 token，复杂模型有可能失败；如果没有 Coding Plan，请谨慎使用，并对 token 消耗做好预期。明确确认后才安装

Do not separately ask about Coding Plan status. The initial installation request is not consent. Without explicit acceptance, stop before copying the Skill, installing dependencies, or running a sample. Reuse informed consent already recorded in the current installation session; do not invent or persist it for future installations. This gate applies to user installations, not development edits or isolated packaging tests.

All assistant-managed user installations must use tools/install_skill.py. Never invent consent flags or bypass the installer by copying files or running pip separately.
