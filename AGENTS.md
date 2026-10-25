# Repository work rules

## Knowledge base after pulling code

Whenever you run `git pull` or otherwise incorporate code from a remote branch into this checkout, review the incoming commits and update `knowledge/` in the same task. Extract reusable development guidance: invariants, failure patterns, diagnostic steps, decision rules, and trade-offs. Keep model-specific incidents, commit ranges, and test run logs in the task or commit record rather than the general guidance page. Link new or revised guidance from `knowledge/README.md`. If the pull yields no reusable lesson, record that finding briefly rather than inventing one.

Before pushing code changes made in this repository, update the knowledge base when they teach or change a reusable rule. Keep guidance tied to observed code and test evidence; distinguish confirmed rules from hypotheses. Correct outdated guidance explicitly instead of silently leaving contradictory advice.

Generated model files are ignored by default. Add one only when the user explicitly requests publication of that artifact, and document its input and validation in the task or release record.

## Installation consent

Before installing this skill or its dependencies for a user, ask whether they have a Coding Plan (coding assistant subscription or usage allowance). Explain that model splitting may consume substantial tokens, complex models have a relatively high risk of splitting failure, and failed runs still consume usage, and actual allowances and charges depend on the provider; a plan does not imply unlimited or free usage. Obtain both the user's plan answer (yes, no, or unsure) and explicit consent to proceed after this notice. An initial install request or a plan answer alone is not consent. Without consent, stop before copying the skill, installing dependencies, or running a sample. Do not require purchasing a plan, inspect billing credentials, or repeatedly ask when informed consent is already recorded in the current installation session. This gate applies to user installations, not isolated packaging tests or development edits.
