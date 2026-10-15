# Repository work rules

## Knowledge base after pulling code

Whenever you run `git pull` or otherwise incorporate code from a remote branch into this checkout, review the incoming commits and update `knowledge/` in the same task. Extract reusable development guidance: invariants, failure patterns, diagnostic steps, decision rules, and trade-offs. Keep model-specific incidents, commit ranges, and test run logs in the task or commit record rather than the general guidance page. Link new or revised guidance from `knowledge/README.md`. If the pull yields no reusable lesson, record that finding briefly rather than inventing one.

Before pushing code changes made in this repository, update the knowledge base when they teach or change a reusable rule. Keep guidance tied to observed code and test evidence; distinguish confirmed rules from hypotheses. Correct outdated guidance explicitly instead of silently leaving contradictory advice.

Generated model files are ignored by default. Add one only when the user explicitly requests publication of that artifact, and document its input and validation in the task or release record.
