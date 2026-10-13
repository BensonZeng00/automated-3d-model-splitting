# Repository work rules

## Knowledge base after pulling code

Whenever you run `git pull` or otherwise incorporate code from a remote branch into this checkout, review the incoming commits and update `knowledge/` in the same task. Record the source branch and commit range, behavior changes, affected files, validation results, known limitations, and any migration or follow-up work. Link the new entry from `knowledge/README.md`. If the pull contains no relevant code change, record that finding briefly rather than inventing a technical lesson.

Before pushing code changes made in this repository, update the knowledge base for those changes as well. Keep entries tied to observed code and test evidence; distinguish a confirmed fix from a diagnostic hypothesis. Do not rewrite past entries to hide superseded conclusions; add a correction or follow-up entry.

Generated model files are ignored by default. Add one only when the user explicitly requests publication of that artifact, and document its input and validation in the matching knowledge entry.
