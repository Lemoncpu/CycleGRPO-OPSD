# Repository Instructions

Before changing any code, configuration, data conversion, training, or evaluation logic, read `code.md` and `Agent.md` in the repository root. Read the complete `code.md`, including the latest entries in its change log, before making the change.

After every code change:

1. Update the affected explanatory section in `code.md`.
2. Append an entry to the `code.md` change log with the date, changed files, behavioral impact, and verification performed. The new entry must be added after reviewing the existing log so the history remains chronological and cumulative.
3. Update the module inventory when files are added, moved, renamed, deprecated, or removed.

A code change is not complete until `code.md` is updated. If the implementation diverges from the paper or the documented main execution path, state that difference explicitly in `code.md`.
