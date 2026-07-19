# Repository Instructions

Before changing any code, configuration, data conversion, training, or evaluation logic, read `code.md` and `Agent.md` in the repository root.

After every code change:

1. Update the affected explanatory section in `code.md`.
2. Append an entry to the `code.md` change log with the date, changed files, behavioral impact, and verification performed.
3. Update the module inventory when files are added, moved, renamed, deprecated, or removed.

A code change is not complete until `code.md` is updated. If the implementation diverges from the paper or the documented main execution path, state that difference explicitly in `code.md`.
