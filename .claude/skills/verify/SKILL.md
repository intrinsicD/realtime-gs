---
name: verify
description: Run the full repository verification (lint, format, CPU tests, docs-sync) and fix what fails. Use before any commit, after any refactor, or when asked "does everything still pass?".
---

# Verify

Run:

```bash
./scripts/verify.sh
```

It executes, in order: `ruff check`, `ruff format --check`, `pytest -q -m "not slow"`,
`python scripts/docs_sync.py`. CI runs the identical sequence, so a clean local run means
a green CI.

## Interpreting failures

- **ruff format**: run `.venv/bin/ruff format .` to fix, then re-verify.
- **pytest quality-threshold failures** (PSNR/error floors in tests): these encode minimum
  acceptable behavior. Investigate the regression — do NOT lower a threshold without a
  dated justification entry in `docs/EXPERIMENTS.md`.
- **docs_sync failures**: each message names the drifted artifact (undocumented subpackage,
  phantom CLI command, missing skill listing, broken path in CLAUDE.md, missing module
  docstring). Fix the docs or the code, whichever is actually stale.
- **cuda-marked tests** are skipped automatically on CPU boxes; that is not a failure.

If you changed anything under `src/rtgs`, also run the slow suite once when the change is
substantial: `.venv/bin/pytest -q` (includes `-m slow` tests).
