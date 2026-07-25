---
name: rtgs-verify
description: Run the full repository verification (lint, format, CPU tests, docs-sync) and fix what fails. Use before any commit, after any refactor, or when asked "does everything still pass?".
---

# Verify

Run:

```bash
./scripts/verify.sh
```

It executes, in order: `ruff check`, `ruff format --check`, `pytest -q -m "not slow"`,
`python scripts/docs_sync.py`, `python scripts/check_ara.py`,
`python scripts/check_script_layout.py`. CI runs the identical sequence, so a clean local run
means a green CI.

## Interpreting failures

- **ruff format**: run `.venv/bin/ruff format .` to fix, then re-verify.
- **pytest quality-threshold failures** (PSNR/error floors in tests): these encode minimum
  acceptable behavior. Investigate the regression — do NOT lower a threshold without a
  dated justification entry in `docs/EXPERIMENTS.md`.
- **docs_sync failures**: each message names the drifted artifact (undocumented subpackage,
  phantom CLI command, missing skill listing, broken path in CLAUDE.md, missing module
  docstring). Fix the docs or the code, whichever is actually stale.
- **cuda-marked tests** are skipped automatically on CPU boxes; that is not a failure.
- **check_ara failures**: a claim in `ara/logic/claims.md` is missing a required field, uses an
  unknown status word, depends on an undefined claim, or cites a proof path that no longer
  exists. Fix the ledger row — never delete the claim to silence the checker. See the "Evidence
  and claims" section of CLAUDE.md.
- **check_script_layout failures**: a new top-level file in `scripts/` is not declared durable.
  Move it to `scripts/experiments/`, or add it to `DURABLE_SCRIPTS` with a reason.

If you changed anything under `src/rtgs`, also run the slow suite once when the change is
substantial: `.venv/bin/pytest -q` (includes `-m slow` tests).
