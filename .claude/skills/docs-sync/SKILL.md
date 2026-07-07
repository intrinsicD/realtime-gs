---
name: docs-sync
description: Reconcile documentation with code using the docs_sync checker. Use when docs_sync fails in verify/CI, after adding/removing modules, CLI commands, lifters, or skills, or when asked to update the docs.
---

# Docs sync

```bash
.venv/bin/python scripts/docs_sync.py
```

The checker is structural: it verifies required docs exist, every `rtgs` subpackage is
described in `docs/ARCHITECTURE.md` and CLAUDE.md, CLI commands and registered lifters
match their documentation, `.claude/skills/*` are listed in CLAUDE.md, paths referenced in
CLAUDE.md exist, all modules have docstrings, and the benchmark markers are intact.

## Resolving drift

For each reported problem decide which side is stale:

- New code, missing docs → document it (ARCHITECTURE for design, CLAUDE.md map for layout,
  BENCHMARKS for perf surface).
- Docs describe something removed → delete the stale doc text; check ROADMAP.md whether the
  removal closes or reopens an item.
- Adding a new check: extend `scripts/docs_sync.py` (keep checks structural — they must not
  produce false positives on prose edits).

After fixing, re-run the checker AND `./scripts/verify.sh` before committing.
