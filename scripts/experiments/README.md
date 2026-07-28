# scripts/experiments/

One-off, experiment-specific scripts live here. `scripts/` itself is reserved for durable
repository tooling that every agent and CI run may need.

## Which directory does my script go in?

| Script | Goes in |
|---|---|
| Runs on every commit, or is part of `verify.sh` / CI | `scripts/` |
| Reusable across experiments (migration, gallery rendering, a validator) | `scripts/` |
| Drives one registered result-bearing experiment task | `scripts/experiments/<task_id>.py` |
| Throwaway, never to be committed | the session scratchpad, not the repo |

`scripts/check_script_layout.py` enforces this: any new top-level file in `scripts/` must be
added to the `DURABLE_SCRIPTS` allowlist in that checker, with a reason. If your script is
experiment-specific, put it here instead and no allowlist entry is needed.

## Grandfathered scripts (do not move)

Two experiment-specific scripts remain at the top level of `scripts/` and are pinned there:

- `scripts/verify_iter1e_development_tree.py`
- `scripts/write_iter1e_verification_receipt.py`

Both paths are bound by source hash in `DECLARED_SOURCE_PATHS` in
`benchmarks/inverse_projection_fiber_iter1e.py` and cited by sealed result notes under
`benchmarks/results/`. Moving them would invalidate the replay integrity of committed evidence,
which the `realtime-gs-results-audit` skill forbids. They are allowlisted with that reason.

The lifecycle policy applies to new scripts. Existing sealed provenance is never rewritten to
satisfy a layout rule.

## Conventions for scripts here

- Register `experiments/tasks/YYYYMMDD_<task_slug>_<data_slug>.json` before creating the script.
- Name the script exactly `<task_id>.py`.
- Put the exact invocation in the module docstring; the task pins seeds, splits, metrics, and
  resource accounting.
- Write only below `runs/<task_id>/`; use the shared experiment report renderer.
- Reference the task, `docs/EXPERIMENTS.md` entry, and matching `benchmarks/results/` artifacts.
- When the experiment closes, leave the script here. It is provenance, not clutter.
