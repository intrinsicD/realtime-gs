# scripts/experiments/

One-off, experiment-specific scripts live here. `scripts/` itself is reserved for durable
repository tooling that every agent and CI run may need.

## Which directory does my script go in?

| Script | Goes in |
|---|---|
| Runs on every commit, or is part of `verify.sh` / CI | `scripts/` |
| Reusable across experiments (migration, gallery rendering, a validator) | `scripts/` |
| Drives one experiment, one protocol, or one dataset conversion | `scripts/experiments/` |
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

- Name the experiment: `<protocol>_<what>.py` (e.g. `iter2_depth_covariance_sweep.py`).
- Put the exact invocation in the module docstring, and pin seeds.
- Reference the `docs/EXPERIMENTS.md` entry or `benchmarks/results/` artifact the script
  produced, so a reader can get from script to result and back.
- When the experiment closes, leave the script here. It is provenance, not clutter.
