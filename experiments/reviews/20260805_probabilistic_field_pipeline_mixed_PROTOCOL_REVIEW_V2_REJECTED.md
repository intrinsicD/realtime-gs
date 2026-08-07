# Prospective Protocol Review

- Task ID: `20260805_probabilistic_field_pipeline_mixed`
- Protocol SHA-256: `3bc8c12ee2e782e53e51c44f85ad6efa846e4e63e289d3bf66d7d4879cf4c417`
- Reviewer: `Codex-probabilistic-field-protocol-reviewer`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

This was an outcome-blind v2 review of the repaired development protocol, exact result-producing
source binding, 549-cell plan, eleven sealed Gaussian2D datasets, input/split guards, decision
rules, failure semantics, resource accounting, aggregate publication, generated per-dataset
reports, tests, documentation, and repository workflow state.

If corrected and executed, the protocol could provide deterministic synthetic mechanism evidence
and development-only operability/utility observations for native-control and all-candidate lifting
over a deterministic 512-component-per-view proxy. It could not establish complete-field
fidelity, source-RGB reconstruction quality, true multi-marginal OT, GPS-Gaussian reproduction,
GPU or real-time performance, cross-scene generality, a production default, or reconstruction
accuracy from independent-half agreement.

## Checks

- Independently recomputed the digest with
  `.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260805_probabilistic_field_pipeline_mixed.json`.
  It exactly matched
  `3bc8c12ee2e782e53e51c44f85ad6efa846e4e63e289d3bf66d7d4879cf4c417`.
- Ran the outcome-free contract and byte-seal checks. Both
  `.venv/bin/python scripts/experiment_contract.py validate` and
  `.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260805_probabilistic_field_pipeline_mixed.json`
  returned `OK`.
- Inspected `--inspect-plan`: it contains 549 unique cells with the frozen counts (324 shape, 60
  association, 81 mask, 6 topology, 6 schedule, 6 independent-half, and 66 calibrated). Every
  schedule cell now declares five cleanup iterations; all eleven datasets have exactly three
  seeds and two calibrated arms.
- Reconfirmed the data inventory and split boundary established in v1: all and only the eleven
  `gaussians2d*` manifests are selected; the seal binds 309 compact/calibration files totaling
  204,306,829 bytes with no external RGB/mask file; every compact view belongs to exactly one of
  train or held-out. Nine Stage bundles carry embedded alpha and both Karate bundles are
  unmasked.
- Recomputed the task-carried source-tree digest through the plan compiler. It matches the
  length-prefixed bytes of the task driver, `scripts/experiment_contract.py`, and every
  `src/rtgs/**/*.py` file. The driver checks that digest before any execution. Development run
  locking now includes a content manifest for untracked files, and its unit test proves that an
  untracked byte change changes the lock input.
- Confirmed that each timed schedule cell is delegated to its own guarded subprocess and that the
  executed refit obtains the five-iteration cleanup value from the frozen task/cell rather than a
  hardcoded alternate value.
- Confirmed the repaired association utility surface: it corrupts the actual cost/gate rows for
  the shuffled negative while retaining unshuffled truth labels, uses a common minimum coverage
  scalar across treatment and controls, records candidate-gate mass, and records the Sinkhorn
  fixed-point residual.
- Confirmed that the mask factorial now runs the production `FieldLifter` support/mass/opacity and
  refit path on synthetic Gaussian fields with train-only nuisance components and a clean held-out
  field. Its utility decision checks support precision, support coverage, held-out density MSE,
  and held-out RGB-numerator MSE for Pareto nondominance against both hard and none.
- Confirmed that shape decisions now require both world-covariance and held-out projected
  covariance improvement. Source mean/covariance, production split conservation, candidate mass,
  transport finiteness/convergence, and progressive final-view measurements are emitted. The
  calibrated candidate uses `failure_policy="raise"` and checks hard result invariants before
  serializing its worker result.
- Confirmed structured `failure.json` production for synthetic-cell, synthetic-coordinator,
  calibrated-worker, orchestration, and aggregation failures, plus a failed `run_receipt.json`.
  Aggregate outputs are built under staging; `aggregate_commit_receipt.json` and the completed
  run receipt form explicit final publication markers. A mid-rename/power failure remains a
  detectable terminal failed run, as the task handoff now states, rather than being represented as
  a completed atomic commit.
- Confirmed calibrated resource receipts cover compact loading, fit, serialization, and directory
  publication and include wall/process/refit/stage times, RSS, CPU/thread/CUDA fields, input bytes,
  and output bytes. Aggregate resource receipts retain raw cell paths and per-dataset/arm
  min/median/max summaries over three repeats.
- Ran
  `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_probabilistic_field_experiment_protocol.py tests/test_probabilistic_field_pipeline.py tests/test_experiment_contract.py tests/test_field_refit.py`;
  all 48 focused outcome-free tests passed. Ran
  `.venv/bin/python scripts/check_agent_workflow.py`; it passed with the task in `In review` and
  the reviewer turn active.

## Findings

The v2 protocol is substantially repaired but is still rejected before outcome access. Three
remaining defects affect hard-gate validity and ordering. Correcting any of them changes the
reviewed source/protocol surface and therefore requires a new source digest, protocol digest, and
prospective review.

1. **Association hard failures are still not fail-closed at cell time.**
   `_association_cell` does not receive the task gates and returns transport real mass,
   finiteness, fixed-point residual, candidate violation, and dustbin values without enforcing
   them. `_synthetic_worker` therefore continues through the rest of the 483 synthetic cells.
   Only after every cell finishes does `_synthetic_decisions` aggregate the hard booleans; the
   coordinator then writes `synthetic_results.json` and `resource_receipt.json` before raising.
   This contradicts the frozen stopping rule to stop a cell immediately on non-finite transport,
   invalid balance, or another hard invariant. Enforce the frozen transport/candidate/dustbin
   thresholds inside each association cell (with its structured failure receipt) and do so before
   any later cell or calibrated work is launched. The aggregate recheck may remain as defense in
   depth.

2. **The held-out isolation gate is asserted, not measured.** For topology, schedule, and
   independent-half records, `_pipeline_synthetic_cell` writes
   `heldout_fit_access_count: 0` as a literal. Mask records omit the field entirely, and
   `_synthetic_decisions` silently substitutes zero through `.get(..., 0)`. Thus
   `heldout_fit_isolation` is guaranteed to pass even if the optimized indices violate the split.
   Compute the count from each result's actual optimized view indices versus the frozen held-out
   indices (and require the field rather than defaulting it). Keep the calibrated worker's
   existing explicit optimized/held-out disjointness check.

3. **The registered dustbin-capacity residual is a mathematical identity, not a validity
   measurement.** Both `_association_cell` and `_association_invariants` compute
   `abs(augmented.sum(dim=1).sum() - augmented.sum(dim=0).sum())`. For any finite matrix these are
   two summation orders over exactly the same entries, so the value is necessarily zero up to
   rounding regardless of dustbin capacities, target construction, or marginal behavior. It
   cannot detect the invalid accounting named by `dustbin_capacity_balance_tolerance`. Define a
   non-tautological dustbin/capacity invariant compatible with finite-penalty unbalanced transport
   (without falsely requiring balanced marginals), measure it for synthetic and calibrated plans,
   and gate it per cell. Alternatively remove/rename the unsupported gate and obtain a new review.

All other v1 rejection items inspected in this pass are resolved sufficiently for a bounded
development experiment: exact source bytes are bound; schedule treatments and process freshness
match; association coverage and shuffled-input semantics are implemented; the mask test is
field-level; shape/source/split/final-view measurements exist; candidate association raises;
failures are structured; aggregate completion is explicitly marked; resource scope is reported;
and the task workflow plus focused tests pass. Those repairs do not make the three remaining hard
gates evidentiary or fail-closed.

## Protected Actions Not Taken

I did not initialize or execute the canonical run, run any official-seed synthetic cell or
calibrated worker, inspect or create
`runs/20260805_probabilistic_field_pipeline_mixed`, access any result/outcome artifact, inspect the
claimed retained rehearsal outcomes, render a report, launch a viewer, update the task/index/task
state, or make any quantitative interpretation. Outcome access remained `none` throughout.
