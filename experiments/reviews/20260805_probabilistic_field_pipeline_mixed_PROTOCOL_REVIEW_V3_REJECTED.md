# Prospective Protocol Review

- Task ID: `20260805_probabilistic_field_pipeline_mixed`
- Protocol SHA-256: `5ea3680db4b33c36d2fbdf334a0d114f0ca5230d52c6321f5b450d8790f3d0e1`
- Source-tree SHA-256: `0d574a1989cd5fa428d74a18b7f55491b0605fb084524b9d6e91c9fe91092e6e`
- Reviewer: `Codex-probabilistic-field-protocol-reviewer`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

This was an outcome-blind v3 review of the repaired development protocol, its exact
result-producing source binding, the 549-cell plan, eleven sealed Gaussian2D datasets, split and
input guards, hard invariants, failure ordering, calibrated workers, independent-half audit, and
repository workflow state. I read the latest driver handoff and both append-only rejection
artifacts before checking the v3 source independently.

If corrected and executed, the protocol could provide deterministic synthetic mechanism evidence
and development-only operability/utility observations for native-control and all-candidate lifting
over a deterministic 512-component-per-view proxy. It could not establish complete-field
fidelity, source-RGB reconstruction quality, true multi-marginal OT, GPS-Gaussian reproduction,
GPU or real-time performance, cross-scene generality, a production default, or reconstruction
accuracy from independent-half agreement.

## Checks

- Independently recomputed the review digest with
  `.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260805_probabilistic_field_pipeline_mixed.json`.
  It exactly matched
  `5ea3680db4b33c36d2fbdf334a0d114f0ca5230d52c6321f5b450d8790f3d0e1`.
- Recomputed the task's length-prefixed source binding over the driver,
  `scripts/experiment_contract.py`, and all `src/rtgs/**/*.py` files. All 102 bound files produce
  `0d574a1989cd5fa428d74a18b7f55491b0605fb084524b9d6e91c9fe91092e6e`, exactly matching the task.
- Ran `.venv/bin/python scripts/experiment_contract.py validate` and
  `.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260805_probabilistic_field_pipeline_mixed.json`;
  both returned `OK`.
- Inspected the outcome-free plan. It has 549 unique cells with the frozen stage counts: 324 shape,
  60 association, 81 mask, 6 topology, 6 schedule, 6 independent-half, and 66 calibrated. The
  schedule cells use five cleanup iterations, and the calibrated matrix is eleven datasets by
  three seeds by two arms.
- Reconfirmed the sealed input boundary: eleven datasets and 309 files totaling 204,306,829 bytes,
  comprising 296 compact `.rtgsv` files, eleven compact manifests, and two calibration files, with
  no image file in the seal.
- Confirmed the first v2 blocker is repaired for synthetic UOT cells. `_association_cell` receives
  the frozen task and checks each view's realized real/dustbin fields, minimum real mass,
  fixed-point residual, and candidate violation before advancing to the next view.
- Confirmed the unsupported finite-penalty UOT balance claim is removed rather than renamed. The
  task has no `dustbin_capacity_balance_tolerance`, the tautological two-summation expression is
  gone, and the remaining wording is bounded to capacity-weighted finite-penalty transport.
- Confirmed the primary reconstruction paths now derive `heldout_fit_access_count` from realized
  optimized/reporting indices, and `_synthetic_decisions` requires the metric without a zero
  default. The finding below concerns the two additional fits performed by the independent-half
  path.
- Ran
  `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_probabilistic_field_experiment_protocol.py tests/test_probabilistic_field_pipeline.py tests/test_experiment_contract.py tests/test_field_refit.py`;
  all 50 focused outcome-free tests passed. Ran
  `.venv/bin/python scripts/check_agent_workflow.py`; it returned `OK` with the reviewer turn
  active.
- Confirmed the canonical run root remains absent. I did not initialize a run or invoke any
  official-seed or calibrated worker while inspecting the plan.

## Findings

The v3 protocol is rejected before outcome access. Two hard-gate paths remain incomplete. Either
repair changes the reviewed source/protocol surface and requires refreshed source and protocol
digests plus another prospective review.

1. **Held-out isolation is still not measured for the two independent-half fits.**
   `run_probabilistic_field_pipeline` performs three fits when independent-half validation is
   enabled: the full reconstruction and two reconstructions whose training sets are alternating
   halves. It returns the latter as `half_reconstructions`. Both `_calibrated_worker` and
   `_pipeline_synthetic_cell` discard `half_reconstructions` and call
   `_heldout_fit_access_count` only on `pipeline.reconstruction` against the original full
   `SceneFits`. Consequently the recorded zero proves only that the full fit excluded the
   original held-out cameras; it does not measure whether either half fit used its complementary
   training half or an original held-out camera. The pipeline unit test demonstrates the intended
   split on its fixture, but it is not a per-cell fail-closed measurement of the realized official
   fits. Retain both half reconstructions and validate each one's realized optimized and reported
   indices against its exact half-specific train/held-out partition before stability is accepted
   or any cell result is serialized. The cell metric should account for all fits it executed, and
   a focused test should inject leakage into either half and prove a structured cell failure.

2. **The calibrated all-candidate arm does not measure or enforce the frozen candidate gate.**
   Synthetic association cells compute `mass[~candidate].sum()` and gate it per view. In contrast,
   `_association_invariants` exposes only plan count, finite/non-negative fields, minimum real
   mass, and fixed-point residual. `_enforce_result_invariants` therefore cannot check
   `candidate_mass_tolerance`, and the calibrated summary contains no candidate-violation receipt.
   `CorrespondencePlan` also does not retain the candidate mask needed to audit the final plans.
   Thus an `all_candidate_mechanisms` calibrated cell can pass hard invariants and serialize
   without measuring the candidate-gate condition named by the frozen invariant and stopping
   rule. Carry or reproducibly reconstruct each final plan's candidate mask, emit the maximum
   disallowed real mass, and enforce the frozen tolerance before serialization, with a focused
   violation test. If candidate exclusion is intended to be synthetic-only, narrow the protocol
   explicitly and submit that new digest for review; exact zeros by construction are not the
   machine-checkable calibrated receipt frozen here.

The remaining v2 defects are resolved sufficiently: UOT association failures are checked inside
each synthetic view before later work, the primary-fit held-out metric is realized and required
without a default, and no balanced-marginal or hard-capacity conclusion is inferred from the
finite-penalty solver.

## Protected Actions Not Taken

I did not initialize or execute the canonical run, invoke an official-seed synthetic cell or
calibrated worker, inspect or create `runs/20260805_probabilistic_field_pipeline_mixed`, access any
result or rehearsal outcome, render a report, launch a viewer, update the task/index/task state,
or make a quantitative interpretation. Outcome access remained `none` throughout.
