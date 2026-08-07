# Prospective Protocol Review

- Task ID: `20260805_probabilistic_field_pipeline_mixed`
- Protocol SHA-256: `d8b4fc34e42059bcd57c5642ba72b6f2d50862a5cd8a5e76035e5db48d3f7421`
- Source-tree SHA-256: `b0208dc7b4364f3f64b0b4ce0bfde64ea8703522f760f8ce08a466ffd878be53`
- Reviewer: `Codex-probabilistic-field-protocol-reviewer`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

This was an outcome-blind v4 review of the exact development protocol, source binding, 549-cell
plan, eleven sealed Gaussian2D datasets, execution chronology, camera-partition receipts, and
production candidate-mask evidence. I read the latest handoff and all three preserved rejection
artifacts before checking the frozen v4 bytes independently.

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
  `d8b4fc34e42059bcd57c5642ba72b6f2d50862a5cd8a5e76035e5db48d3f7421`.
- Recomputed the length-prefixed binding over the task driver,
  `scripts/experiment_contract.py`, and every `src/rtgs/**/*.py` file. All 102 bound files produce
  `b0208dc7b4364f3f64b0b4ce0bfde64ea8703522f760f8ce08a466ffd878be53`, exactly matching the task.
- Ran `.venv/bin/python scripts/experiment_contract.py validate` and
  `.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260805_probabilistic_field_pipeline_mixed.json`;
  both returned `OK`.
- Inspected the outcome-free plan. It still contains 549 unique cells with the frozen stage counts:
  324 shape, 60 association, 81 mask, 6 topology, 6 schedule, 6 independent-half, and 66
  calibrated. The schedule cells use five cleanup iterations, and the calibrated matrix is
  eleven datasets by three seeds by two arms.
- Reconfirmed that all and only the eleven `dataset/**/gaussians2d*/manifest.json` bundles are in
  the task. The seal contains 309 files totaling 204,306,829 bytes: 296 compact `.rtgsv` files,
  eleven compact manifests, and two calibration files, with no image file. Every dataset's frozen
  train and held-out identifiers are disjoint and exactly partition its compact manifest.
- Confirmed the camera-partition repair. `run_probabilistic_field_pipeline` constructs each
  half-specific `SceneFits`, validates the primary and both realized half results before computing
  stability, and raises on the injected second-half leak. The driver independently checks that
  the two declared halves are disjoint and cover the original train set, validates each result
  against its exact train/reporting complement, sums realized access, and emits a checked-fit
  count of three.
- Traced the candidate mask through both plan constructors, `_make_plan`, `_scatter_track_plan`,
  `_detached_plan`, the empty-view path, the final E-step plans, and the primary calibrated result.
  `_association_invariants` reports `candidate_gate_violation_mass_max`, treats a missing mask as
  invalid evidence, and `_enforce_result_invariants` applies the frozen tolerance before the
  worker enters serialization. The injected primary-result off-gate mass raises as intended.
- Ran
  `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_fiber_correspondence.py tests/test_probabilistic_field_experiment_protocol.py tests/test_probabilistic_field_pipeline.py tests/test_experiment_contract.py tests/test_field_refit.py`;
  all 69 focused outcome-free tests passed.
- Ran `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh`. Ruff, format, the complete non-slow CPU suite,
  docs sync, ARA, script layout, agent workflow, and experiment-contract checks all passed. Only
  the two previously documented PyTorch warnings were emitted.
- Recomputed both digests after verification and confirmed the canonical run root and canonical
  review path remained absent before this artifact was written.

## Finding

The v4 protocol is rejected before outcome access for one remaining fail-closed path.

1. **The calibrated independent-half reconstructions bypass the frozen transport/candidate
   invariants.** For the frozen independent-half seed, an `all_candidate_mechanisms` calibrated
   cell runs the same association-enabled `FieldLiftConfig` three times: primary, first half, and
   second half. `_pipeline_fit_access_metrics` now validates the camera partition of all three,
   but `_calibrated_worker` calls `_enforce_result_invariants` only on the primary `result`.
   `run_probabilistic_field_pipeline` likewise validates camera partitions only; it does not gate
   candidate mass, minimum real mass, fixed-point residual, or the other experiment-level hard
   result invariants for either half. The half models are then used to compute the stability value
   that the worker serializes. An outcome-free injected counterexample confirmed the gap: all
   three camera receipts passed with `heldout_fit_access_count=0` and
   `heldout_fit_checked_fit_count=3`, the primary candidate violation was zero, and the second
   half retained `0.25` real mass outside its candidate mask without any invoked worker gate.
   This contradicts the frozen stopping rule to stop a cell on a candidate-gate or transport
   violation. Apply every applicable hard result invariant to the primary and both realized half
   reconstructions before accepting or serializing stability, aggregate the candidate violation
   across every checked association plan, and add an injected second-half off-gate regression.

The two v3 findings are otherwise repaired on the paths they now cover: every realized fit has an
exact camera-partition receipt, and candidate masks reach and gate the primary calibrated result.
Extending the same hard-result check to the two calibrated half reconstructions changes the bound
source and requires refreshed source/protocol digests plus another prospective review.

## Protected Actions Not Taken

I did not initialize or execute the canonical run, invoke an official-seed synthetic cell or
calibrated worker, inspect or create `runs/20260805_probabilistic_field_pipeline_mixed`, access any
result or rehearsal outcome, render a report, launch a viewer, update the task/index/task state,
or make a quantitative interpretation. Outcome access remained `none` throughout.
