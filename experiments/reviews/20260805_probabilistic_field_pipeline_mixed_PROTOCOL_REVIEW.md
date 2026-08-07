# Prospective Protocol Review

- Task ID: `20260805_probabilistic_field_pipeline_mixed`
- Protocol SHA-256: `ae3b9cbb11157480f33c1520244582409c0f09cd7cad7f6637cde161e8811044`
- Source-tree SHA-256: `0684bc11ac3d4b5057bf18f81db842440f790ffdc4ce6395c1a44338fa1840b9`
- Reviewer: `Codex-probabilistic-field-protocol-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

This was an outcome-blind v5 prospective review of the exact task, source binding, eleven sealed
Gaussian2D datasets, 549-cell plan, execution chronology, failure behavior, metrics, gates,
resource protocol, and claim boundary. I read the latest v5 handoff and all four preserved
rejection artifacts before independently checking the frozen v5 bytes. The focused question was
whether the sole v4 blocker is closed: for calibrated seed `80501`, the primary reconstruction and
both association-bearing independent-half reconstructions must pass every hard result invariant
before any result or stability value can be serialized.

If executed successfully, this protocol may provide deterministic synthetic mechanism evidence
and development-only operability/utility observations for native-control and all-candidate lifting
over a deterministic 512-component-per-view proxy of the eleven sealed fields. It cannot establish
complete-field fidelity, source-RGB reconstruction quality, true globally coupled multi-marginal
OT, GPS-Gaussian reproduction, GPU or real-time performance, cross-scene generality, a production
default, or reconstruction accuracy from independent-half agreement.

## Checks

- Recomputed the review digest with
  `.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260805_probabilistic_field_pipeline_mixed.json`.
  It exactly matched
  `ae3b9cbb11157480f33c1520244582409c0f09cd7cad7f6637cde161e8811044`.
- Recomputed the length-prefixed binding over `scripts/experiment_contract.py`, the task driver,
  and every `src/rtgs/**/*.py` file. All 102 bound files produced
  `0684bc11ac3d4b5057bf18f81db842440f790ffdc4ce6395c1a44338fa1840b9`, exactly matching the task.
- Ran contract and sealed-data validation; both returned `OK`. Reconfirmed all and only the eleven
  `dataset/**/gaussians2d*/manifest.json` compact bundles are included. The seal contains 309 files
  totaling 204,306,829 bytes: 296 `.rtgsv` files, eleven compact manifests, and two calibration
  files, with no image file. Each frozen train/held-out split is disjoint and exactly partitions
  its compact manifest.
- Inspected the outcome-free expanded plan. It contains 549 unique producer cells: 324 exact-shape,
  60 association, 81 support-mask, 6 topology, 6 schedule, 6 independent-half, and 66 calibrated
  cells. Schedule cells use five cleanup iterations. The calibrated matrix is eleven datasets by
  seeds `80501`, `80502`, and `80503` by arms `native_controls` and
  `all_candidate_mechanisms`.
- Rechecked the frozen command, CPU-only resource scope, one discarded calibrated warmup, fresh
  single-thread worker isolation, retained raw repeats/failure receipts, image/import guards,
  held-out fit-access prohibition, stopping rules, and development-only claim boundary. The task
  remains `draft`, its distinct owner is `Codex-probabilistic-field-driver`, review is pending, and
  no blocker is recorded; approval therefore precedes the owner's administrative transition to
  `ready` and any `init-run` action.
- Traced `_enforce_pipeline_result_invariants`. It builds one result set from the primary
  reconstruction plus both realized half reconstructions and invokes the complete
  `_enforce_result_invariants` gate separately on every member. For association-bearing fits that
  gate requires a plan, finite transport evidence, minimum real mass, fixed-point residual within
  tolerance, and candidate-gate mass within tolerance, in addition to source-projection and split
  conservation checks.
- Independently exercised the v4 counterexample surface with outcome-free in-memory surrogate
  results. Off-candidate real mass in the second half raised `candidate gate`; insufficient real
  mass in the first half raised `transport real mass`; an excessive second-half residual raised
  `transport fixed point`; and non-finite first-half evidence raised `transport non-finite`. Every
  failure occurred inside `_enforce_pipeline_result_invariants`; none reached result serialization.
- Exercised a valid three-fit bundle with deliberately different per-fit evidence. Aggregation
  selected the worst maximum for source, covariance, split, fixed-point, and candidate evidence,
  the minimum real mass across fits, the conjunction of transport finiteness, and the sum of plan
  counts. It reported `hard_invariant_checked_fit_count=3`.
- Confirmed calibrated seed `80501` first validates exact camera access for the primary and both
  halves, then calls `_enforce_pipeline_result_invariants` with association required for the
  all-candidate arm. The worker changes phase to `serialization` only after the helper returns.
  Thus either half's hard failure raises before model files, stability payload, summary, or
  canonical cell publication; the structured worker-failure path remains available.
- Confirmed the same multi-fit helper is used on the synthetic independent-half path and that the
  summary/report schema retains the aggregate invariant evidence and checked-fit count.
- Ran
  `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_fiber_correspondence.py tests/test_probabilistic_field_experiment_protocol.py tests/test_probabilistic_field_pipeline.py tests/test_experiment_contract.py tests/test_field_refit.py`;
  all 69 focused outcome-free tests passed. The task/workflow checker also passed.
- Ran `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh`. Ruff, format, the complete non-slow CPU suite,
  docs sync, ARA, script layout, agent workflow, and experiment-contract checks all passed. Only
  the two already documented PyTorch warnings were emitted.
- Recomputed both digests after verification. The four rejected review artifacts remained
  preserved, and the canonical run root and canonical review path remained absent before this
  artifact was written.

## Findings

The exact protocol and source binding above are **approved** for execution. The sole v4 blocker is
closed: every association-bearing fit that contributes to calibrated independent-half stability
is fail-closed under the complete hard-invariant set before serialization, and the published
evidence represents all three checked fits with conservative aggregation. No concrete protocol
blocker remains.

This approval authorizes only the owner's normal lifecycle recording of this exact review,
transition to `ready`, and subsequent guarded initialization/execution. Any change to a
digest-bearing task field or bound source byte invalidates this approval and requires a new
prospective review. Approval establishes fitness to execute; it does not establish a successful
run, favorable metric, method superiority, or any scientific outcome.

## Protected Actions Not Taken

I did not invoke `init-run`, execute the canonical coordinator or a calibrated worker, run an
official-seed result cell, inspect or create `runs/20260805_probabilistic_field_pipeline_mixed`,
access any rehearsal or protected outcome, render a report, create or open an index/viewer, launch
the orbit viewer, update the task/index/current-task state, or interpret quantitative results.
Outcome Access remained `none` throughout. Checks were limited to frozen protocol/source/input
metadata, outcome-free plan inspection, static source review, in-memory invariant counterexamples,
and outcome-free tests.
