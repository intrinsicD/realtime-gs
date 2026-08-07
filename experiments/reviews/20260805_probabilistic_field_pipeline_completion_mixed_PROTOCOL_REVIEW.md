# Prospective Protocol Review

- Task ID: `20260805_probabilistic_field_pipeline_completion_mixed`
- Protocol SHA-256: `1b0382109dfaaba43807daf59afac6e646c5cfb6f062e428be50ae3544b257dd`
- Source-tree SHA-256: `b6982acb92923ed38f41d0fe05f414cf3bb1d2da816592a8b0f340a647ff26b0`
- Reviewer: `Codex-probabilistic-field-protocol-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

This was an outcome-blind prospective review of the failure-tolerant completion task. I reviewed
the exact task/source binding, the immutable scientific matrix inherited from the failed input
retry, the restricted continuation boundary, failure/input/resource receipt validation,
failure-only aggregation and rendering, exact terminal accounting, root-failure phase
provenance, run-root isolation, and outcome-free verification evidence. I read the final handoff
and independently checked the frozen bytes before creating this artifact.

If executed successfully, this task remains development-only evidence over a deterministic
512-component-per-view proxy of the eleven sealed Gaussian2D fields. It may measure synthetic
mechanism behavior and calibrated operability, quality, convergence, resource use, and
presentation artifacts under the frozen controls. It cannot establish complete-field fidelity,
source-RGB reconstruction quality, spatial resolution, true globally coupled multi-marginal OT,
GPS-Gaussian reproduction, GPU or real-time performance, cross-scene generality, production-
default suitability, or accuracy from independent-half agreement.

## Checks

- Recomputed the protocol digest with
  `.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260805_probabilistic_field_pipeline_completion_mixed.json`.
  It exactly matched
  `1b0382109dfaaba43807daf59afac6e646c5cfb6f062e428be50ae3544b257dd`.
- Recomputed the length-prefixed binding over `scripts/experiment_contract.py`, this task's
  driver, and every `src/rtgs/**/*.py` file. All 102 bound files produced
  `b6982acb92923ed38f41d0fe05f414cf3bb1d2da816592a8b0f340a647ff26b0`, exactly matching the
  frozen source binding.
- Ran repository contract validation and completion-task data validation; both returned `OK`.
  The unchanged data seal has SHA-256
  `20e719d89628375c515db94102abf6e5018dbd6d686d0633235407fad5c7deb6`, covers eleven datasets
  and 309 files totaling 204,306,829 bytes, and contains no RGB or mask path.
- Restricted predecessor inspection to the already failed input retry's root failure receipt,
  run receipt, and failed measured-cell failure receipt. I did not open its synthetic results,
  successful summaries, raw metric payloads, models, report, or other outcome artifacts. The
  permitted receipts establish only that seven measured cells terminated successfully before
  the eighth, `stage_00008_native_fullres` seed `80501` candidate cell, stopped in `field_fit`
  with `RuntimeError: hard invariant violation: transport real mass`.
- Compared the completion task with the immutable input retry. The question, hypothesis,
  datasets and roles, train/held-out splits, seeds, input policy, guards, stages, comparators,
  required charts, data seal, CPU protocol, synthetic generation, mechanism cells, mask test,
  pipeline, invariant gates, and decision rules are byte-equivalent. Every old metric retains
  its ID, label, unit, and direction; the five calibrated metrics are now explicitly conditional,
  and calibrated-cell success fraction is the sole new primary metric.
- Compared the complete outcome-free plans by `cell_id`. All 549 identifiers remain present;
  all 483 synthetic cells are identical. The 66 calibrated cells differ only by the five declared
  failure-accounting factors: cell-failure policy, rejected-model preservation, and the hashes of
  the continued-failure receipt, failure-reporting, and conditional-aggregate contracts. The
  canonical cell-list SHA-256 is
  `acc3ed8d2de7a271b2f95f7039d08b5b5d931d71ecf4d75a2e08bd29bae62a6b`. Stage counts remain
  324 exact-shape, 60 association, 81 support-mask, 6 topology, 6 schedule, 6 independent-half,
  and 66 calibrated cells.
- Diffed the drivers structurally. Sixty-two shared top-level definitions are unchanged. The
  changed shared definitions and new helpers are confined to calibrated failure preservation and
  validation, outcome enumeration, aggregation/history/report/viewer fallbacks, plan disclosure,
  and orchestration receipts. No synthetic generator, scientific treatment, comparator, metric
  calculation, hard invariant, threshold, optimizer, seed, or split was changed.
- Confirmed the continuation boundary is fail-closed. A cell may continue only for an exact
  `RuntimeError` message prefix in `field_fit`, with exact failure/boundary/resource key sets and
  context; clean input guard plus all three negative controls; live sealed path/byte/hash
  inventory; no external mask or held-out fit access; two complete rejected-model artifacts; no
  preservation error; single-thread CPU inventory; and complete finite timing, byte, CUDA, and
  positive-RSS evidence. The frozen receipt-contract SHA-256 is
  `65a687550a6ef11029185e5f55d0fd52420255f32c6b70ff97dbc75e22d7b283`.
- Built one exact valid failure receipt from live sealed metadata and applied 73 independent
  one-field adversarial mutations: 51 exact-key removals and 22 semantic/context, guard, input,
  artifact, resource, and timing corruptions. The valid receipt was accepted and every mutation
  was rejected as not safely continuable.
- Independently constructed all 66 exact frozen cells as valid eligible failures and passed them
  through live receipt loading and aggregate generation. Accounting was 0 successes and 66
  failures. All five conditional results were JSON null with denominator zero; the finite
  canonical metric table contained only the four synthetic metrics, calibrated success fraction,
  and five zero successful-cell counts. No failed quality, runtime, convergence, resource, or
  topology value was imputed.
- In that all-failure counterexample, fitting history contained exactly 66 zero-valued
  `calibrated_cell_success` records and 924 stage markers. Each of the three root comparison cards
  was labeled unavailable and contained the 22 frozen dataset/arm successful-cell counts, all
  zero. Resource accounting reported 66 attempts, zero successes, and 66 failures, with empty
  successful-repeat metric summaries.
- Rendered that outcome-free all-failure fixture through the canonical report code in a temporary
  root. `validate_run(..., require_index=False)` and the post-render
  `validate_run(..., require_index=True)` both returned no errors; the report produced all eleven
  child dataset pages and a 488-entry checksum manifest. Its report-server argv matched the exact
  canonical `.venv/bin/python -m http.server 8765 --directory runs/<task-id>` contract.
- Exercised exact terminal accounting independently. Sixty-five exact terminals plus a hidden
  worker staging failure counted as 65; the exact final terminal raised the count to 66; adding a
  contradictory success terminal to that cell reduced the valid count to 65. Unfrozen and hidden
  paths did not contribute.
- Exercised the actual orchestration body through both late-failure branches. A rejected 66th
  receipt before aggregate entry produced root phase `orchestration` and measured count 66; a
  failure raised from aggregate publication after all 66 success terminals produced phase
  `aggregation` and measured count 66. Both `failure.json` and `run_receipt.json` retained the
  same phase provenance.
- Ran
  `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_fiber_correspondence.py tests/test_probabilistic_field_experiment_protocol.py tests/test_probabilistic_field_pipeline.py tests/test_experiment_contract.py tests/test_field_refit.py`;
  all 82 focused tests passed.
- Ran `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh`. Ruff, format, the complete non-slow CPU test
  suite, docs sync, ARA, script layout, agent workflow, and experiment-contract checks all passed.
  Only the two documented pre-existing PyTorch warnings were emitted.
- Recomputed task, source, data, and plan bindings after verification. Immediately before this
  artifact was written, the task remained `draft` with a pending review and both the canonical
  review path and `runs/20260805_probabilistic_field_pipeline_completion_mixed` were absent.

## Findings

The exact completion protocol and source binding above are **approved** for execution. The new
policy does not weaken a scientific hard gate: every worker still stops at the gate, while only a
complete and live-validated field-fit failure becomes an eligible terminal for the independent
next cell. Failed cells contribute exactly one zero success indicator and no imputed conditional
measurement. Empty successful subsets remain schema-valid, visibly unavailable, and auditable.
Exact cell enumeration and explicit aggregate-entry state prevent staging debris, contradictory
terminals, or a late orchestration error from forging completion or phase provenance.

Four prospective defects were found and repaired in superseded completion freezes before this
approval: permissive receipt evidence, empty-success aggregation, a noncanonical report command,
and directory-count-derived terminal/phase provenance. The final frozen bytes pass the direct
counterexamples for each issue. No concrete protocol blocker remains.

Only after the owner records this exact approval and transitions the task to `ready` may the
canonical run be initialized and executed. Any digest-bearing task edit or bound-source change
invalidates this approval. Approval establishes fitness to execute; it does not establish
successful completion, favorable metrics, method superiority, report usability on the user's
browser, or any scientific outcome.

## Protected Actions Not Taken

I did not transition the task to `ready`, initialize or execute the official completion run,
create its canonical run root or lock, invoke an official synthetic or calibrated worker, inspect
any successful predecessor outcome, attach or read an official result/audit payload, render or
open an official index page, launch an orbit viewer, change a default, make a scientific claim,
commit, push, or publish. Temporary counterexamples contained only reviewer-authored synthetic
fixtures and were deleted after validation. Outcome Access remained `none` throughout.
