# Prospective Protocol Review

- Task ID: `20260805_probabilistic_field_pipeline_association_rollback_mixed`
- Protocol SHA-256: `e57d58112fd6f95467e8ddacdb4daad7fc9d83ed48b8b9f336a32b1966a92e87`
- Source-tree SHA-256: `b17dec48edf2f07469dc9f6b197d062e4c0ee59698a05772c3674ad1fdf9b2eb`
- Successor-driver SHA-256:
  `9a53815e2e0f17c2b40c9c67295c319eec4fff163541012804438717d5801bff`
- Pinned base-driver SHA-256:
  `9d453b967b09005b63d3bef6aac48b817ac841ea2b7faa3593d4f980e4310169`
- Reviewer: `Codex-probabilistic-field-protocol-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

This was a distinct, prospective, outcome-blind review of the transactional-association-rollback
successor. I reviewed the exact task and source bindings, immutable predecessor failure chronology,
private-clone rollback behavior, committed-versus-rolled-back provenance, unchanged required-
transport hard gates, structured zero-success continuation, primary/independent-half coverage,
native-arm isolation, exact plan preservation, subprocess identity, and draft-to-ready lifecycle.

If executed successfully, this remains development-only evidence over deterministic
512-component-per-view proxies of eleven sealed Gaussian2D fields. It may measure the frozen
synthetic mechanisms and calibrated compact-field operability, quality, convergence, resource,
and presentation behavior. It cannot establish complete-field fidelity, source-RGB
reconstruction quality, spatial resolution, true globally coupled multi-marginal OT,
GPS-Gaussian reproduction, GPU or real-time performance, cross-scene generality,
production-default suitability, or accuracy from independent-half agreement.

## Checks

- Recomputed the protocol digest independently and with `scripts/experiment_contract.py`; both
  produced `e57d58112fd6f95467e8ddacdb4daad7fc9d83ed48b8b9f336a32b1966a92e87`.
  The task is `draft`, its review fields are pending, `depends_on` and `blockers` are empty, and
  neither the canonical review path nor successor run root existed before this artifact.
- Recomputed the length-prefixed binding over `scripts/experiment_contract.py`, the successor
  wrapper, and every `src/rtgs/**/*.py` file. All 102 files produced
  `b17dec48edf2f07469dc9f6b197d062e4c0ee59698a05772c3674ad1fdf9b2eb`.
- Recomputed the successor byte digest as
  `9a53815e2e0f17c2b40c9c67295c319eec4fff163541012804438717d5801bff` and the
  pinned support-fallback base-driver digest as
  `9d453b967b09005b63d3bef6aac48b817ac841ea2b7faa3593d4f980e4310169`.
  The wrapper verifies the latter before import and requires the exact base path, algorithm, and
  hash in the task contract.
- Recomputed the data-seal digest as
  `20e719d89628375c515db94102abf6e5018dbd6d686d0633235407fad5c7deb6`.
  Repository contract validation and task data validation both returned `OK`.
- Recomputed the 549-cell-list digest as
  `1af09dabc5de411ff09acdef30efa5da17e010f6fcad2b450b0dc08d31f005fc`
  and the full successor plan-payload digest as
  `dc1f350715518c2d78a6e066c7173a12e9b038ba290903f09800d88322cd34fa`.
  Every cell dictionary is exactly equal to the approved forward-AABB plan: 324 exact-shape, 60
  association, 81 mask, six topology, six schedule, six independent-half, and 66 calibrated
  cells.
- Compared realized calibrated configurations against the approved forward-AABB predecessor.
  Native configurations are exactly equal. Candidate configurations differ only at
  `association.failure_policy`, from `raise` to `rollback`; placement, masks, topology, refit,
  caps, seeds, thresholds, and all other fields are unchanged.
- Confirmed `InverseProjectionFiber.subset` reconstructs and copies every trainable parameter and
  registered buffer. A reviewer-injected exact partial-M-step `RuntimeError` mutated the working
  fiber before raising; rollback returned the original fiber, and every original fiber state,
  field mass, color, opacity, support, and lineage tensor remained bitwise unchanged. The bound
  correspondence implementation also clones supplied track capacities before use.
- Confirmed both caught types produce one typed failure string and a `rolled_back` placement while
  uncaught types and `failure_policy=raise` remain outside continuation. Empty failure text is
  rejected by the successor's provenance validator rather than admitted as a continuable hard
  failure.
- Exercised exact committed and rolled-back records, then attacked missing/mismatched statuses,
  absent failures, empty failures, wrong exception types, wrong association presence, mismatched
  result/placement failure text, boolean status aliases, and forbidden failure keys on committed
  results. All thirteen malformed states failed closed; native results remained outside the new
  candidate-only diagnostic contract.
- Exercised primary and both independent-half dispatch through the patched invariant seam. Exact
  committed fits passed the new completion check, a malformed half aborted with an association-
  diagnostic error, and a rolled-back primary or half reached the unchanged exact hard failure:
  `hard invariant violation: transport plan missing, transport real mass, transport fixed point,
  candidate gate`.
- Confirmed this hard failure occurs before success serialization. The pinned base producer records
  a failed cell only as `calibrated_cell_success=0`, excludes it from quality/runtime medians and
  conditional metrics, never substitutes another arm, and labels preserved rejected models as
  presentation-only. Malformed rollback provenance does not begin with `hard invariant violation:`
  and therefore aborts the root instead of entering structured continuation.
- Mutated task ID, driver identity, run command, source hash/algorithm/key set, base hash/key set,
  anchor outcome inputs, rollback outcome inputs, caught exception types, and pipeline failure
  policy. All twelve mutations were rejected. All synthetic aggregate/cell, calibrated worker,
  and warmup commands resolve to the successor driver, task, and run root after base-module identity
  patching.
- Re-ran the administrative transition counterexample. Converting an in-memory task copy from
  `draft`/pending to `ready`/approved leaves the protocol digest unchanged; adding a blocker changes
  it. The task can therefore record this exact approval without changing reviewed protocol bytes.
- Restricted predecessor access to the explicitly permitted failure chronology. Root failure,
  run-receipt, and terminal failure hashes independently remained
  `63206958f3fd963e277d8487000f9b76383ae5a8aa9120b4235a65cbd32216d4`,
  `0d1abb1b84bba0d3d72edede63cf3582e31e22a665961ef9ab25448fc97758fa`, and
  `07a2ff8e0fbe8cfcf2926a56b77ea8ac40704f509b38a53fca35b4324671396b`.
  No predecessor metric, summary, aggregate, model, report, preview, or viewer was opened.
- All eleven focused forward-AABB/rollback tests passed. The broader correspondence, protocol,
  probabilistic-pipeline, experiment-contract, refit, and field-lifter suite passed.
- Ran `./scripts/verify.sh`. Ruff, format, the complete non-slow CPU suite, docs sync, ARA, script
  layout, agent workflow, and experiment-contract checks all passed with only the two documented
  PyTorch warnings. `git diff --check` was clean.
- Recomputed every protocol/source/driver/base/cell/plan/data binding after verification. The task
  remained `draft` and pending, `blockers` remained empty, and the successor run root remained
  absent.

## Findings

The exact protocol and source bindings above are **approved** for execution. In the bound source,
association optimization mutates a copied inverse-projection fiber and commits that fiber only
after a successful return. A caught `RuntimeError` or `ValueError` returns the untouched placement
with exact rollback provenance; the successor validator then forces that result through the
existing required-transport gate. Consequently rollback cannot become a successful cell, supply a
quality or runtime value, borrow a native result, or make a rejected model claim-eligible.

The native path is unchanged, all 549 scientific cells are unchanged, and the one realized
candidate configuration difference is the preregistered failure policy. Malformed provenance,
wrong task/source/base/command identity, and any exception outside the two narrow caught types fail
closed. No concrete prospective blocker remains.

Only after the Driver records this exact approval and transitions the task's administrative
status/review fields to `ready` may the single canonical run be initialized and the exact command
executed. Any digest-bearing task edit, bound-source change, successor-driver byte change, or
pinned-base byte change invalidates this approval. Approval establishes fitness to execute; it
does not establish successful completion, favorable metrics, method superiority, browser/report
usability, orbit-viewer availability, or any scientific outcome.

## Required Changes

None before the Driver records the exact approval and performs the administrative ready
transition. No result-producing action is authorized on different bytes.

## Optional Hardening

The bound correspondence fitter already clones field-mass track capacities immediately on entry,
so the reviewed composite transaction leaves placement state unchanged. A future source revision
could additionally clone those capacities at the outer `_run_association` boundary for defense in
depth. That revision is not required for these exact bytes and would require a new source binding
and prospective review.

## Protected Actions Not Taken

I did not edit the task into `ready`, fill its review record, initialize or execute the successor
run, invoke an official synthetic or calibrated worker, access a successor outcome, inspect a
predecessor metric or model, attach or read result/audit payloads, render or open an official index
page, launch an orbit viewer, change a default, make a scientific claim, commit, push, or publish.
Reviewer-authored counterexamples were in-memory deterministic fixtures and created no retained
result. Outcome Access remained `none` throughout.
