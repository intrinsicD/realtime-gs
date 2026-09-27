# Current Task

## Title

Active silhouette-hull floater relocation for field-only distillation

## Task ID

RTGS-026

## Role Assignment

- Driver: Claude-Code-Opus-5.5-driver
- Reviewer: Claude-Code-Fable-5.1-reviewer
- Turn: human

## Mode

Implement

## Risk

Protected

## Maturity

- Target: Calibrated
- Reached: Calibrated

## Goal

Use the masks of all calibrated views to identify 3D Gaussians that project outside a silhouette
and actively move them into the mask-consistent visual hull during field-only distillation; test
novel-view colour on two held-out RGB-field views; if the method works, retrain on all data.

## Motivation

RTGS-025 showed field-only distillation close to a photograph reference that itself missed its
floor, with passive mask losses only. The user proposed active, all-mask floater relocation on
2026-09-27 (decisions: a Gaussian is a floater if any view's dilated mask rejects it; all masks,
including those of the two held-out RGB views, define the hull).

## Success Criteria

- An opt-in, CPU-tested relocation mechanism (hull membership, nearest-hull target, optimizer
  state reset) behind a generic Trainer parameter-step hook, with no default change.
- A registered, independently reviewed protocol comparing relocation against the RTGS-025
  masked objective with paired seeds and a relative decision rule.
- One audited calibrated run with report, viewer receipt and bundle gates; ARA/docs updated.
- If the frozen rule passes, one declared all-data production model (not evidence).

## Constraints

CPU-first imports; relocation opt-in only. Held-out RGB fields and photographs are evaluation-only;
held-out masks are declared training inputs for the hull, so held-out alpha metrics are
in-sample. Frame 00008 is outcome-exposed. The `rtgs-025-main-path` merge remains a human decision.

## Non-Goals

Changing Trainer defaults, footprint/scale clamping, new frames, or promoting a default.

## Selected Skills

- `rtgs-core`
- `rtgs-task-workflow`
- `rtgs-experiment`
- `realtime-gs-results-audit`
- `rtgs-review`
- `rtgs-docs-sync`
- `rtgs-verify`

## Experiment Contract

experiments/tasks/20260927_silhouette_relocation_index_decode_stage_frame00008.json

## Current Evidence

RTGS-025 (C49, C50) and its audited run; `rtgs.data.field_targets`.

## Minimal Plan

1. Implement and test the relocation mechanism and Trainer hook.
2. Register the protocol and driver; obtain Fable 5.1 prospective review.
3. Execute once, audit, report, and, on a pass, train the all-data model.

## Status

Blocked on human decision

## Human Decisions

### Question
How should floaters be identified and which masks define the hull?
### Options
Any single rejecting view vs at least two; all masks vs excluding the held-out views' masks.
### Recommendation
Any single rejecting view with a small mask dilation; all masks, with held-out alpha metrics
declared in-sample.
### Decision
User chose both recommendations.
### Date
2026-09-27

### Question
RTGS-026 closed its audited screen with H1 inconclusive. Merge `rtgs-025-main-path` into `main`,
and which follow-up (if any) should be registered next?
### Options
Merge and register a colour-quality follow-up (budget/resolution, hull-initialized no-relocation
arm); merge only; hold the branch.
### Recommendation
Merge; next target colour quality inside the mask rather than floaters.
### Decision
Pending.
### Date
2026-09-27

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### Handoff (2026-09-27, mechanism and protocol for prospective review)

#### Objective
Obtain Fable 5.1 prospective review of the RTGS-026 relocation protocol.

#### Reviewed state
Commit `d41eb5e` (mechanism) plus the draft task, driver, report module and tests on branch
`rtgs-025-main-path`; digest `57370d132b561b48b5bc95b7f7c15a8bfd8b3b01c129e55d200f209ce084f76f`,
source binding 120 files `263b011ac4a0222f60426a5b416c0705fbfc28a9a34bd7a49b7d2ee4e8598bc2`.

#### Changes
Opt-in `Trainer.train(parameter_step_callback=...)`, `rtgs.optim.silhouette_relocation`,
RTGS-026 driver/report/tests, protocol and data seal.

#### Evidence
CPU tests pass (mechanism 9, protocol 5, plus existing trainer tests); `verify.sh` exit 0 at the
mechanism commit. Non-protocol GPU smoke exercised every phase; relocation moved most random start
points into the hull at the first event and reduced the final hull-rejected fraction.

#### Assumptions
Held-out masks as hull inputs are the user's explicit design; held-out colour stays evaluation-only.

#### Uncertainties
The first relocation event doubles as hull-guided re-initialization (disclosed in the boundary).

#### Review Focus
Leakage (held-out views contribute packed alpha only), relocation correctness and optimizer-state
handling, gate definitions, production gating.

#### Protected actions not taken
No init-run, protected execution or held-out colour access.

#### Recommended Next Action
Fable 5.1 prospective review of the exact digest.

### Review (2026-09-27, Fable 5.1 prospective protocol review V1)

#### Verdict
Revision required

#### Self-reviewed
No

#### Correctness
Rejected digest `57370d13...` for B1 (production phase outside the bundle lifecycle) and B2
(coordinator re-entry); scientific core, leakage boundary and mechanism accepted. Verbatim:
`experiments/reviews/20260927_silhouette_relocation_stage_frame00008_PROTOCOL_REVIEW_V1_REJECTED.md`.

#### Evidence Quality
Digest recomputed, focused CPU tests run; no outcome access.

#### Simplicity
Bounded lifecycle fixes only.

#### Missing Cases
Production sequencing; consumed-root refusal.

#### Required Changes
B1, B2.

#### Optional Improvements
R1-R10 (R1 found the inert gsplat opacity reset).

### Handoff (2026-09-27, revision 1 for review round 2)

#### Objective
Second prospective review of the revised digest.

#### Reviewed state
Digest `f53d1074c29c8ddc8614f9872a05556014ab4b99ec131d4d9ef83fb5c814f2ad`, binding 120 files
`f7877b92c3655feb1efde420c6ca37e27d6abe2f711308ba1323c11a6bcf0e8f`.

#### Changes
See `experiments/reviews/20260927_silhouette_relocation_stage_frame00008_DRIVER_RESPONSE_R1.md`.

#### Evidence
22 focused CPU tests pass; repeated GPU smoke passed.

#### Assumptions
Production after the audited gates (option b) matches the user's "if it works" instruction.

#### Uncertainties
None new beyond the disclosed ones.

#### Review Focus
B1/B2 fixes, adopted optional changes, no design drift.

#### Protected actions not taken
No init-run or protected execution.

#### Recommended Next Action
Fable 5.1 review round 2; a second rejection escalates to the user.

### Review (2026-09-27, Fable 5.1 prospective protocol review round 2)

#### Verdict
Accepted

#### Self-reviewed
No

#### Correctness
Approved digest `f53d1074c29c8ddc8614f9872a05556014ab4b99ec131d4d9ef83fb5c814f2ad`: B1/B2
resolved, optional changes defect-free, no design drift. Readiness only, not results. Verbatim:
`experiments/reviews/20260927_silhouette_relocation_stage_frame00008_PROTOCOL_REVIEW.md`.

#### Evidence Quality
Digest and 22 CPU tests reproduced; aggregate pinned by git object ids (`src/rtgs` tree
`9293a19a...`) and enforced by `init-run`.

#### Simplicity
No further changes.

#### Missing Cases
Residual execution conditions 1-6 (single invocation, production sequence, disclosures).

#### Required Changes
None.

#### Optional Improvements
Validate `dilation_px >= 0` in `SilhouetteHull`; broaden the non-position-row test (later task).

### Handoff (2026-09-27, predecessor run failed closed; index-decoding retry for review)

#### Objective
Replace the consumed predecessor run with a new task that differs only in exact CPU-index decoding.

#### Reviewed state
New task `experiments/tasks/20260927_silhouette_relocation_index_decode_stage_frame00008.json`, digest
`27f7b23ac0ad4be45e22fc2f8afbd17bb223d2d1680e70dc2a336b288000f815`, source binding 120 files
`8dacd07f7eb57b3d0343f1b7107cd3e2aaa7f4901e726771f5b29b96c05bb365`.

#### Changes
The approved predecessor `20260927_silhouette_relocation_stage_frame00008` ran once (lock at
`f88f3d0`, preflight passed) and failed closed in `prepare` at its frozen CUDA-vs-CPU decoder
parity gate: 1.889e-4 > 2e-5 at 1 of 512 sites of training view C0026 (normalized blend, weight
sum ~4e-7). No initialization, cell, held-out access or outcome. Receipts:
`ara/evidence/tables/20260927_silhouette_relocation_failed_run/`. The retry copies the approved
driver/report/tests with decoding switched to the exact CPU index (parity vs reference only);
hypothesis, thresholds, seeds, split, arms, metrics and budget are unchanged.

#### Evidence
17 focused CPU tests pass; a non-protocol prepare+initialize smoke over the frozen split passed
(all 24 training views, index parity <= 1.8e-7, 21391 hull voxels).

#### Assumptions
Seeds may be reused because the predecessor produced no outcome.

#### Uncertainties
The CUDA query kernel's near-zero-weight normalization behaviour remains uninvestigated beyond this site.

#### Review Focus
That the only substantive change is the decoder backend and parity rule, and that the failure is disclosed.

#### Protected actions not taken
No init-run of the retry, no protected execution.

#### Recommended Next Action
Fable 5.1 prospective review of the retry digest.

### Review (2026-09-27, Fable 5.1 prospective review of the index-decoding retry)

#### Verdict
Accepted

#### Self-reviewed
No

#### Correctness
Approved digest `27f7b23ac0ad4be45e22fc2f8afbd17bb223d2d1680e70dc2a336b288000f815`; the only
substantive change from the approved predecessor is exact CPU-index decoding; failure disclosed;
seed reuse acceptable. Verbatim: `experiments/reviews/20260927_silhouette_relocation_index_decode_stage_frame00008_PROTOCOL_REVIEW.md`.

#### Evidence Quality
Byte-level diffs against the predecessor, digest and focused tests reproduced.

#### Simplicity
Minimal retry.

#### Missing Cases
Residual conditions 1-6 carried over (single invocation, production sequence, disclosures).

#### Required Changes
None.

#### Optional Improvements
Investigate the CUDA query kernel's near-zero-weight normalization in a later task.

### Handoff (2026-09-27, retry run completed; results audit pending)

#### Objective
Execute the approved retry once and hand the producer records to an independent results audit.

#### Reviewed state
Run `runs/20260927_silhouette_relocation_index_decode_stage_frame00008/` locked at `70e4bc6` (clean, non-development); one `run` invocation, exit 0,
12/12 cells; GPU shared with an unrelated process (launch context in `.scratch/20260927_silhouette_relocation_index_decode_stage_frame00008/`).

#### Changes
None to source or protocol.

#### Evidence
Producer decision: H1 inconclusive (seed 9361 passes at +0.228 dB; seeds 9362/9363 at -0.016 and
-0.023 dB), so no production model is trained. Relocation lowered the final hull-rejected
Gaussian fraction from about 5.5 percent to about 0.2 percent and in-sample outside-mask alpha by
about a quarter; the first event moved about 19400 of 20000 start points. Pre-audit values only.

#### Assumptions
Held-out alpha metrics are in-sample by design.

#### Uncertainties
Visual effect on floaters/halos; whether a longer budget or hull-guided initialization changes H1.

#### Review Focus
Results audit per the approved conditions; no post-hoc reinterpretation of the per-seed rule.

#### Protected actions not taken
No production phase (gate not passed), no audit dispatch, render, viewer receipt or commit of
the RESULT. The audit would read dome-derived previews; user authorization is required.

#### Recommended Next Action
Ask the user to authorize the Fable 5.1 results audit including dome-derived previews.

### Review (2026-09-27, independent results audit of the retry)

#### Verdict
Accepted with follow-up

#### Self-reviewed
No

#### Correctness
Claude-Code-Fable-5.1-reviewer (claude-fable-5-1, effort max) returned `accepted_with_limits`
after the user authorized reading 13 dome-derived previews; all numbers reproduce; H1
inconclusive; no production model; no producer statement retired. Verbatim:
`benchmarks/results/20260927_silhouette_relocation_index_decode_stage_frame00008_AUDIT.md/json`.

#### Evidence Quality
Receipts, bindings, boundaries, relocation events and all 12 cells checked; downscaled sheets only.

#### Simplicity
No rerun or source change.

#### Missing Cases
Hull-initialized no-relocation arm; masks excluded from the hull for out-of-sample floater tests.

#### Required Changes
Carry four disclosures into EXPERIMENTS.md/ARA and label interior alpha in-sample (done).

#### Optional Improvements
Future colour-quality task; CUDA query kernel near-zero-weight normalization.

### Handoff (2026-09-27, bundle complete)

#### Objective
Close the RTGS-026 bundle after the independent audit.

#### Reviewed state
Run `runs/20260927_silhouette_relocation_index_decode_stage_frame00008/` with AUDIT records, second render and viewer receipt; tracked receipts in
`ara/evidence/tables/20260927_silhouette_relocation_final_handoff/`.

#### Changes
AUDIT persisted verbatim; HTTP mirrors and favicon in the run root; headless viewer receipt;
EXPERIMENTS.md entry; N249, O181, C51.

#### Evidence
`check-run` OK; `check_results_bundle.py` OK; 517 report targets HTTP 200.

#### Assumptions
Headless SwiftShader viewer receipt is acceptable (as in RTGS-016/025).

#### Uncertainties
Whether relocation helps with a longer budget or other frames.

#### Review Focus
Human decision on merge and the next task.

#### Protected actions not taken
No production phase (gate inconclusive), no default change, no merge.

#### Recommended Next Action
User decides merge and follow-up.
