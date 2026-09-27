# Current Task

## Title

Active silhouette-hull floater relocation for field-only distillation

## Task ID

RTGS-026

## Role Assignment

- Driver: Claude-Code-Opus-5.5-driver
- Reviewer: Claude-Code-Fable-5.1-reviewer
- Turn: driver

## Mode

Implement

## Risk

Protected

## Maturity

- Target: Calibrated
- Reached: CPU-contracted

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

experiments/tasks/20260927_silhouette_relocation_stage_frame00008.json

## Current Evidence

RTGS-025 (C49, C50) and its audited run; `rtgs.data.field_targets`.

## Minimal Plan

1. Implement and test the relocation mechanism and Trainer hook.
2. Register the protocol and driver; obtain Fable 5.1 prospective review.
3. Execute once, audit, report, and, on a pass, train the all-data model.

## Status

In progress

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
