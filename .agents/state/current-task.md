# Current Task

## Title

Field-only distillation improvement portfolio

## Task ID

RTGS-029

## Role Assignment

- Driver: Claude-Code-Opus-5.5-driver
- Reviewer: Claude-Code-Fable-5.1-reviewer
- Turn: driver

## Mode

Validate

## Risk

Protected

## Maturity

- Target: Calibrated
- Reached: CPU-contracted

## Goal

Screen every available single lever for held-out colour quality of field-only distillation: SH
degree, opacity/scale regularization, iterations with the densification window fixed (with and
without the 8000-step learning-rate decay), training resolution, and view-count sensitivity,
against the best tested 8000-step configuration and a photograph reference.

## Motivation

RTGS-028 rejected the 30000-step package; the user asked on 2026-09-28 to "try everything"
(views: no additional calibrated photographs exist for frame 00008, so only view-count
sensitivity can be measured).

## Success Criteria

- Registered, independently reviewed protocol with per-arm relative rules and a multiplicity note.
- One audited calibrated run with report, viewer receipt and bundle gates; ARA and docs updated.

## Constraints

CPU-first imports; no default change; held-out views evaluation-only; frame 00008 is exposed.

## Non-Goals

Combining arms (a later confirmation), new data, default promotion.

## Selected Skills

- `rtgs-core`
- `rtgs-task-workflow`
- `rtgs-experiment`
- `realtime-gs-results-audit`
- `rtgs-review`
- `rtgs-docs-sync`
- `rtgs-verify`

## Experiment Contract

experiments/tasks/20260928_field_only_portfolio_stage_frame00008.json

## Current Evidence

C49, C51, C52, O182; RTGS-025-028 bundles.

## Minimal Plan

1. Driver, report, tests, protocol; non-protocol smoke.
2. Fable 5.1 prospective review; one run on a free GPU; results audit; report and gates.

## Status

In progress

## Human Decisions

### Question
Which further levers should be tested?
### Options
Regularization, SH degree, factor separation of the budget, resolution, more views.
### Recommendation
All levers as one-factor arms; view count only as a sensitivity arm (no extra views exist).
### Decision
User: "Bitte probiere alles aus was du kannst" (2026-09-28).
### Date
2026-09-28

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### Handoff (2026-09-28, protocol and driver for prospective review)

#### Objective
Fable 5.1 prospective review of the RTGS-029 portfolio protocol.

#### Reviewed state
Branch `rtgs-029-field-only-portfolio` (commit adding this entry); digest `4dd390c27ff30d7db4b5fdb4be0d59f86c849f8be1dd910930adbd756868236c`.

#### Changes
Driver/report/tests derived from the accepted RTGS-028 driver: nine conditions (base, SH 1, SH 0,
opacity/scale regularization, 30000 steps with densification stop 6000, the same with the 8000-step
means learning-rate decay, downscale-4 training, half training views, photograph reference);
evaluation renders every model at downscale 4 and box-averages to the downscale-8 grid.

#### Evidence
6 CPU tests (one CUDA-only box-render test skips on CPU); non-protocol GPU smoke of all phases and
arms passed; `./scripts/verify.sh` exit 0.

#### Assumptions
Box-averaged downscale-4 rendering is the fair common evaluation for mixed training resolutions.

#### Uncertainties
Six simultaneous comparisons; downscale-4 cells take about four times longer.

#### Review Focus
Factor isolation per arm, evaluation operator, view-subset rule, leakage, gates and multiplicity.

#### Protected actions not taken
No init-run or protected execution.

#### Recommended Next Action
Fable 5.1 prospective review.

### Review (2026-09-28, Fable 5.1 prospective protocol review V1)

#### Verdict
Revision required

#### Self-reviewed
No

#### Correctness
Rejected digest `4dd390c2...`: the primary box operator favours nb_ds4 alone because gsplat's
0.3 px^2 dilation acts in render pixels (two-operator evaluation required); five disclosures
(means-LR factors, post-8000 regime, ds4 package side effects, relative densification-window
comparison, half-view sampling) must enter the frozen protocol. Verbatim:
`experiments/reviews/20260928_field_only_portfolio_stage_frame00008_PROTOCOL_REVIEW_V1_REJECTED.md`.

#### Evidence Quality
Digest, validators and tests reproduced; published RTGS-028 outcome read by design.

#### Simplicity
No added cells.

#### Missing Cases
Operator bias for mixed training resolutions.

#### Required Changes
Two-operator evaluation and disclosures.

#### Optional Improvements
None blocking.

### Handoff (2026-09-28, revision 1 for review round 2)

#### Objective
Second prospective review of the revised digest.

#### Reviewed state
Digest `345e57a95cac3edb93fa7cc56d7a0ec43edb9d398f08da4e895583d9b32dfd68`, binding 119 files
`2848f1d6695c69680fd8a724dc8d1408d33ad92d031e5c0ae3fa061e98831b6e`.

#### Changes
See `experiments/reviews/20260928_field_only_portfolio_stage_frame00008_DRIVER_RESPONSE_R1.md`.

#### Evidence
7 protocol tests pass; validate OK; smoke evaluation produced both operators.

#### Assumptions
Point-sampled ds8 matches the RTGS-028 operator exactly.

#### Uncertainties
None new.

#### Review Focus
Two-operator implementation and gating; disclosures.

#### Protected actions not taken
No init-run or protected execution.

#### Recommended Next Action
Fable 5.1 review round 2; a second rejection escalates to the user.

### Review (2026-09-28, Fable 5.1 prospective protocol review round 2)

#### Verdict
Accepted

#### Self-reviewed
No

#### Correctness
Approved digest `345e57a95cac3edb93fa7cc56d7a0ec43edb9d398f08da4e895583d9b32dfd68` with residual
conditions (verbatim: `experiments/reviews/20260928_field_only_portfolio_stage_frame00008_PROTOCOL_REVIEW.md`). Readiness only.

#### Evidence Quality
Digest, binding, validators and tests reproduced.

#### Simplicity
No added cells.

#### Missing Cases
Carried as residual conditions.

#### Required Changes
None.

#### Optional Improvements
The Driver trimmed a stray transmittal line from the archived V1 review file (content otherwise
byte-identical), as the reviewer recommended.

### Handoff (2026-09-28, session pause before execution)

#### Objective
Pause at the user's request after the protocol approval; execution is the next step.

#### Reviewed state
Task `experiments/tasks/20260928_field_only_portfolio_stage_frame00008.json` is `ready` with the approved digest
`345e57a95cac3edb93fa7cc56d7a0ec43edb9d398f08da4e895583d9b32dfd68` and source binding 119 files
`2848f1d6695c69680fd8a724dc8d1408d33ad92d031e5c0ae3fa061e98831b6e`; merged to `main` by user request.

#### Changes
Approval recorded; no run root exists; nothing executed.

#### Evidence
Round-2 approval; 7 protocol tests pass; `./scripts/verify.sh` exit 0 on this tree.

#### Assumptions
The approval stays valid only while every bound file is unchanged (review condition 2).

#### Uncertainties
Runtime of the downscale-4 and 30000-step cells (about 5-13 minutes each on a free GPU).

#### Review Focus
None pending before execution.

#### Protected actions not taken
No init-run, protected execution, audit or default change.

#### Recommended Next Action
When no other compute process uses the GPU (review condition 3), run from a clean tree:
`.venv/bin/python scripts/experiment_contract.py init-run experiments/tasks/20260928_field_only_portfolio_stage_frame00008.json`, then the
frozen `run_command` once. Then results audit (ask the user to authorize the dome-derived preview
payload), render, headless viewer smoke, second render, both gates, EXPERIMENTS/ARA, closeout.
