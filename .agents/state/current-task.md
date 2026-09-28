# Current Task

## Title

Field-only distillation improvement portfolio

## Task ID

RTGS-029

## Role Assignment

- Driver: Claude-Code-Opus-5.5-driver
- Reviewer: Claude-Code-Fable-5.1-reviewer
- Turn: reviewer

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

In review

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
