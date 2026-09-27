# Current Task

## Title

Colour quality of field-only distillation: training budget x intended opacity reset

## Task ID

RTGS-028

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

Test whether a 30000-step budget and the intended gsplat Default opacity reset (RTGS-027) improve
held-out colour inside the mask for field-only distillation, and how much of the gap to
photograph supervision remains.

## Motivation

RTGS-025/026 showed field-only distillation 0.4-0.6 dB below photograph supervision, with floaters
already rare; all runs used 8000 steps and, unknowingly, no opacity reset. The user approved this
follow-up on 2026-09-27 ("mach alle 3").

## Success Criteria

- A registered, independently reviewed protocol with relative per-seed rules (H1 budget, H2 reset).
- One audited calibrated run with report, viewer receipt and bundle gates; ARA and docs updated.

## Constraints

CPU-first imports; no default change; held-out views evaluation-only; frame 00008 is exposed.

## Non-Goals

Relocation, hull initialization, resolution changes, default promotion.

## Selected Skills

- `rtgs-core`
- `rtgs-task-workflow`
- `rtgs-experiment`
- `realtime-gs-results-audit`
- `rtgs-review`
- `rtgs-docs-sync`
- `rtgs-verify`

## Experiment Contract

experiments/tasks/20260927_color_budget_reset_stage_frame00008.json

## Current Evidence

C49, C51, O182; RTGS-027 `IntendedOpacityReset`.

## Minimal Plan

1. Driver, report, tests and protocol; non-protocol smoke.
2. Fable 5.1 prospective review; one run; results audit; report and gates.

## Status

In progress

## Human Decisions

### Question
Which follow-up should address colour quality?
### Options
Budget, opacity reset, resolution, hull initialization without relocation.
### Recommendation
Budget x intended opacity reset factorial with photograph references; relocation and hull
initialization dropped (no effect in RTGS-025/026).
### Decision
User approved the follow-up ("mach alle 3").
### Date
2026-09-27

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### Handoff (2026-09-27, protocol and driver for prospective review)

#### Objective
Fable 5.1 prospective review of the RTGS-028 protocol.

#### Reviewed state
Branch `rtgs-028-color-budget-reset` (commit adding this entry); digest
`7a55197b8d83df6c241b41fd9037acc4e6cd4f93bc101e26e463231e68b096b9`, binding 120 files
`28b7fb80b74d1b520e9ec176e0e9161d6c81ec699868340c43f6736158486af5`.

#### Changes
Driver/report/tests derived from the accepted RTGS-026 retry driver (exact CPU-index decoding,
guards, coordinator re-entry refusal) without hull or relocation; conditions cross budget (8000 /
30000 steps, densification stop 6000 / 15000) with the RTGS-027 intended reset; photograph
references at 8000 and at 30000 with reset; RTGS-025 split (22 / 4 views); seeds 9461-9463.

#### Evidence
6 CPU protocol tests pass; non-protocol GPU smoke passed every phase with resets at the expected
gsplat iterations; `./scripts/verify.sh` exit 0.

#### Assumptions
The 30000-step arm follows the 3DGS densification convention (stop 15000).

#### Uncertainties
Runtime of 30000-step cells on the shared GPU (estimated about 10 minutes each).

#### Review Focus
Relative rules, budget/densification coupling, reset wiring, leakage boundary.

#### Protected actions not taken
No init-run or protected execution.

#### Recommended Next Action
Fable 5.1 prospective review.

### Review (2026-09-27, Fable 5.1 prospective protocol review)

#### Verdict
Accepted

#### Self-reviewed
No

#### Correctness
Approved digest `7a55197b8d83df6c241b41fd9037acc4e6cd4f93bc101e26e463231e68b096b9` with residual
execution conditions 1-8 (verbatim: `experiments/reviews/20260927_color_budget_reset_stage_frame00008_PROTOCOL_REVIEW.md`).
Readiness only, not results.

#### Evidence Quality
Digest and source-binding aggregate reproduced; 14 CPU tests pass.

#### Simplicity
Minimal factorial for the stated questions.

#### Missing Cases
Means-LR horizon and 100000-cap interaction carried as disclosures (condition 5).

#### Required Changes
None.

#### Optional Improvements
A 30000-step arm with densification stop 6000 in a later task.

Driver pre-start checks: condition 4 canary passed on the GPU (9 tests, gsplat 1.5.3); condition 3
prior 8000-step cell walls were 142-187 s (below 700 s), but the GPU is currently shared with an
unrelated heavy process (path-wm experiments.latent_agent, pid 4057089, ~1.6 GB, ~46% util), so
the run waits until that process exits.
