# Current Task

## Title

Inert gsplat Default opacity reset: evidence, disclosure and opt-in intended reset

## Task ID

RTGS-027

## Role Assignment

- Driver: Claude-Code-Opus-5.5-driver
- Reviewer: Claude-Code-Fable-5.1-reviewer
- Turn: reviewer

## Mode

Implement

## Risk

Standard

## Maturity

- Target: CPU-contracted
- Reached: CPU-contracted

## Goal

Record that gsplat 1.5.3 `DefaultStrategy.step_post_backward` never executes its opacity reset
(`step % self.reset_every == 0 & step > 0` is a chained comparison ending in `0 > 0`), correct the
architecture wording, and provide an opt-in intended reset usable through the existing
`parameter_step_callback` seam without changing defaults or historical configurations.

## Motivation

Found by the Fable 5.1 RTGS-026 prospective review and confirmed by the Driver; the user asked on
2026-09-27 to handle it. The repository's geometric-arena path already performs the reset, while
the default dynamic path relies on the inert upstream call.

## Success Criteria

- A CPU test demonstrates the upstream precedence defect independent of gsplat installation.
- `rtgs.optim.strategies.IntendedOpacityReset` clamps opacity logits and zeroes the opacity Adam
  moments exactly at the intended steps, CPU-tested; a CUDA test exercises it inside gsplat-default
  training; `chain_parameter_callbacks` composes it with other seam users.
- ARCHITECTURE wording, a staging observation and the experiment log disclose the inert reset.
- No default, TrainConfig or DensityConfig field changes; historical protocol tests stay green.

## Constraints

CPU-first imports; no gsplat import in the new CPU path; no historical evidence edits.

## Non-Goals

Changing the default, patching the installed gsplat, rerunning historical experiments.

## Selected Skills

- `rtgs-core`
- `rtgs-task-workflow`
- `rtgs-review`
- `rtgs-docs-sync`
- `rtgs-verify`

## Experiment Contract

None

## Current Evidence

RTGS-026 review V1 (R1) and audit; gsplat 1.5.3 `strategy/default.py` source; the arena path in
`rtgs.optim.strategies.GsplatStrategyController._arena_default_step`.

## Minimal Plan

1. Tests for the upstream defect and the intended reset.
2. Implement the opt-in reset and callback chaining.
3. Docs, staging observation, verify, self-review.

## Status

In review

## Human Decisions

### Question
How to handle the inert gsplat opacity reset?
### Options
Document only; opt-in intended reset; change the default.
### Recommendation
Document plus opt-in intended reset; defaults unchanged.
### Decision
User asked to handle it (2026-09-27: "mach alle 3").
### Date
2026-09-27

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### Handoff (2026-09-27, for independent code review)

#### Objective
Independent review of the RTGS-027 diff.

#### Reviewed state
Branch `rtgs-027-gsplat-opacity-reset`, diff against `main` at `271cce5`.

#### Changes
`upstream_default_reset_fires`, `IntendedOpacityReset`, `chain_parameter_callbacks` in
`rtgs.optim.strategies`; `tests/test_gsplat_opacity_reset.py`; ARCHITECTURE wording; O182;
experiment-log note.

#### Evidence
6 tests pass on the local GPU (incl. gsplat-default training with the reset firing at step 20);
5 pass and 1 skips on CPU; `./scripts/verify.sh` exit 0.

#### Assumptions
Intended semantics follow gsplat `reset_opa` (clamp to `2 * prune_opa`, zero opacity moments)
within `0 < step < refine_stop_iter`, as in the repository's arena path.

#### Uncertainties
No quality effect measured.

#### Review Focus
Semantics match upstream intent and the arena path; no default/config change; CPU-first imports.

#### Protected actions not taken
No default change, no merge.

#### Recommended Next Action
Fable 5.1 code review.
